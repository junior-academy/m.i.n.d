"""Optional EEGNet baseline for comparison against the reliability pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


@dataclass
class EEGNetConfig:
    epochs: int = 60
    batch_size: int = 32
    lr: float = 1e-3
    dropout: float = 0.5
    patience: int = 12
    random_state: int = 42


def _require_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "EEGNet requires PyTorch. Install it with `python3 -m pip install torch` "
            "or use `requirements-eegnet.txt`."
        ) from exc
    return torch, nn, DataLoader, TensorDataset


def _standardize_epochs(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (X_train - mean) / std, (X_test - mean) / std


def _build_eegnet(torch, nn, n_channels: int, n_times: int, n_classes: int, dropout: float):
    class EEGNet(nn.Module):
        def __init__(self):
            super().__init__()
            f1 = 8
            depth = 2
            f2 = f1 * depth
            kernel = min(64, max(16, n_times // 8))
            self.features = nn.Sequential(
                nn.Conv2d(1, f1, kernel_size=(1, kernel), padding=(0, kernel // 2), bias=False),
                nn.BatchNorm2d(f1),
                nn.Conv2d(f1, f2, kernel_size=(n_channels, 1), groups=f1, bias=False),
                nn.BatchNorm2d(f2),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, 4)),
                nn.Dropout(dropout),
                nn.Conv2d(f2, f2, kernel_size=(1, 16), padding=(0, 8), groups=f2, bias=False),
                nn.Conv2d(f2, f2, kernel_size=(1, 1), bias=False),
                nn.BatchNorm2d(f2),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, 8)),
                nn.Dropout(dropout),
                nn.Flatten(),
            )
            with torch.no_grad():
                dummy = torch.zeros(1, 1, n_channels, n_times)
                n_features = int(self.features(dummy).shape[1])
            self.classifier = nn.Linear(n_features, n_classes)

        def forward(self, x):
            return self.classifier(self.features(x))

    return EEGNet()


def fit_predict_eegnet(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: EEGNetConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit EEGNet and return encoded y_test plus probability predictions.

    EEGNet is included as a modern deep-learning comparison only. The main
    reliability claim remains the calibrated LDA/SVM ensemble.
    """

    torch, nn, DataLoader, TensorDataset = _require_torch()
    cfg = config or EEGNetConfig()
    torch.manual_seed(cfg.random_state)
    np.random.seed(cfg.random_state)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train).astype(np.int64)
    y_test_enc = le.transform(y_test).astype(np.int64)
    X_train_std, X_test_std = _standardize_epochs(X_train, X_test)

    indices = np.arange(y_train_enc.size)
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=cfg.random_state,
        stratify=y_train_enc,
    )

    X_train_t = torch.tensor(X_train_std[train_idx, None, :, :], dtype=torch.float32)
    y_train_t = torch.tensor(y_train_enc[train_idx], dtype=torch.long)
    X_val_t = torch.tensor(X_train_std[val_idx, None, :, :], dtype=torch.float32)
    y_val_t = torch.tensor(y_train_enc[val_idx], dtype=torch.long)
    X_test_t = torch.tensor(X_test_std[:, None, :, :], dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_eegnet(
        torch,
        nn,
        n_channels=int(X_train_std.shape[1]),
        n_times=int(X_train_std.shape[2]),
        n_classes=int(len(le.classes_)),
        dropout=cfg.dropout,
    ).to(device)

    loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    best_state = None
    best_val = float("inf")
    stale = 0

    for _ in range(cfg.epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t.to(device))
            val_loss = float(loss_fn(val_logits, y_val_t.to(device)).item())
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t.to(device))
        proba = torch.softmax(logits, dim=1).cpu().numpy()
    return y_test_enc, proba
