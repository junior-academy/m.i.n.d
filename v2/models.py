"""Model definitions and fitting utilities for M.I.N.D. v2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from decode import fit_predict_subject

from .calibration import TemperatureScaler, logits_from_proba, softmax
from .config import DEEP_BATCH_SIZE, DEEP_EPOCHS, DEEP_LR, RANDOM_SEED, VALIDATION_SIZE


@dataclass
class ModelOutput:
    name: str
    y_true: np.ndarray
    proba: np.ndarray
    val_y: np.ndarray | None = None
    val_proba: np.ndarray | None = None
    val_logits: np.ndarray | None = None
    calibrated: bool = False


def _require_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Deep v2 models require PyTorch. Install `requirements-eegnet.txt`."
        ) from exc
    return torch, nn, DataLoader, TensorDataset


def _standardize(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (X_train - mean) / std, (X_val - mean) / std, (X_test - mean) / std


def _split_train_val(X: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(y.size)
    tr, va = train_test_split(idx, test_size=VALIDATION_SIZE, random_state=seed, stratify=y)
    return X[tr], y[tr], X[va], y[va]


def fit_v1_lda(
    *,
    subject: str | int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> ModelOutput:
    """Fit the v1 calibrated LDA baseline."""

    pred = fit_predict_subject(
        subject=int(subject) if str(subject).isdigit() else 0,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_names=("LDA",),
    )
    return ModelOutput(name="v1_lda", y_true=pred.y_true, proba=pred.proba["LDA"], calibrated=True)


def _build_eegnet(torch, nn, n_channels: int, n_times: int, n_classes: int, dropout: float = 0.5):
    class EEGNet(nn.Module):
        def __init__(self):
            super().__init__()
            f1, depth = 8, 2
            f2 = f1 * depth
            kernel = min(64, max(16, n_times // 8))
            self.features = nn.Sequential(
                nn.Conv2d(1, f1, kernel_size=(1, kernel), padding=(0, kernel // 2), bias=False),
                nn.BatchNorm2d(f1),
                nn.Conv2d(f1, f2, kernel_size=(n_channels, 1), groups=f1, bias=False),
                nn.BatchNorm2d(f2),
                nn.ELU(),
                nn.AvgPool2d((1, 4)),
                nn.Dropout(dropout),
                nn.Conv2d(f2, f2, kernel_size=(1, 16), padding=(0, 8), groups=f2, bias=False),
                nn.Conv2d(f2, f2, kernel_size=(1, 1), bias=False),
                nn.BatchNorm2d(f2),
                nn.ELU(),
                nn.AvgPool2d((1, 8)),
                nn.Dropout(dropout),
                nn.Flatten(),
            )
            with torch.no_grad():
                n_features = int(self.features(torch.zeros(1, 1, n_channels, n_times)).shape[1])
            self.classifier = nn.Linear(n_features, n_classes)

        def forward(self, x):
            return self.classifier(self.features(x))

    return EEGNet()


def _build_spatial_attention(torch, nn, n_channels: int, n_times: int, n_classes: int, dropout: float = 0.5):
    class SpatialAttentionEEG(nn.Module):
        def __init__(self):
            super().__init__()
            self.channel_logits = nn.Parameter(torch.zeros(n_channels))
            self.temporal = nn.Sequential(
                nn.Conv1d(n_channels, 32, kernel_size=25, padding=12, bias=False),
                nn.BatchNorm1d(32),
                nn.ELU(),
                nn.AvgPool1d(4),
                nn.Dropout(dropout),
                nn.Conv1d(32, 64, kernel_size=15, padding=7, bias=False),
                nn.BatchNorm1d(64),
                nn.ELU(),
                nn.AdaptiveAvgPool1d(8),
                nn.Flatten(),
            )
            self.classifier = nn.Linear(64 * 8, n_classes)

        def forward(self, x):
            x = x.squeeze(1)
            weights = torch.softmax(self.channel_logits, dim=0).view(1, -1, 1) * x.shape[1]
            return self.classifier(self.temporal(x * weights))

    return SpatialAttentionEEG()


def _train_torch_model(
    *,
    model_kind: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    n_classes: int,
    epochs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    torch, nn, DataLoader, TensorDataset = _require_torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    X_train, X_val, X_test = _standardize(X_train, X_val, X_test)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_kind == "eegnet":
        model = _build_eegnet(torch, nn, X_train.shape[1], X_train.shape[2], n_classes)
    elif model_kind == "spatial":
        model = _build_spatial_attention(torch, nn, X_train.shape[1], X_train.shape[2], n_classes)
    else:
        raise ValueError(f"Unknown deep model: {model_kind}")
    model.to(device)

    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train[:, None, :, :], dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        ),
        batch_size=DEEP_BATCH_SIZE,
        shuffle=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=DEEP_LR, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    best_state = None
    best_val = float("inf")
    stale = 0
    X_val_t = torch.tensor(X_val[:, None, :, :], dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(X_val_t), y_val_t).item())
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 12:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_t).cpu().numpy()
        test_logits = model(torch.tensor(X_test[:, None, :, :], dtype=torch.float32).to(device)).cpu().numpy()
    return val_logits, test_logits, softmax(test_logits)


def fit_deep_model(
    *,
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    calibrate: bool,
    epochs: int = DEEP_EPOCHS,
    seed: int = RANDOM_SEED,
) -> ModelOutput:
    """Fit EEGNet or spatial-attention EEG model with optional temperature scaling."""

    le = LabelEncoder()
    y_enc = le.fit_transform(y_train).astype(np.int64)
    y_test_enc = le.transform(y_test).astype(np.int64)
    X_tr, y_tr, X_val, y_val = _split_train_val(X_train.astype(np.float32), y_enc, seed)
    val_logits, test_logits, proba = _train_torch_model(
        model_kind=name,
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test.astype(np.float32),
        n_classes=len(le.classes_),
        epochs=epochs,
        seed=seed,
    )
    val_proba = softmax(val_logits)
    if calibrate:
        scaler = TemperatureScaler().fit(val_logits, y_val)
        proba = scaler.predict_proba(test_logits)
        val_proba = scaler.predict_proba(val_logits)
    return ModelOutput(
        name=f"{name}_{'calibrated' if calibrate else 'uncalibrated'}",
        y_true=y_test_enc,
        proba=proba,
        val_y=y_val,
        val_proba=val_proba,
        val_logits=val_logits,
        calibrated=calibrate,
    )


def fit_deep_model_variants(
    *,
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    include_calibrated: bool,
    epochs: int = DEEP_EPOCHS,
    seed: int = RANDOM_SEED,
) -> list[ModelOutput]:
    """Train one deep model once and return uncalibrated/calibrated variants."""

    le = LabelEncoder()
    y_enc = le.fit_transform(y_train).astype(np.int64)
    y_test_enc = le.transform(y_test).astype(np.int64)
    X_tr, y_tr, X_val, y_val = _split_train_val(X_train.astype(np.float32), y_enc, seed)
    val_logits, test_logits, proba_uncal = _train_torch_model(
        model_kind=name,
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test.astype(np.float32),
        n_classes=len(le.classes_),
        epochs=epochs,
        seed=seed,
    )
    val_uncal = softmax(val_logits)
    outputs = [
        ModelOutput(
            name=f"{name}_uncalibrated",
            y_true=y_test_enc,
            proba=proba_uncal,
            val_y=y_val,
            val_proba=val_uncal,
            val_logits=val_logits,
            calibrated=False,
        )
    ]
    if include_calibrated:
        scaler = TemperatureScaler().fit(val_logits, y_val)
        outputs.append(
            ModelOutput(
                name=f"{name}_calibrated",
                y_true=y_test_enc,
                proba=scaler.predict_proba(test_logits),
                val_y=y_val,
                val_proba=scaler.predict_proba(val_logits),
                val_logits=val_logits,
                calibrated=True,
            )
        )
    return outputs


def equal_ensemble(name: str, outputs: list[ModelOutput], post_calibrate: bool = False) -> ModelOutput:
    """Average probabilities. Optionally temperature-scale log probabilities on validation."""

    proba = np.mean([out.proba for out in outputs], axis=0)
    val_proba = None
    val_y = None
    calibrated = False
    if all(out.val_proba is not None for out in outputs):
        val_proba = np.mean([out.val_proba for out in outputs if out.val_proba is not None], axis=0)
        val_y = outputs[0].val_y
    if post_calibrate and val_proba is not None and val_y is not None:
        scaler = TemperatureScaler().fit(logits_from_proba(val_proba), val_y)
        proba = scaler.predict_proba(logits_from_proba(proba))
        val_proba = scaler.predict_proba(logits_from_proba(val_proba))
        calibrated = True
    return ModelOutput(
        name=name,
        y_true=outputs[0].y_true,
        proba=proba,
        val_y=val_y,
        val_proba=val_proba,
        calibrated=calibrated,
    )


def validation_weighted_ensemble(name: str, outputs: list[ModelOutput], post_calibrate: bool = False) -> ModelOutput:
    """Weight models by validation accuracy from training-session validation only."""

    if not all(out.val_proba is not None and out.val_y is not None for out in outputs):
        return equal_ensemble(name, outputs, post_calibrate=post_calibrate)
    weights = []
    for out in outputs:
        acc = np.mean(out.val_proba.argmax(axis=1) == out.val_y)
        weights.append(max(float(acc), 1e-3))
    weights_arr = np.asarray(weights, dtype=float)
    weights_arr = weights_arr / weights_arr.sum()
    proba = np.tensordot(weights_arr, np.stack([out.proba for out in outputs], axis=0), axes=(0, 0))
    val_proba = np.tensordot(weights_arr, np.stack([out.val_proba for out in outputs], axis=0), axes=(0, 0))
    val_y = outputs[0].val_y
    calibrated = False
    if post_calibrate:
        scaler = TemperatureScaler().fit(logits_from_proba(val_proba), val_y)
        proba = scaler.predict_proba(logits_from_proba(proba))
        val_proba = scaler.predict_proba(logits_from_proba(val_proba))
        calibrated = True
    return ModelOutput(name=name, y_true=outputs[0].y_true, proba=proba, val_y=val_y, val_proba=val_proba, calibrated=calibrated)
