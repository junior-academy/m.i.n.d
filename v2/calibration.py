"""Temperature scaling for deep models and soft-vote ensembles."""

from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def logits_from_proba(proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba, dtype=float)
    proba = np.clip(proba, 1e-8, 1.0)
    return np.log(proba)


class TemperatureScaler:
    """Scalar temperature fitted on validation logits only."""

    def __init__(self) -> None:
        self.temperature = 1.0

    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> "TemperatureScaler":
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("Temperature scaling requires PyTorch.") from exc

        x = torch.tensor(np.asarray(logits), dtype=torch.float32)
        y = torch.tensor(np.asarray(y_true), dtype=torch.long)
        log_t = torch.nn.Parameter(torch.zeros(()))
        optimizer = torch.optim.LBFGS([log_t], lr=0.05, max_iter=80)
        loss_fn = torch.nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            temp = torch.exp(log_t).clamp(0.05, 20.0)
            loss = loss_fn(x / temp, y)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = float(torch.exp(log_t).detach().cpu().clamp(0.05, 20.0))
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        return softmax(np.asarray(logits, dtype=float) / self.temperature)

