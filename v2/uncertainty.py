"""Uncertainty scores for selective prediction."""

from __future__ import annotations

import numpy as np


def normalize_proba(proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba, dtype=float)
    denom = proba.sum(axis=1, keepdims=True)
    denom = np.where(denom <= 0.0, 1.0, denom)
    return proba / denom


def score_predictions(proba: np.ndarray, score: str) -> np.ndarray:
    """Return a confidence-like score where larger means more reliable."""

    proba = normalize_proba(proba)
    score = score.lower().replace("_", "-")
    if score in {"max-prob", "max_softmax", "msp"}:
        return proba.max(axis=1)
    if score == "entropy":
        entropy = -np.sum(proba * np.log(proba + 1e-12), axis=1)
        return -entropy
    if score == "margin":
        sorted_p = np.sort(proba, axis=1)
        return sorted_p[:, -1] - sorted_p[:, -2]
    raise ValueError(f"Unknown uncertainty score: {score}")


def mask_for_coverage(proba: np.ndarray, coverage: float, score: str) -> tuple[np.ndarray, float]:
    scores = score_predictions(proba, score)
    n_keep = int(np.ceil(float(coverage) * scores.size))
    n_keep = min(max(n_keep, 1), scores.size)
    order = np.argsort(scores)[::-1]
    mask = np.zeros(scores.size, dtype=bool)
    mask[order[:n_keep]] = True
    return mask, float(scores[order[n_keep - 1]])

