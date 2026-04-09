from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class DebounceConfig:
    """
    Debounced confidence gate configuration.

    - Enter FIRE when at least `k` of the last `n` trials are above `t_on` AND
      agree on the same predicted class (mode).
    - Stay in FIRE until confidence drops below `t_off` (hysteresis).
    - When entering FIRE, we latch the mode class and keep it until we exit.
    """

    t_on: float
    t_off: float
    k: int = 3
    n: int = 5

    def __post_init__(self) -> None:
        if not (0.0 <= float(self.t_off) <= 1.0 and 0.0 <= float(self.t_on) <= 1.0):
            raise ValueError("t_on and t_off must be in [0, 1]")
        if float(self.t_off) > float(self.t_on):
            raise ValueError("Expected t_off <= t_on for hysteresis")
        if int(self.n) < 1:
            raise ValueError("n must be >= 1")
        if int(self.k) < 1 or int(self.k) > int(self.n):
            raise ValueError("k must satisfy 1 <= k <= n")


def _mode_int(values: Iterable[int]) -> Optional[int]:
    vals = list(values)
    if not vals:
        return None
    c = Counter(vals)
    # deterministic tie-break: smaller class id
    best = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return int(best)


def debounced_gate(
    *,
    p_ens: np.ndarray,
    y_hat: Optional[np.ndarray] = None,
    cfg: DebounceConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute (fired_mask, latched_pred) from per-trial ensemble probabilities.

    Returns:
      fired: bool array, shape (N,)
      pred: int array, shape (N,), latched class when fired, else -1
    """
    p_ens = np.asarray(p_ens, dtype=float)
    if p_ens.ndim != 2:
        raise ValueError("p_ens must be 2D (n_trials, n_classes)")
    n_trials, _n_classes = p_ens.shape
    if y_hat is None:
        y_hat = p_ens.argmax(axis=1)
    y_hat = np.asarray(y_hat, dtype=int)
    if y_hat.shape != (n_trials,):
        raise ValueError("y_hat must be shape (n_trials,)")

    p_max = np.max(p_ens, axis=1)

    fired = np.zeros((n_trials,), dtype=bool)
    pred = np.full((n_trials,), -1, dtype=int)

    # Sliding window over "entry" condition.
    window: deque[Tuple[float, int]] = deque(maxlen=int(cfg.n))
    state_on = False
    latched: Optional[int] = None

    for i in range(n_trials):
        window.append((float(p_max[i]), int(y_hat[i])))

        if state_on:
            # Exit condition.
            if float(p_max[i]) < float(cfg.t_off):
                state_on = False
                latched = None
        else:
            # Entry condition: k-of-n above t_on, same class (mode).
            above = [c for (pm, c) in window if pm >= float(cfg.t_on)]
            if above:
                mode = _mode_int(above)
                if mode is not None and sum(1 for c in above if c == mode) >= int(cfg.k):
                    state_on = True
                    latched = int(mode)

        fired[i] = state_on
        pred[i] = int(latched) if (state_on and latched is not None) else -1

    return fired, pred


def toggle_rate(fired: np.ndarray) -> float:
    fired = np.asarray(fired, dtype=bool)
    if fired.size < 2:
        return float("nan")
    return float(np.mean(fired[1:] != fired[:-1]))


def wrong_fire_rate_all(y_true: np.ndarray, y_pred: np.ndarray, fired: np.ndarray) -> float:
    """
    Fraction of all trials that are wrong and fired.
    This is a useful safety proxy (false activation rate per attempt).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    fired = np.asarray(fired, dtype=bool)
    if y_true.shape != y_pred.shape or y_true.shape != fired.shape:
        raise ValueError("y_true, y_pred, fired must have the same shape")
    if y_true.size == 0:
        return float("nan")
    wrong_and_fired = fired & (y_pred != y_true)
    return float(np.mean(wrong_and_fired))
