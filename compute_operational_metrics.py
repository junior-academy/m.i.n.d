from __future__ import annotations

from collections import Counter, deque
from pathlib import Path
import time

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "outputs" / "key_numbers"
MODEL_DIR = (
    BASE_DIR
    / "outputs"
    / "ensemble_v2"
    / "models-LDA_SVM__weights-baseline_subject__feat-fbcsp__ens-softvote__tune-none__cal-sigmoid"
)


def kappa_from_labels(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.size == 0:
        return float("nan")

    labels = np.unique(np.concatenate([y_true, y_pred]))
    idx = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=float)

    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1.0

    n = cm.sum()
    if n == 0:
        return float("nan")

    po = float(np.trace(cm) / n)
    pe = float((cm.sum(axis=1) * cm.sum(axis=0)).sum() / (n * n))
    if not np.isfinite(pe) or pe >= 1.0:
        return float("nan")
    return float((po - pe) / (1.0 - pe))


def mode_int(values: list[int]) -> int:
    c = Counter(values)
    return int(sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0])


def debounced_gate(
    p_ens: np.ndarray,
    y_hat: np.ndarray,
    t_on: float = 0.80,
    t_off: float = 0.75,
    k: int = 3,
    n: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    p_ens = np.asarray(p_ens, dtype=float)
    y_hat = np.asarray(y_hat, dtype=int)
    p_max = p_ens.max(axis=1)

    fired = np.zeros(len(y_hat), dtype=bool)
    pred = np.full(len(y_hat), -1, dtype=int)

    window: deque[tuple[float, int]] = deque(maxlen=n)
    state_on = False
    latched: int | None = None

    for i in range(len(y_hat)):
        window.append((float(p_max[i]), int(y_hat[i])))

        if state_on:
            if float(p_max[i]) < t_off:
                state_on = False
                latched = None
        else:
            above = [c for pm, c in window if pm >= t_on]
            if above:
                m = mode_int(above)
                if sum(1 for c in above if c == m) >= k:
                    state_on = True
                    latched = m

        fired[i] = state_on
        pred[i] = int(latched) if (state_on and latched is not None) else -1

    return fired, pred


def main() -> None:
    files = sorted(MODEL_DIR.glob("predictions_subject_*.csv"))
    if not files:
        raise FileNotFoundError(f"No prediction files found under {MODEL_DIR}")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    threshold = 0.60
    committed = df[df["ensemble_max_prob"] >= threshold].copy()

    kappa_global = kappa_from_labels(
        committed["y_true"].to_numpy(dtype=int),
        committed["ensemble_pred"].to_numpy(dtype=int),
    )

    per_subject_kappa = []
    for _sid, g in df.groupby("subject"):
        gc = g[g["ensemble_max_prob"] >= threshold]
        if len(gc) >= 2:
            per_subject_kappa.append(
                kappa_from_labels(gc["y_true"].to_numpy(dtype=int), gc["ensemble_pred"].to_numpy(dtype=int))
            )

    pcols = ["p_ens_c0", "p_ens_c1", "p_ens_c2", "p_ens_c3"]
    loops = 2000
    total_trials = 0
    t0 = time.perf_counter()
    for _ in range(loops):
        for _sid, g in df.groupby("subject"):
            p = g[pcols].to_numpy(dtype=float)
            y = g["ensemble_pred"].to_numpy(dtype=int)
            debounced_gate(p, y, t_on=0.80, t_off=0.75, k=3, n=5)
            total_trials += len(g)
    elapsed = time.perf_counter() - t0
    policy_us_per_trial = (elapsed * 1e6) / max(total_trials, 1)

    rows = [
        {"metric": "kappa_committed_global_t0.60", "value": float(kappa_global)},
        {
            "metric": "kappa_committed_mean_subject_t0.60",
            "value": float(np.nanmean(per_subject_kappa)) if per_subject_kappa else float("nan"),
        },
        {"metric": "committed_trials_t0.60", "value": int(len(committed))},
        {"metric": "total_trials", "value": int(len(df))},
        {"metric": "coverage_t0.60", "value": float(len(committed) / len(df))},
        {
            "metric": "policy_latency_us_per_trial_software_only",
            "value": float(policy_us_per_trial),
        },
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "operational_metrics.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
