from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _betacf(a: float, b: float, x: float) -> float:
    """
    Continued fraction approximation for the incomplete beta function.
    Numerical Recipes style implementation (stable for our small df use-cases).
    """
    max_iter = 200
    eps = 3e-12
    fpmin = 1e-30

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0

    c = 1.0
    d = 1.0 - (qab * x) / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d

    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = (m * (b - m) * x) / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c

        aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta

        if abs(delta - 1.0) < eps:
            break

    return float(h)


def betainc_reg(a: float, b: float, x: float) -> float:
    """
    Regularized incomplete beta I_x(a,b).
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    ln_bt = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    bt = math.exp(ln_bt)

    # Use symmetry to improve convergence
    if x < (a + 1.0) / (a + b + 2.0):
        return float(bt * _betacf(a, b, x) / a)
    return float(1.0 - (bt * _betacf(b, a, 1.0 - x) / b))


def student_t_cdf(t: float, df: int) -> float:
    """
    CDF of Student's t distribution with `df` degrees of freedom.
    """
    if df <= 0:
        return float("nan")
    if t == 0.0:
        return 0.5
    v = float(df)
    x = v / (v + float(t) * float(t))
    ib = betainc_reg(v / 2.0, 0.5, x)
    if t > 0:
        return float(1.0 - 0.5 * ib)
    return float(0.5 * ib)


def student_t_pvalue_two_sided(t_stat: float, df: int) -> float:
    if not math.isfinite(t_stat) or df <= 0:
        return float("nan")
    cdf = student_t_cdf(abs(float(t_stat)), df=int(df))
    return float(max(0.0, min(1.0, 2.0 * (1.0 - cdf))))


def f_cdf(x: float, dfn: int, dfd: int) -> float:
    """
    CDF of F distribution using regularized incomplete beta.
    """
    if x <= 0.0:
        return 0.0
    if dfn <= 0 or dfd <= 0:
        return float("nan")
    d1 = float(dfn)
    d2 = float(dfd)
    z = (d1 * float(x)) / (d1 * float(x) + d2)
    return float(betainc_reg(d1 / 2.0, d2 / 2.0, z))


def f_pvalue_right_tail(f_stat: float, dfn: int, dfd: int) -> float:
    if not math.isfinite(f_stat):
        return float("nan")
    return float(max(0.0, min(1.0, 1.0 - f_cdf(float(f_stat), int(dfn), int(dfd)))))


@dataclass(frozen=True)
class PairedTResult:
    statistic: float
    pvalue: float
    df: int


def paired_ttest_rel(a: np.ndarray, b: np.ndarray) -> PairedTResult:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    diffs = a[mask] - b[mask]
    n = int(diffs.size)
    if n < 2:
        return PairedTResult(statistic=float("nan"), pvalue=float("nan"), df=max(0, n - 1))
    mean = float(diffs.mean())
    sd = float(diffs.std(ddof=1))
    if sd == 0.0:
        t_stat = float("inf") if mean > 0 else float("-inf") if mean < 0 else 0.0
        p = 0.0 if mean != 0.0 else 1.0
        return PairedTResult(statistic=t_stat, pvalue=p, df=n - 1)
    t_stat = mean / (sd / (n**0.5))
    p = student_t_pvalue_two_sided(t_stat, df=n - 1)
    return PairedTResult(statistic=float(t_stat), pvalue=float(p), df=n - 1)


@dataclass(frozen=True)
class LeveneResult:
    statistic: float
    pvalue: float
    dfn: int
    dfd: int


def levene_two_groups(a: np.ndarray, b: np.ndarray) -> LeveneResult:
    """
    Levene's test (center=median) for two groups.
    Returns F statistic and right-tail p-value.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    n1 = int(a.size)
    n2 = int(b.size)
    if n1 < 2 or n2 < 2:
        return LeveneResult(statistic=float("nan"), pvalue=float("nan"), dfn=1, dfd=max(0, n1 + n2 - 2))

    z1 = np.abs(a - np.median(a))
    z2 = np.abs(b - np.median(b))
    zbar1 = float(z1.mean())
    zbar2 = float(z2.mean())
    zbar = float(np.concatenate([z1, z2]).mean())

    # One-way ANOVA on z's
    ss_between = n1 * (zbar1 - zbar) ** 2 + n2 * (zbar2 - zbar) ** 2
    ss_within = float(((z1 - zbar1) ** 2).sum() + ((z2 - zbar2) ** 2).sum())

    dfn = 1
    dfd = n1 + n2 - 2
    if ss_within == 0.0:
        f_stat = float("inf") if ss_between > 0 else 0.0
        p = 0.0 if ss_between > 0 else 1.0
        return LeveneResult(statistic=f_stat, pvalue=p, dfn=dfn, dfd=dfd)

    ms_between = ss_between / dfn
    ms_within = ss_within / dfd
    f_stat = ms_between / ms_within
    p = f_pvalue_right_tail(f_stat, dfn=dfn, dfd=dfd)
    return LeveneResult(statistic=float(f_stat), pvalue=float(p), dfn=dfn, dfd=dfd)

