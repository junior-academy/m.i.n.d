from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import pygame

from gating import DebounceConfig, debounced_gate, toggle_rate, wrong_fire_rate_all

def _require_pygame():
    try:
        import pygame 

        return pygame
    except ModuleNotFoundError as e:
        raise SystemExit(
            "Missing dependency: pygame\n\n"
            "Install it in your current environment:\n"
            "  python -m pip install pygame\n"
        ) from e


# ──────────────────────────────────────────────────────────────────────────────
# Styling (dark / cyber-ish)
# ──────────────────────────────────────────────────────────────────────────────
BG = (10, 10, 15)  # #0a0a0f
PANEL = (13, 17, 23)  # #0d1117
PANEL_2 = (18, 24, 33)
CYAN = (0, 245, 255)  # #00f5ff
MAGENTA = (255, 0, 170)  # #ff00aa
TEXT = (232, 244, 248)  # #e8f4f8
TEXT_DIM = (122, 155, 181)  # #7a9bb5
GREEN = (80, 220, 140)
RED = (230, 85, 85)
YELLOW = (240, 210, 120)


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _scanlines(pygame, size: Tuple[int, int]) -> "pygame.Surface":
    w, h = size
    overlay = pygame.Surface((w, h), pygame.SRCALPHA)
    for y in range(0, h, 3):
        overlay.fill((0, 0, 0, 24), pygame.Rect(0, y, w, 1))
    return overlay


def _dot_grid(pygame, size: Tuple[int, int], spacing: int = 18) -> "pygame.Surface":
    w, h = size
    grid = pygame.Surface((w, h), pygame.SRCALPHA)
    dot = pygame.Surface((2, 2), pygame.SRCALPHA)
    dot.fill((0, 245, 255, 14))
    for y in range(0, h, spacing):
        for x in range(0, w, spacing):
            grid.blit(dot, (x, y))
    return grid


def _glow_rect(pygame, surf, rect, color: Tuple[int, int, int], radius: int = 6) -> None:
    # Cheap glow: multiple thin outlines on an alpha surface.
    for w, a in [(6, 24), (3, 48), (1, 110)]:
        glow = pygame.Surface((rect.width + 2 * w, rect.height + 2 * w), pygame.SRCALPHA)
        pygame.draw.rect(
            glow,
            (*color, a),
            pygame.Rect(0, 0, glow.get_width(), glow.get_height()),
            width=1,
            border_radius=radius + w,
        )
        surf.blit(glow, (rect.x - w, rect.y - w))


def _rounded_panel(pygame, surf, rect, *, fill=PANEL, border=CYAN, border_alpha: int = 110, radius: int = 6) -> None:
    panel = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    panel.fill((*fill, 220))
    pygame.draw.rect(panel, (*border, border_alpha), panel.get_rect(), width=1, border_radius=radius)
    surf.blit(panel, rect.topleft)
    _glow_rect(pygame, surf, rect, border, radius=radius)


def _mono_font(pygame, size: int):
    candidates = ["consolas", "couriernew", "menlo", "dejavusansmono", "liberationmono", "monospace"]
    path = None
    for name in candidates:
        p = pygame.font.match_font(name)
        if p:
            path = p
            break
    return pygame.font.Font(path, size) if path else pygame.font.SysFont("Courier New", size)

def _fit_text(font, text: str, max_w: int) -> str:
    if max_w <= 0:
        return ""
    if font.size(text)[0] <= max_w:
        return text
    ell = "…"
    base = text
    while base and font.size(base + ell)[0] > max_w:
        base = base[:-1]
    return (base + ell) if base else ell


def _color_for_class(cls_idx: int) -> Tuple[int, int, int]:
    palette = [
        (0, 245, 255),  # cyan
        (255, 0, 170),  # magenta
        (255, 206, 86),  # yellow
        (54, 162, 235),  # blue
        (153, 102, 255),  # purple
        (75, 192, 192),  # teal
    ]
    return palette[int(cls_idx) % len(palette)]


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────
def _prob_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.startswith("p_ens_c")]

    def _key(c: str) -> int:
        try:
            return int(c.replace("p_ens_c", ""))
        except Exception:
            return 10**9

    return sorted(cols, key=_key)


@dataclass
class RunData:
    name: str
    run_dir: Path
    subject: int
    df: pd.DataFrame
    prob_cols: List[str]


def load_predictions_csv(run_dir: Path, subject: int) -> RunData:
    path = run_dir / f"predictions_subject_{subject}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions CSV: {path}")

    df = pd.read_csv(path)
    required = {"trial_index", "ensemble_pred", "ensemble_max_prob"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    if "y_true" not in df.columns:
        df["y_true"] = np.nan

    prob_cols = _prob_cols(df)
    if not prob_cols:
        raise ValueError(f"{path} missing probability columns like p_ens_c0..p_ens_cK")

    df = df.sort_values("trial_index").reset_index(drop=True)
    return RunData(name=run_dir.name, run_dir=run_dir, subject=int(subject), df=df, prob_cols=prob_cols)


def _safe_int(v) -> Optional[int]:
    try:
        return int(v)
    except Exception:
        return None


def _safe_float(v) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


def _metric_safe_mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return float(x.mean()) if x.size else float("nan")

def _parse_csv_list(arg: str) -> List[str]:
    return [p.strip() for p in arg.split(",") if p.strip()]

def _text_height(font) -> int:
    # pygame doesn't expose a stable "line height" in older versions; get_linesize is best.
    h = getattr(font, "get_linesize", None)
    return int(h()) if callable(h) else int(font.get_height())

def _draw_text(
    pygame,
    surf,
    *,
    font,
    text: str,
    rect,
    color: Tuple[int, int, int] = TEXT,
    align: str = "left",
    valign: str = "top",
    pad: int = 0,
    ellipsize: bool = True,
    clip: bool = True,
) -> None:
    """Draw single-line text inside rect; never overflows (ellipsis + clip)."""
    if rect.width <= 0 or rect.height <= 0:
        return
    max_w = max(0, rect.width - 2 * pad)
    s = _fit_text(font, text, max_w) if ellipsize else text
    img = font.render(s, True, color)
    x = rect.x + pad
    if align == "center":
        x = rect.x + (rect.width - img.get_width()) // 2
    elif align == "right":
        x = rect.right - pad - img.get_width()

    y = rect.y + pad
    if valign == "middle":
        y = rect.y + (rect.height - img.get_height()) // 2
    elif valign == "bottom":
        y = rect.bottom - pad - img.get_height()

    if clip:
        prev = surf.get_clip()
        surf.set_clip(rect)
        surf.blit(img, (x, y))
        surf.set_clip(prev)
    else:
        surf.blit(img, (x, y))

def _wrap_lines(font, text: str, max_w: int) -> List[str]:
    words = str(text).split()
    if not words:
        return [""]
    out: List[str] = []
    cur = words[0]
    for w in words[1:]:
        cand = cur + " " + w
        if font.size(cand)[0] <= max_w:
            cur = cand
        else:
            out.append(cur)
            cur = w
    out.append(cur)
    return out

def _draw_paragraph(
    pygame,
    surf,
    *,
    font,
    text: str,
    rect,
    color: Tuple[int, int, int] = TEXT_DIM,
    pad: int = 0,
    line_gap: int = 2,
    max_lines: Optional[int] = None,
) -> None:
    """Draw wrapped text; clipped to rect."""
    if rect.width <= 0 or rect.height <= 0:
        return
    max_w = max(0, rect.width - 2 * pad)
    lines = _wrap_lines(font, text, max_w)
    if max_lines is not None:
        lines = lines[:max_lines]

    prev = surf.get_clip()
    surf.set_clip(rect)
    lh = _text_height(font) + line_gap
    x = rect.x + pad
    y = rect.y + pad
    for i, line in enumerate(lines):
        if y + lh > rect.bottom:
            break
        img = font.render(_fit_text(font, line, max_w), True, color)
        surf.blit(img, (x, y))
        y += lh
    surf.set_clip(prev)


# ──────────────────────────────────────────────────────────────────────────────
# Simple UI widgets (pure pygame)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Tab:
    label: str
    rect: "object"  # pygame.Rect


class Tabs:
    def __init__(self, pygame, rect, labels: Sequence[str]):
        self.pygame = pygame
        self.rect = rect
        self.labels = [str(l) for l in labels]
        self.short_labels = {
            "OVERVIEW": "OVR",
            "CONTROLS": "CTRL",
            "DATA": "DATA",
            "LOGS": "LOGS",
        }
        self.active = self.labels[0] if self.labels else ""
        self.tabs: List[Tab] = []
        self._use_short = False
        self._layout(font=None, start_x=self.rect.x + 220, end_x=self.rect.right - 200)

    def _layout(self, *, font, start_x: int, end_x: int) -> None:
        self.tabs = []
        x = int(start_x)
        y = self.rect.y + 10
        h = 34

        # Compute a reasonable per-tab width that fits the available space.
        avail = max(0, int(end_x) - int(start_x))
        gap = 10
        if not self.labels:
            return
        if avail < 80:
            # Not enough space: hide tabs rather than overlapping other navbar content.
            return
        n = len(self.labels)
        w = (avail - gap * (n - 1)) // n if n else 0
        w = int(_clamp(w, 90, 160))
        if w * n + gap * (n - 1) > avail and avail > 0:
            # tighten gap if still overflowing
            gap = 6
            w = (avail - gap * (n - 1)) // n
            w = int(_clamp(w, 84, 150))

        for lab in self.labels:
            r = self.pygame.Rect(x, y, w, h)
            self.tabs.append(Tab(label=lab, rect=r))
            x += w + gap

        # If we have a font, slightly expand tab widths to fit text when possible.
        if font is not None and avail > 0:
            # reflow using text widths but still keep within bounds
            x = int(start_x)
            pad_x = 16
            gap = 8
            for t in self.tabs:
                lab = self.short_labels.get(t.label, t.label) if self._use_short else t.label
                text_w = font.size(f"[{lab}]")[0]
                tw = int(_clamp(text_w + pad_x * 2, 90, 180))
                t.rect = self.pygame.Rect(x, y, tw, h)
                x += tw + gap
            if x - gap > end_x:
                # Try shortening labels first.
                if not self._use_short:
                    self._use_short = True
                    self._layout(font=font, start_x=start_x, end_x=end_x)
                else:
                    # fallback to equal widths if still overflowing
                    self._layout(font=None, start_x=start_x, end_x=end_x)

    def layout(self, *, font, start_x: int, end_x: int) -> None:
        self._use_short = False
        self._layout(font=font, start_x=start_x, end_x=end_x)

    def handle_click(self, pos: Tuple[int, int]) -> bool:
        for t in self.tabs:
            if t.rect.collidepoint(pos):
                self.active = t.label
                return True
        return False

    def draw(self, surf, font) -> None:
        for t in self.tabs:
            is_active = t.label == self.active
            fill = PANEL_2 if is_active else PANEL
            border = MAGENTA if is_active else CYAN
            _rounded_panel(self.pygame, surf, t.rect, fill=fill, border=border, border_alpha=150, radius=6)
            lab = self.short_labels.get(t.label, t.label) if self._use_short else t.label
            _draw_text(
                self.pygame,
                surf,
                font=font,
                text=f"[{lab}]",
                rect=t.rect,
                color=(MAGENTA if is_active else TEXT_DIM),
                align="center",
                valign="middle",
                pad=8,
            )


class MiniButton:
    def __init__(self, pygame, rect, label: str):
        self.pygame = pygame
        self.rect = rect
        self.label = label

    def hit(self, pos: Tuple[int, int]) -> bool:
        return self.rect.collidepoint(pos)

    def draw(self, surf, font, *, active: bool = False) -> None:
        fill = PANEL_2 if active else PANEL
        border = MAGENTA if active else CYAN
        _rounded_panel(self.pygame, surf, self.rect, fill=fill, border=border, border_alpha=150, radius=6)
        text = font.render(self.label, True, TEXT)
        surf.blit(text, (self.rect.centerx - text.get_width() // 2, self.rect.centery - text.get_height() // 2))


# ──────────────────────────────────────────────────────────────────────────────
# Prosthetic hand (visual mock)
# ──────────────────────────────────────────────────────────────────────────────
def _draw_hand(pygame, surf, rect, *, gesture: str, fired: bool, title: str = "HAND OUTPUT") -> None:
    # gesture in {"LEFT","RIGHT","FEET","TONGUE","NULL"} (or any string)
    _rounded_panel(pygame, surf, rect, fill=PANEL, border=CYAN, border_alpha=110, radius=8)

    cx, cy = rect.center
    palm = pygame.Rect(0, 0, 120, 120)
    palm.center = (cx, cy + 30)
    color = GREEN if fired else TEXT_DIM
    pygame.draw.rect(surf, (*color, 255), palm, border_radius=14)
    pygame.draw.rect(surf, (*CYAN, 120), palm, width=1, border_radius=14)

    # fingers (5 rectangles); "curl" by shortening.
    finger_w = 18
    gap = 6
    base_y = palm.y - 70
    base_x = palm.x + 6
    lengths = [62, 72, 78, 70, 58]

    g = gesture.upper()
    curl = 0.0
    if g in {"LEFT", "RIGHT"}:
        curl = 0.55
    elif g in {"FEET"}:
        curl = 0.75
    elif g in {"TONGUE"}:
        curl = 0.25
    else:
        curl = 0.0
    if not fired:
        curl = 0.0

    for i in range(5):
        L = int(lengths[i] * (1.0 - curl))
        r = pygame.Rect(base_x + i * (finger_w + gap), base_y + (78 - L), finger_w, L)
        pygame.draw.rect(surf, (*color, 255), r, border_radius=10)
        pygame.draw.rect(surf, (*CYAN, 90), r, width=1, border_radius=10)

    # label
    font = _mono_font(pygame, 16)
    label = f"[{title}]  {('FIRE' if fired else 'HOLD')}  {gesture}"
    surf.blit(font.render(label, True, CYAN if fired else TEXT_DIM), (rect.x + 14, rect.y + 12))


def _safe_int_series(s: pd.Series) -> np.ndarray:
    out = pd.to_numeric(s, errors="coerce").to_numpy()
    out = np.where(np.isfinite(out), out, -1).astype(int)
    return out


def _compute_gate_arrays(
    *,
    df: pd.DataFrame,
    prob_cols: Sequence[str],
    threshold: float,
    off_gap: float,
    k: int,
    n: int,
) -> Dict[str, np.ndarray]:
    p_ens = df[list(prob_cols)].to_numpy(dtype=float)
    y_hat = p_ens.argmax(axis=1).astype(int)

    # Raw threshold gate (no hysteresis, no latch).
    p_max = p_ens.max(axis=1)
    fired_raw = np.isfinite(p_max) & (p_max >= float(threshold))
    pred_raw = np.where(fired_raw, y_hat, -1).astype(int)

    # Debounced gate (hysteresis + k-of-n + latch).
    cfg = DebounceConfig(
        t_on=float(threshold),
        t_off=float(max(0.0, float(threshold) - float(off_gap))),
        k=int(k),
        n=int(n),
    )
    fired_deb, pred_deb = debounced_gate(p_ens=p_ens, y_hat=y_hat, cfg=cfg)

    return {
        "y_hat": y_hat,
        "p_max": p_max,
        "fired_raw": fired_raw.astype(bool),
        "pred_raw": pred_raw.astype(int),
        "fired_deb": fired_deb.astype(bool),
        "pred_deb": pred_deb.astype(int),
    }


def _draw_model_cards(
    pygame,
    surf,
    rect,
    *,
    df: pd.DataFrame,
    trial: int,
    class_names: Sequence[str],
    model_pred_cols: Sequence[str],
) -> None:
    _rounded_panel(pygame, surf, rect, fill=PANEL_2, border=CYAN, border_alpha=120, radius=8)
    font = _mono_font(pygame, 16)
    font_s = _mono_font(pygame, 14)

    def label(idx: Optional[int]) -> str:
        if idx is None:
            return "NA"
        return class_names[idx] if 0 <= idx < len(class_names) else f"Class {idx}"

    surf.blit(font.render("[ MODELS ]", True, CYAN), (rect.x + 14, rect.y + 12))

    cols = list(model_pred_cols)
    if not cols:
        surf.blit(font_s.render("No model *_pred columns found in this run.", True, TEXT_DIM), (rect.x + 14, rect.y + 44))
        return

    upto = df.iloc[: trial + 1]
    y_true = upto["y_true"].to_numpy(dtype=float) if "y_true" in upto.columns else np.full((len(upto),), np.nan)

    # Compute rolling accuracies to label best model.
    acc_by_col = {}
    for col in cols:
        preds = upto[col].to_numpy(dtype=float)
        if np.isfinite(y_true).any() and np.isfinite(preds).any():
            acc_by_col[col] = _metric_safe_mean((y_true == preds).astype(float))
        else:
            acc_by_col[col] = float("nan")

    best_col = None
    best_acc = float("-inf")
    for col, acc in acc_by_col.items():
        if np.isfinite(acc) and acc > best_acc:
            best_acc = acc
            best_col = col

    # Spread-out layout: 2 columns grid (or 1 column if only 1 model)
    gap = 14
    n = len(cols)
    grid_cols = 1 if n == 1 else 2
    grid_rows = (n + grid_cols - 1) // grid_cols
    x0 = rect.x + 14
    y0 = rect.y + 44
    avail_w = rect.width - 28
    avail_h = rect.height - 54
    card_w = (avail_w - gap * (grid_cols - 1)) // grid_cols
    card_h = (avail_h - gap * (grid_rows - 1)) // grid_rows

    for idx, col in enumerate(cols):
        r_i = idx // grid_cols
        c_i = idx % grid_cols
        card = pygame.Rect(x0 + c_i * (card_w + gap), y0 + r_i * (card_h + gap), card_w, card_h)
        is_best = (best_col == col)
        border = MAGENTA if is_best else CYAN
        border_a = 170 if is_best else 90
        _rounded_panel(pygame, surf, card, fill=PANEL, border=border, border_alpha=border_a, radius=10)
        name = col.replace("_pred", "").upper()
        title = name + ("  ★ BEST" if is_best else "")
        title = _fit_text(font_s, title, card.width - 24)
        _draw_text(
            pygame,
            surf,
            font=font_s,
            text=title,
            rect=pygame.Rect(card.x + 10, card.y + 8, card.width - 20, _text_height(font_s) + 2),
            color=(MAGENTA if is_best else TEXT_DIM),
            align="left",
            valign="top",
            pad=0,
        )

        cur_pred = _safe_int(df.iloc[trial].get(col, np.nan))
        cur_true = _safe_int(df.iloc[trial].get("y_true", np.nan))
        correct = (cur_pred is not None) and (cur_true is not None) and (cur_pred == cur_true)
        c = GREEN if correct else (RED if (cur_pred is not None and cur_true is not None) else TEXT_DIM)

        pad = 10
        y = card.y + 8 + _text_height(font_s) + 10

        pred_txt = _fit_text(font, label(cur_pred), card.width - 2 * pad)
        pred_rect = pygame.Rect(card.x + pad, y, card.width - 2 * pad, _text_height(font) + 2)
        _draw_text(pygame, surf, font=font, text=pred_txt, rect=pred_rect, color=c, pad=0)
        y += _text_height(font) + 8

        # Only show true label if we have room.
        true_line = f"true: {label(cur_true)}" if cur_true is not None else "true: NA"
        true_h = _text_height(font_s) + 2
        acc_h = _text_height(font_s) + 2
        remaining = card.bottom - pad - y
        show_true = remaining >= (true_h + acc_h + 6)

        if show_true:
            true_rect = pygame.Rect(card.x + pad, y, card.width - 2 * pad, true_h)
            _draw_text(pygame, surf, font=font_s, text=_fit_text(font_s, true_line, true_rect.width), rect=true_rect, color=TEXT_DIM)
            y += true_h + 6

        acc = acc_by_col.get(col, float("nan"))
        acc_line = f"acc so far: {acc:.3f}" if np.isfinite(acc) else "acc so far: NA"
        acc_rect = pygame.Rect(card.x + pad, y, card.width - 2 * pad, acc_h)
        _draw_text(pygame, surf, font=font_s, text=_fit_text(font_s, acc_line, acc_rect.width), rect=acc_rect, color=TEXT_DIM)

        # Small correctness light
        light = pygame.Rect(card.right - 22, card.y + 14, 10, 10)
        pygame.draw.rect(surf, (*c, 255), light, border_radius=3)
        pygame.draw.rect(surf, (*border, 150), light, width=1, border_radius=3)


# ──────────────────────────────────────────────────────────────────────────────
# Drawing helpers for charts/bars
# ──────────────────────────────────────────────────────────────────────────────
def _draw_bar(pygame, surf, rect, frac: float, *, fg=(80, 200, 120), label: Optional[str] = None, font=None):
    pygame.draw.rect(surf, (30, 32, 40), rect, border_radius=6)
    frac = _clamp(frac, 0.0, 1.0)
    filled = pygame.Rect(rect.x, rect.y, int(rect.width * frac), rect.height)
    pygame.draw.rect(surf, fg, filled, border_radius=6)
    pygame.draw.rect(surf, (*CYAN, 120), rect, width=1, border_radius=6)
    if label and font:
        img = font.render(label, True, TEXT_DIM)
        surf.blit(img, (rect.x, rect.y - 18))


def _draw_line_chart(pygame, surf, rect, ys: Sequence[float], *, threshold: float):
    pygame.draw.rect(surf, (16, 18, 24), rect, border_radius=8)
    pygame.draw.rect(surf, (*CYAN, 90), rect, width=1, border_radius=8)
    if len(ys) < 2:
        return
    n = len(ys)
    xs = np.linspace(rect.x + 6, rect.right - 6, n)

    def y_to_px(v: float) -> int:
        v = _clamp(float(v), 0.0, 1.0)
        return rect.y + int((1.0 - v) * (rect.height - 1))

    pts = [(int(xs[i]), y_to_px(ys[i])) for i in range(n)]
    pygame.draw.lines(surf, CYAN, False, pts, 2)

    # threshold line
    y_thr = y_to_px(threshold)
    pygame.draw.line(surf, (*MAGENTA, 255), (rect.x + 4, y_thr), (rect.right - 4, y_thr), 1)


# ──────────────────────────────────────────────────────────────────────────────
# Main app
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    base_dir = Path(__file__).resolve().parent
    ens_root = base_dir / "outputs" / "ensemble_v2"
    if not ens_root.exists():
        raise SystemExit(f"Missing outputs folder: {ens_root}")

    def latest_dir(glob_pat: str) -> Optional[Path]:
        cands = sorted([p for p in ens_root.glob(glob_pat) if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
        return cands[0] if cands else None

    def discover_run_sets() -> Dict[str, List[Path]]:
        sets: Dict[str, List[Path]] = {}
        latest = latest_dir("models-*")
        main = latest_dir("models-LDA_SVM__*")
        ablation = latest_dir("models-LDA_SVM_RF__*")
        stacking = latest_dir("models-LDA_SVM__*__ens-stacking*")

        if latest:
            sets["Latest"] = [latest]
        if main:
            sets["Main (LDA+SVM)"] = [main]
        if ablation:
            sets["Ablation (LDA+SVM+RF)"] = [ablation]
        if stacking:
            sets["Stacking (meta)"] = [stacking]
        compare = [p for p in [main, ablation, stacking] if p is not None]
        if len(compare) >= 2:
            sets["Compare (cycle TAB)"] = compare  # type: ignore[assignment]
        return sets

    run_sets = discover_run_sets()
    if not run_sets:
        raise SystemExit(f"No run folders found under {ens_root}. Run ensemble_v2 first.")
    run_set_labels = list(run_sets.keys())
    patients = list(range(1, 10))
    class_names = ["Left", "Right", "Feet", "Tongue"]

    pygame = _require_pygame()
    pygame.init()

    W, H = 1280, 800
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("M.I.N.D Visualizer")
    clock = pygame.time.Clock()

    font = _mono_font(pygame, 18)
    font_s = _mono_font(pygame, 14)
    font_b = _mono_font(pygame, 24)

    grid = _dot_grid(pygame, (W, H), spacing=18)
    scan = _scanlines(pygame, (W, H))

    # Layout
    NAV_H = 56
    LEFT_W = 340
    RIGHT_W = 360
    PAD = 18

    nav_rect = pygame.Rect(0, 0, W, NAV_H)
    left_rect = pygame.Rect(0, NAV_H, LEFT_W, H - NAV_H)
    right_rect = pygame.Rect(W - RIGHT_W, NAV_H, RIGHT_W, H - NAV_H)
    main_rect = pygame.Rect(LEFT_W, NAV_H, W - LEFT_W - RIGHT_W, H - NAV_H)

    main_panel = main_rect.inflate(-2 * PAD, -2 * PAD)
    gap = 16
    info_rect = pygame.Rect(main_panel.x, main_panel.y, main_panel.width, 230)
    chart_rect = pygame.Rect(main_panel.x, info_rect.bottom + gap, main_panel.width, 230)
    probs_rect = pygame.Rect(main_panel.x, chart_rect.bottom + gap, main_panel.width, main_panel.bottom - (chart_rect.bottom + gap))

    hand_rect = pygame.Rect(right_rect.x + PAD, right_rect.y + PAD, right_rect.width - 2 * PAD, 300)
    stats_rect = pygame.Rect(
        right_rect.x + PAD,
        hand_rect.bottom + gap,
        right_rect.width - 2 * PAD,
        right_rect.bottom - (hand_rect.bottom + gap + PAD),
    )

    tabs = Tabs(pygame, nav_rect, ["OVERVIEW", "DATA", "LOGS"])

    btn_play = MiniButton(pygame, pygame.Rect(left_rect.x + PAD, left_rect.y + 300, LEFT_W - 2 * PAD, 48), "SPACE  Play/Pause")
    btn_prev = MiniButton(pygame, pygame.Rect(left_rect.x + PAD, left_rect.y + 360, (LEFT_W - 3 * PAD) // 2, 40), "◀ Step")
    btn_next = MiniButton(pygame, pygame.Rect(btn_prev.rect.right + PAD, left_rect.y + 360, (LEFT_W - 3 * PAD) // 2, 40), "Step ▶")
    btn_thr_dn = MiniButton(pygame, pygame.Rect(left_rect.x + PAD, left_rect.y + 414, (LEFT_W - 3 * PAD) // 2, 40), "Thr -")
    btn_thr_up = MiniButton(pygame, pygame.Rect(btn_thr_dn.rect.right + PAD, left_rect.y + 414, (LEFT_W - 3 * PAD) // 2, 40), "Thr +")
    btn_run_next = MiniButton(pygame, pygame.Rect(left_rect.x + PAD, left_rect.y + 472, LEFT_W - 2 * PAD, 40), "TAB  Next Run")
    btn_reload = MiniButton(pygame, pygame.Rect(left_rect.x + PAD, left_rect.y + 520, LEFT_W - 2 * PAD, 40), "R  Reload CSVs")

    # Menu selections
    menu_run_set_idx = 0
    menu_patient_idx = 6  # S07
    threshold = 0.60

    # Playback
    window = 100
    speed = 1.0
    fps_target = 60
    playing = True
    trial = 0
    last_advance = time.time()

    # Debounced gate config (stability controller)
    DEB_OFF_GAP = 0.05
    DEB_K = 3
    DEB_N = 5
    gate_cache_key: Optional[Tuple[str, int, float]] = None
    gate_cache: Optional[Dict[str, np.ndarray]] = None

    runs: List[RunData] = []
    run_idx = 0
    run: Optional[RunData] = None

    MODE_MENU = "menu"
    MODE_PLAY = "play"
    mode = MODE_MENU

    log_lines: List[str] = ["[boot] visualizer ready"]

    def set_run(new_idx: int) -> None:
        nonlocal run_idx, run, trial
        run_idx = int(new_idx) % len(runs)
        run = runs[run_idx]
        trial = 0
        log_lines.append(f"[run] {run.name}")

    def load_selected() -> None:
        nonlocal runs, run, run_idx, trial
        label_rs = run_set_labels[menu_run_set_idx]
        dirs = run_sets[label_rs]
        subj = patients[menu_patient_idx]
        runs = [load_predictions_csv(d, subject=subj) for d in dirs]
        run_idx = 0
        run = runs[0]
        trial = 0
        log_lines.append(f"[start] patient=S{subj:02d} runset={label_rs}")

    while True:
        # navbar layout each frame (prevents overlap)
        status = "● RUNNING" if playing else "■ PAUSED"
        status_color = GREEN if playing else RED
        st_img = font.render(status, True, status_color)
        status_w = st_img.get_width()
        status_rect = pygame.Rect(W - PAD - status_w - 18, 10, status_w + 18, NAV_H - 20)

        title_rect = pygame.Rect(PAD, 0, max(100, status_rect.x - PAD - 20), NAV_H)
        title_text = _fit_text(font, f"M.I.N.D / DECODER_UI  |  patient=S{patients[menu_patient_idx]:02d}", title_rect.width - 6)
        title_img = font.render(title_text, True, CYAN)

        # Tabs get the remaining width, and will auto-short labels as needed.
        tabs_start = title_rect.x + title_img.get_width() + 24
        tabs_end = status_rect.x - 12
        tabs.layout(font=font_s, start_x=tabs_start, end_x=tabs_end)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    pygame.quit()
                    return
                if mode == MODE_MENU:
                    if event.key == pygame.K_RETURN:
                        load_selected()
                        mode = MODE_PLAY
                    elif event.key == pygame.K_UP:
                        menu_patient_idx = (menu_patient_idx - 1) % len(patients)
                    elif event.key == pygame.K_DOWN:
                        menu_patient_idx = (menu_patient_idx + 1) % len(patients)
                    elif event.key == pygame.K_LEFT:
                        menu_run_set_idx = (menu_run_set_idx - 1) % len(run_set_labels)
                    elif event.key == pygame.K_RIGHT:
                        menu_run_set_idx = (menu_run_set_idx + 1) % len(run_set_labels)
                    elif event.key == pygame.K_LEFTBRACKET:
                        threshold = _clamp(threshold - 0.01, 0.0, 1.0)
                    elif event.key == pygame.K_RIGHTBRACKET:
                        threshold = _clamp(threshold + 0.01, 0.0, 1.0)
                else:
                    if event.key == pygame.K_m:
                        mode = MODE_MENU
                    elif event.key == pygame.K_SPACE:
                        playing = not playing
                    elif event.key == pygame.K_LEFT:
                        trial = max(0, trial - 1)
                    elif event.key == pygame.K_RIGHT and run is not None:
                        trial = min(len(run.df) - 1, trial + 1)
                    elif event.key == pygame.K_TAB and runs:
                        set_run(run_idx + 1)
                    elif event.key == pygame.K_r:
                        load_selected()
                    elif event.key == pygame.K_LEFTBRACKET:
                        step = 0.05 if (event.mod & pygame.KMOD_SHIFT) else 0.01
                        threshold = _clamp(threshold - step, 0.0, 1.0)
                    elif event.key == pygame.K_RIGHTBRACKET:
                        step = 0.05 if (event.mod & pygame.KMOD_SHIFT) else 0.01
                        threshold = _clamp(threshold + step, 0.0, 1.0)
                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        speed = max(0.1, speed / 1.25)
                    elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                        speed = min(10.0, speed * 1.25)
                    elif event.key == pygame.K_1:
                        tabs.active = "OVERVIEW"
                    elif event.key == pygame.K_2:
                        tabs.active = "DATA"
                    elif event.key == pygame.K_3:
                        tabs.active = "LOGS"

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and mode == MODE_PLAY:
                pos = event.pos
                if tabs.handle_click(pos):
                    log_lines.append(f"[tab] {tabs.active}")
                elif btn_play.hit(pos):
                    playing = not playing
                elif btn_prev.hit(pos):
                    trial = max(0, trial - 1)
                elif btn_next.hit(pos) and run is not None:
                    trial = min(len(run.df) - 1, trial + 1)
                elif btn_thr_dn.hit(pos):
                    threshold = _clamp(threshold - 0.01, 0.0, 1.0)
                elif btn_thr_up.hit(pos):
                    threshold = _clamp(threshold + 0.01, 0.0, 1.0)
                elif btn_run_next.hit(pos):
                    set_run(run_idx + 1)
                elif btn_reload.hit(pos):
                    load_selected()

        if mode == MODE_PLAY and playing and speed > 0 and run is not None:
            now = time.time()
            interval = 0.10 / speed
            if now - last_advance >= interval:
                trial = min(len(run.df) - 1, trial + 1)
                last_advance = now

        screen.fill(BG)
        screen.blit(grid, (0, 0))

        if mode == MODE_MENU:
            menu = pygame.Rect(220, 150, 840, 470)
            _rounded_panel(pygame, screen, menu, fill=PANEL, border=CYAN, border_alpha=140, radius=14)
            _draw_text(
                pygame,
                screen,
                font=font_b,
                text="M.I.N.D VISUALIZER",
                rect=pygame.Rect(menu.x + 24, menu.y + 16, menu.width - 48, 36),
                color=CYAN,
                pad=0,
            )
            _draw_text(
                pygame,
                screen,
                font=font,
                text="Choose patient + run-set (ENTER).",
                rect=pygame.Rect(menu.x + 24, menu.y + 58, menu.width - 48, 24),
                color=TEXT_DIM,
                pad=0,
            )

            row_y = menu.y + 118
            _draw_text(
                pygame,
                screen,
                font=font,
                text="Patient:",
                rect=pygame.Rect(menu.x + 24, row_y, 140, 26),
                color=MAGENTA,
            )
            _draw_text(
                pygame,
                screen,
                font=font,
                text=f"S{patients[menu_patient_idx]:02d}",
                rect=pygame.Rect(menu.x + 180, row_y, 90, 26),
                color=TEXT,
            )
            _draw_text(
                pygame,
                screen,
                font=font_s,
                text="UP/DOWN",
                rect=pygame.Rect(menu.x + 280, row_y + 2, menu.width - 320, 22),
                color=TEXT_DIM,
            )

            row_y = menu.y + 158
            _draw_text(pygame, screen, font=font, text="Run set:", rect=pygame.Rect(menu.x + 24, row_y, 140, 26), color=MAGENTA)
            label_rs = run_set_labels[menu_run_set_idx]
            _draw_text(
                pygame,
                screen,
                font=font,
                text=label_rs,
                rect=pygame.Rect(menu.x + 180, row_y, menu.width - 220, 26),
                color=TEXT,
            )
            _draw_text(
                pygame,
                screen,
                font=font_s,
                text="LEFT/RIGHT",
                rect=pygame.Rect(menu.x + 24, row_y + 32, menu.width - 48, 22),
                color=TEXT_DIM,
            )

            row_y = menu.y + 230
            _draw_text(pygame, screen, font=font, text="Threshold:", rect=pygame.Rect(menu.x + 24, row_y, 140, 26), color=MAGENTA)
            _draw_text(pygame, screen, font=font, text=f"{threshold:.2f}", rect=pygame.Rect(menu.x + 180, row_y, 90, 26), color=TEXT)
            _draw_text(pygame, screen, font=font_s, text="[ / ]", rect=pygame.Rect(menu.x + 280, row_y + 2, menu.width - 320, 22), color=TEXT_DIM)

            _draw_text(pygame, screen, font=font, text="Included runs:", rect=pygame.Rect(menu.x + 24, menu.y + 278, menu.width - 48, 26), color=MAGENTA)
            y = menu.y + 312
            for p in run_sets[label_rs]:
                _draw_text(
                    pygame,
                    screen,
                    font=font_s,
                    text=f"- {p.name}",
                    rect=pygame.Rect(menu.x + 24, y, menu.width - 48, 22),
                    color=TEXT_DIM,
                )
                y += 22

            _draw_text(
                pygame,
                screen,
                font=font_s,
                text="ENTER start  |  ESC quit",
                rect=pygame.Rect(menu.x + 24, menu.bottom - 40, menu.width - 48, 24),
                color=TEXT_DIM,
            )
            screen.blit(scan, (0, 0))
            pygame.display.flip()
            clock.tick(fps_target)
            continue

        assert run is not None
        df = run.df
        row = df.iloc[trial]
        y_true = _safe_int(row.get("y_true", np.nan))
        y_pred = _safe_int(row.get("ensemble_pred", np.nan))

        # Gate arrays (raw vs debounced) computed from full-sequence probabilities.
        gkey = (run.name, int(run.subject), float(threshold))
        if gate_cache_key != gkey or gate_cache is None:
            gate_cache_key = gkey
            gate_cache = _compute_gate_arrays(
                df=df,
                prob_cols=run.prob_cols,
                threshold=float(threshold),
                off_gap=float(DEB_OFF_GAP),
                k=int(DEB_K),
                n=int(DEB_N),
            )

        assert gate_cache is not None
        p_max_all = gate_cache["p_max"]
        fired_raw_all = gate_cache["fired_raw"]
        pred_raw_all = gate_cache["pred_raw"]
        fired_deb_all = gate_cache["fired_deb"]
        pred_deb_all = gate_cache["pred_deb"]

        max_prob = float(p_max_all[trial]) if trial < len(p_max_all) else float("nan")
        fired_raw = bool(fired_raw_all[trial])
        fired_deb = bool(fired_deb_all[trial])

        probs = row[run.prob_cols].to_numpy(dtype=float)
        pred_cls = int(np.nanargmax(probs)) if np.isfinite(probs).any() else (y_pred if y_pred is not None else 0)

        upto_idx = slice(0, trial + 1)
        y_true_full = _safe_int_series(df["y_true"]) if "y_true" in df.columns else np.full((len(df),), -1, dtype=int)
        known = y_true_full[upto_idx] != -1

        # All-trials argmax accuracy (no gate).
        y_hat_full = gate_cache["y_hat"]
        acc_all = float(np.mean((y_hat_full[upto_idx][known] == y_true_full[upto_idx][known]))) if known.any() else float("nan")

        def _conf_acc(pred: np.ndarray, fired: np.ndarray) -> float:
            fm = fired[upto_idx] & known
            if not np.any(fm):
                return float("nan")
            return float(np.mean(pred[upto_idx][fm] == y_true_full[upto_idx][fm]))

        cov_raw = float(np.mean(fired_raw_all[upto_idx])) if trial >= 0 else 0.0
        cov_deb = float(np.mean(fired_deb_all[upto_idx])) if trial >= 0 else 0.0
        acc_conf_raw = _conf_acc(pred_raw_all, fired_raw_all)
        acc_conf_deb = _conf_acc(pred_deb_all, fired_deb_all)

        togg_raw = toggle_rate(fired_raw_all[upto_idx])
        togg_deb = toggle_rate(fired_deb_all[upto_idx])
        # Wrong-fire safety proxy: wrong & fired as fraction of attempts (all trials with known y_true).
        if known.any():
            wrong_fire_raw = wrong_fire_rate_all(
                y_true=y_true_full[upto_idx][known],
                y_pred=pred_raw_all[upto_idx][known],
                fired=fired_raw_all[upto_idx][known],
            )
            wrong_fire_deb = wrong_fire_rate_all(
                y_true=y_true_full[upto_idx][known],
                y_pred=pred_deb_all[upto_idx][known],
                fired=fired_deb_all[upto_idx][known],
            )
        else:
            wrong_fire_raw = float("nan")
            wrong_fire_deb = float("nan")

        def label(idx: Optional[int]) -> str:
            if idx is None:
                return "NA"
            return class_names[idx] if 0 <= idx < len(class_names) else f"Class {idx}"

        # NAV
        _rounded_panel(pygame, screen, nav_rect, fill=BG, border=CYAN, border_alpha=110, radius=0)
        screen.blit(title_img, (PAD, (NAV_H - title_img.get_height()) // 2))
        tabs.draw(screen, font_s)
        # status badge rect (clip-protected)
        _rounded_panel(pygame, screen, status_rect, fill=PANEL, border=CYAN, border_alpha=120, radius=8)
        _draw_text(pygame, screen, font=font, text=status, rect=status_rect, color=status_color, align="center", valign="middle", pad=6)

        # LEFT
        _rounded_panel(pygame, screen, left_rect, fill=PANEL, border=CYAN, border_alpha=90, radius=0)
        _draw_text(pygame, screen, font=font, text="PARAMETERS", rect=pygame.Rect(left_rect.x + PAD, left_rect.y + PAD, left_rect.width - 2 * PAD, 24), color=MAGENTA)
        _draw_text(pygame, screen, font=font_s, text=f"Run: {run.name}", rect=pygame.Rect(left_rect.x + PAD, left_rect.y + 44, left_rect.width - 2 * PAD, 20), color=TEXT_DIM)
        _draw_text(pygame, screen, font=font_s, text=f"Patient: S{patients[menu_patient_idx]:02d}", rect=pygame.Rect(left_rect.x + PAD, left_rect.y + 64, left_rect.width - 2 * PAD, 20), color=TEXT_DIM)
        _draw_text(pygame, screen, font=font_s, text=f"Trial speed: {speed:.2f}x", rect=pygame.Rect(left_rect.x + PAD, left_rect.y + 84, left_rect.width - 2 * PAD, 20), color=TEXT_DIM)
        _draw_text(pygame, screen, font=font_s, text=f"Threshold: {threshold:.2f}", rect=pygame.Rect(left_rect.x + PAD, left_rect.y + 104, left_rect.width - 2 * PAD, 20), color=TEXT_DIM)

        btn_play.draw(screen, font_s, active=playing)
        btn_prev.draw(screen, font_s)
        btn_next.draw(screen, font_s)
        btn_thr_dn.draw(screen, font_s)
        btn_thr_up.draw(screen, font_s)
        btn_run_next.draw(screen, font_s)
        btn_reload.draw(screen, font_s)
        _draw_text(
            pygame,
            screen,
            font=font_s,
            text="Keys: [ ] thr | -/+ speed | TAB next run | M menu | ESC quit",
            rect=pygame.Rect(left_rect.x + PAD, left_rect.bottom - 34, left_rect.width - 2 * PAD, 22),
            color=TEXT_DIM,
        )

        # MAIN
        _rounded_panel(pygame, screen, main_rect, fill=PANEL, border=CYAN, border_alpha=90, radius=0)
        _rounded_panel(pygame, screen, info_rect, fill=PANEL_2, border=CYAN, border_alpha=120, radius=8)
        _rounded_panel(pygame, screen, chart_rect, fill=PANEL_2, border=CYAN, border_alpha=120, radius=8)
        _rounded_panel(pygame, screen, probs_rect, fill=PANEL_2, border=CYAN, border_alpha=120, radius=8)

        header_rect = pygame.Rect(info_rect.x + 14, info_rect.y + 12, info_rect.width - 28, 34)
        _draw_text(pygame, screen, font=font_b, text="[ DECODER SNAPSHOT ]", rect=header_rect, color=CYAN)
        # Show raw vs debounced status as two right-aligned badges.
        badge_w = 220
        badge_h = 30
        b2 = pygame.Rect(header_rect.right - badge_w, header_rect.y + 2, badge_w, badge_h)
        b1 = pygame.Rect(b2.x - 12 - badge_w, b2.y, badge_w, badge_h)
        _draw_text(
            pygame,
            screen,
            font=font,
            text=f"RAW: {'FIRE' if fired_raw else 'HOLD'}",
            rect=b1,
            color=(GREEN if fired_raw else YELLOW),
            align="right",
            valign="middle",
        )
        _draw_text(
            pygame,
            screen,
            font=font,
            text=f"STABLE: {'FIRE' if fired_deb else 'HOLD'}",
            rect=b2,
            color=(GREEN if fired_deb else YELLOW),
            align="right",
            valign="middle",
        )

        # Two-column stats that auto-fit within info_rect.
        left_col = pygame.Rect(info_rect.x + 14, info_rect.y + 56, (info_rect.width - 28) // 2, 90)
        right_col = pygame.Rect(left_col.right + 14, left_col.y, (info_rect.width - 28) - left_col.width - 14, 90)
        line_h = _text_height(font) + 4
        raw_pred_now = _safe_int(pred_raw_all[trial]) if fired_raw else None
        deb_pred_now = _safe_int(pred_deb_all[trial]) if fired_deb else None
        for i, txt in enumerate([f"y_true: {label(y_true)}", f"raw_out: {label(raw_pred_now)}", f"stable_out: {label(deb_pred_now)}"]):
            _draw_text(pygame, screen, font=font, text=txt, rect=pygame.Rect(left_col.x, left_col.y + i * line_h, left_col.width, line_h), color=TEXT)
        r_lines = [
            f"raw cov: {cov_raw:.3f} | stable cov: {cov_deb:.3f}",
            (f"raw conf: {acc_conf_raw:.3f} | stable conf: {acc_conf_deb:.3f}" if (np.isfinite(acc_conf_raw) or np.isfinite(acc_conf_deb)) else "conf_acc: NA"),
            (f"all_acc: {acc_all:.3f}" if np.isfinite(acc_all) else "all_acc: NA"),
        ]
        for i, txt in enumerate(r_lines):
            _draw_text(pygame, screen, font=font, text=txt, rect=pygame.Rect(right_col.x, right_col.y + i * line_h, right_col.width, line_h), color=TEXT_DIM)

        bar = pygame.Rect(info_rect.x + 14, info_rect.y + 160, info_rect.width - 28, 18)
        _draw_bar(pygame, screen, bar, 0.0 if not np.isfinite(max_prob) else float(max_prob), fg=CYAN)
        x_thr = bar.x + int(bar.width * _clamp(threshold, 0.0, 1.0))
        pygame.draw.line(screen, MAGENTA, (x_thr, bar.y - 4), (x_thr, bar.bottom + 4), 2)

        win = int(window)
        start = max(0, trial - win + 1)
        hist = df.iloc[start : trial + 1]
        ys = hist["ensemble_max_prob"].to_numpy(dtype=float)
        ys = [float(v) if np.isfinite(v) else 0.0 for v in ys.tolist()]
        _draw_text(pygame, screen, font=font, text="Max prob history", rect=pygame.Rect(chart_rect.x + 14, chart_rect.y + 12, chart_rect.width - 28, 24), color=TEXT_DIM)
        plot = pygame.Rect(chart_rect.x + 14, chart_rect.y + 44, chart_rect.width - 28, chart_rect.height - 60)
        _draw_line_chart(pygame, screen, plot, ys, threshold=threshold)

        _draw_text(
            pygame,
            screen,
            font=font,
            text="Ensemble class probabilities",
            rect=pygame.Rect(probs_rect.x + 14, probs_rect.y + 12, probs_rect.width - 28, 24),
            color=TEXT_DIM,
        )
        y0 = probs_rect.y + 48
        max_show = min(len(run.prob_cols), 6)
        for i in range(max_show):
            col = run.prob_cols[i]
            cls_i = _safe_int(col.replace("p_ens_c", "")) or i
            p = float(row[col])
            name = _fit_text(font_s, label(cls_i), 150)
            c = _color_for_class(cls_i)
            is_top = cls_i == pred_cls
            _draw_text(
                pygame,
                screen,
                font=font_s,
                text=name,
                rect=pygame.Rect(probs_rect.x + 14, y0 + i * 42, 160, 20),
                color=(c if is_top else TEXT_DIM),
            )
            r = pygame.Rect(probs_rect.x + 188, y0 + i * 42 + 6, probs_rect.width - 280, 16)
            _draw_bar(pygame, screen, r, p, fg=c)
            _draw_text(
                pygame,
                screen,
                font=font_s,
                text=f"{p:.3f}",
                rect=pygame.Rect(r.right + 8, r.y - 2, probs_rect.right - (r.right + 16), 20),
                color=TEXT,
                align="right",
            )

        _rounded_panel(pygame, screen, right_rect, fill=PANEL, border=CYAN, border_alpha=90, radius=0)
        # Split prosthetic mock into raw vs stable panels.
        hand_gap = 12
        hand_h = (hand_rect.height - hand_gap) // 2
        hand_raw_rect = pygame.Rect(hand_rect.x, hand_rect.y, hand_rect.width, hand_h)
        hand_deb_rect = pygame.Rect(hand_rect.x, hand_rect.y + hand_h + hand_gap, hand_rect.width, hand_h)

        raw_gesture = label(raw_pred_now).upper() if fired_raw else "NULL"
        deb_gesture = label(deb_pred_now).upper() if fired_deb else "NULL"
        _draw_hand(pygame, screen, hand_raw_rect, gesture=raw_gesture, fired=fired_raw, title="RAW GATE")
        _draw_hand(pygame, screen, hand_deb_rect, gesture=deb_gesture, fired=fired_deb, title=f"STABLE GATE  k={DEB_K}/{DEB_N}")
        _rounded_panel(pygame, screen, stats_rect, fill=PANEL_2, border=CYAN, border_alpha=120, radius=8)
        model_pred_cols = [c for c in df.columns if c.endswith("_pred") and c != "ensemble_pred"]

        if tabs.active == "LOGS":
            _draw_text(pygame, screen, font=font, text="[ LOGS ]", rect=pygame.Rect(stats_rect.x + 14, stats_rect.y + 12, stats_rect.width - 28, 24), color=CYAN)
            lines = log_lines[-18:]
            y = stats_rect.y + 48
            for ln in lines:
                _draw_text(
                    pygame,
                    screen,
                    font=font_s,
                    text=ln,
                    rect=pygame.Rect(stats_rect.x + 14, y, stats_rect.width - 28, 20),
                    color=TEXT_DIM,
                )
                y += 20
        else:
            title = "[ DATA ]" if tabs.active == "DATA" else "[ OVERVIEW ]"
            _draw_text(pygame, screen, font=font, text=title, rect=pygame.Rect(stats_rect.x + 14, stats_rect.y + 12, stats_rect.width - 28, 24), color=CYAN)
            cards_rect = pygame.Rect(stats_rect.x + 10, stats_rect.y + 44, stats_rect.width - 20, 220)
            _draw_model_cards(pygame, screen, cards_rect, df=df, trial=trial, class_names=class_names, model_pred_cols=model_pred_cols)
            y = cards_rect.bottom + 12
            if tabs.active == "DATA":
                _draw_text(pygame, screen, font=font, text=f"trial: {trial+1}/{len(df)}", rect=pygame.Rect(stats_rect.x + 14, y, stats_rect.width - 28, 22), color=TEXT_DIM)
                y += 24
                _draw_text(pygame, screen, font=font, text=f"threshold: {threshold:.2f}", rect=pygame.Rect(stats_rect.x + 14, y, stats_rect.width - 28, 22), color=TEXT_DIM)
                y += 24
                _draw_text(pygame, screen, font=font, text=f"raw coverage: {cov_raw:.3f}", rect=pygame.Rect(stats_rect.x + 14, y, stats_rect.width - 28, 22), color=TEXT_DIM)
                y += 22
                _draw_text(pygame, screen, font=font, text=f"stable coverage: {cov_deb:.3f}", rect=pygame.Rect(stats_rect.x + 14, y, stats_rect.width - 28, 22), color=TEXT_DIM)
                y += 22
                _draw_text(pygame, screen, font=font, text=f"raw toggle rate: {togg_raw:.3f}" if np.isfinite(togg_raw) else "raw toggle rate: NA", rect=pygame.Rect(stats_rect.x + 14, y, stats_rect.width - 28, 22), color=TEXT_DIM)
                y += 22
                _draw_text(pygame, screen, font=font, text=f"stable toggle rate: {togg_deb:.3f}" if np.isfinite(togg_deb) else "stable toggle rate: NA", rect=pygame.Rect(stats_rect.x + 14, y, stats_rect.width - 28, 22), color=TEXT_DIM)
            else:
                _draw_text(pygame, screen, font=font, text="RAW vs STABLE gate (hysteresis + k-of-n).", rect=pygame.Rect(stats_rect.x + 14, y, stats_rect.width - 28, 22), color=TEXT_DIM)
                y += 24
                _draw_text(pygame, screen, font=font, text="Adjust threshold live (accuracy vs coverage).", rect=pygame.Rect(stats_rect.x + 14, y, stats_rect.width - 28, 22), color=TEXT_DIM)
                y += 22
                _draw_text(
                    pygame,
                    screen,
                    font=font,
                    text=f"wrong-fire (raw/stable): {wrong_fire_raw:.3f}/{wrong_fire_deb:.3f}" if (np.isfinite(wrong_fire_raw) or np.isfinite(wrong_fire_deb)) else "wrong-fire: NA",
                    rect=pygame.Rect(stats_rect.x + 14, y, stats_rect.width - 28, 22),
                    color=TEXT_DIM,
                )

        screen.blit(scan, (0, 0))
        pygame.display.flip()
        clock.tick(fps_target)


if __name__ == "__main__":
    main()
