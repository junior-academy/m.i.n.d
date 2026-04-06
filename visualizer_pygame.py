from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import pygame
import numpy as np
import pandas as pd


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
        self.active = self.labels[0] if self.labels else ""
        self.tabs: List[Tab] = []
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
                text_w = font.size(f"[{t.label}]")[0]
                tw = int(_clamp(text_w + pad_x * 2, 90, 180))
                t.rect = self.pygame.Rect(x, y, tw, h)
                x += tw + gap
            if x - gap > end_x:
                # fallback to equal widths if text-based layout overflows
                self._layout(font=None, start_x=start_x, end_x=end_x)

    def layout(self, *, font, start_x: int, end_x: int) -> None:
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
            text = font.render(f"[{t.label}]", True, MAGENTA if is_active else TEXT_DIM)
            surf.blit(text, (t.rect.x + 16, t.rect.y + 8))


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
def _draw_hand(pygame, surf, rect, *, gesture: str, fired: bool) -> None:
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
    label = f"[HAND OUTPUT]  {('FIRE' if fired else 'HOLD')}  {gesture}"
    surf.blit(font.render(label, True, CYAN if fired else TEXT_DIM), (rect.x + 14, rect.y + 12))

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
        title = _fit_text(font_s, title, card.width - 38)
        surf.blit(font_s.render(title, True, MAGENTA if is_best else TEXT_DIM), (card.x + 12, card.y + 10))

        cur_pred = _safe_int(df.iloc[trial].get(col, np.nan))
        cur_true = _safe_int(df.iloc[trial].get("y_true", np.nan))
        correct = (cur_pred is not None) and (cur_true is not None) and (cur_pred == cur_true)
        c = GREEN if correct else (RED if (cur_pred is not None and cur_true is not None) else TEXT_DIM)

        pred_txt = _fit_text(font, label(cur_pred), card.width - 24)
        surf.blit(font.render(pred_txt, True, c), (card.x + 12, card.y + 34))
        if cur_true is not None:
            true_txt = _fit_text(font_s, f"true: {label(cur_true)}", card.width - 24)
            surf.blit(font_s.render(true_txt, True, TEXT_DIM), (card.x + 12, card.y + 60))

        acc = acc_by_col.get(col, float("nan"))
        surf.blit(
            font_s.render(f"acc so far: {acc:.3f}" if np.isfinite(acc) else "acc so far: NA", True, TEXT_DIM),
            (card.x + 12, card.y + 86),
        )

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
        status_left = W - st_img.get_width() - PAD

        def ellipsize(text: str, max_w: int) -> str:
            return _fit_text(font, text, max_w)

        title_text = ellipsize(
            f"M.I.N.D / DECODER_UI  |  patient=S{patients[menu_patient_idx]:02d}",
            max(120, status_left - PAD - 24),
        )
        title_img = font.render(title_text, True, CYAN)
        title_right = PAD + title_img.get_width()
        tabs.layout(font=font_s, start_x=title_right + 32, end_x=status_left - 18)

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
            screen.blit(font_b.render("M.I.N.D VISUALIZER", True, CYAN), (menu.x + 24, menu.y + 22))
            screen.blit(font.render("Choose patient + run-set (ENTER).", True, TEXT_DIM), (menu.x + 24, menu.y + 60))

            screen.blit(font.render("Patient:", True, MAGENTA), (menu.x + 24, menu.y + 120))
            screen.blit(font.render(f"S{patients[menu_patient_idx]:02d}", True, TEXT), (menu.x + 180, menu.y + 120))
            screen.blit(font_s.render("UP/DOWN", True, TEXT_DIM), (menu.x + 260, menu.y + 124))

            screen.blit(font.render("Run set:", True, MAGENTA), (menu.x + 24, menu.y + 160))
            label_rs = run_set_labels[menu_run_set_idx]
            screen.blit(font.render(label_rs, True, TEXT), (menu.x + 180, menu.y + 160))
            screen.blit(font_s.render("LEFT/RIGHT", True, TEXT_DIM), (menu.x + 24, menu.y + 190))

            screen.blit(font.render("Threshold:", True, MAGENTA), (menu.x + 24, menu.y + 232))
            screen.blit(font.render(f"{threshold:.2f}", True, TEXT), (menu.x + 180, menu.y + 232))
            screen.blit(font_s.render("[ / ]", True, TEXT_DIM), (menu.x + 260, menu.y + 236))

            screen.blit(font.render("Included runs:", True, MAGENTA), (menu.x + 24, menu.y + 280))
            y = menu.y + 312
            for p in run_sets[label_rs]:
                screen.blit(font_s.render(f"- {p.name}", True, TEXT_DIM), (menu.x + 24, y))
                y += 22

            screen.blit(font_s.render("ENTER start  |  ESC quit", True, TEXT_DIM), (menu.x + 24, menu.bottom - 34))
            screen.blit(scan, (0, 0))
            pygame.display.flip()
            clock.tick(fps_target)
            continue

        assert run is not None
        df = run.df
        row = df.iloc[trial]
        y_true = _safe_int(row.get("y_true", np.nan))
        y_pred = _safe_int(row.get("ensemble_pred", np.nan))
        max_prob = _safe_float(row.get("ensemble_max_prob", np.nan))
        max_prob = float(max_prob) if max_prob is not None else float("nan")
        fired = bool(np.isfinite(max_prob) and max_prob >= threshold)
        probs = row[run.prob_cols].to_numpy(dtype=float)
        pred_cls = int(np.nanargmax(probs)) if np.isfinite(probs).any() else (y_pred if y_pred is not None else 0)

        upto = df.iloc[: trial + 1]
        maxp = upto["ensemble_max_prob"].to_numpy(dtype=float)
        fired_mask = np.isfinite(maxp) & (maxp >= threshold)
        coverage = float(fired_mask.mean()) if len(upto) else 0.0
        y_true_all = upto["y_true"].to_numpy(dtype=float)
        y_pred_all = upto["ensemble_pred"].to_numpy(dtype=float)
        acc_all = _metric_safe_mean((y_true_all == y_pred_all).astype(float)) if np.isfinite(y_true_all).any() else float("nan")
        acc_conf = (
            _metric_safe_mean((y_true_all[fired_mask] == y_pred_all[fired_mask]).astype(float))
            if fired_mask.any() and np.isfinite(y_true_all).any()
            else float("nan")
        )

        def label(idx: Optional[int]) -> str:
            if idx is None:
                return "NA"
            return class_names[idx] if 0 <= idx < len(class_names) else f"Class {idx}"

        # NAV
        _rounded_panel(pygame, screen, nav_rect, fill=BG, border=CYAN, border_alpha=110, radius=0)
        screen.blit(title_img, (PAD, 18))
        tabs.draw(screen, font_s)
        screen.blit(st_img, (W - st_img.get_width() - PAD, 18))

        # LEFT
        _rounded_panel(pygame, screen, left_rect, fill=PANEL, border=CYAN, border_alpha=90, radius=0)
        screen.blit(font.render("PARAMETERS", True, MAGENTA), (left_rect.x + PAD, left_rect.y + PAD))
        screen.blit(font_s.render(f"Run: {run.name}", True, TEXT_DIM), (left_rect.x + PAD, left_rect.y + 44))
        screen.blit(font_s.render(f"Patient: S{patients[menu_patient_idx]:02d}", True, TEXT_DIM), (left_rect.x + PAD, left_rect.y + 64))
        screen.blit(font_s.render(f"Trial speed: {speed:.2f}x", True, TEXT_DIM), (left_rect.x + PAD, left_rect.y + 84))
        screen.blit(font_s.render(f"Threshold: {threshold:.2f}", True, TEXT_DIM), (left_rect.x + PAD, left_rect.y + 104))

        btn_play.draw(screen, font_s, active=playing)
        btn_prev.draw(screen, font_s)
        btn_next.draw(screen, font_s)
        btn_thr_dn.draw(screen, font_s)
        btn_thr_up.draw(screen, font_s)
        btn_run_next.draw(screen, font_s)
        btn_reload.draw(screen, font_s)
        screen.blit(font_s.render("Keys: [ ] thr | -/+ speed | TAB next run | M menu | ESC quit", True, TEXT_DIM), (left_rect.x + PAD, left_rect.bottom - 30))

        # MAIN
        _rounded_panel(pygame, screen, main_rect, fill=PANEL, border=CYAN, border_alpha=90, radius=0)
        _rounded_panel(pygame, screen, info_rect, fill=PANEL_2, border=CYAN, border_alpha=120, radius=8)
        _rounded_panel(pygame, screen, chart_rect, fill=PANEL_2, border=CYAN, border_alpha=120, radius=8)
        _rounded_panel(pygame, screen, probs_rect, fill=PANEL_2, border=CYAN, border_alpha=120, radius=8)

        screen.blit(font_b.render("[ DECODER SNAPSHOT ]", True, CYAN), (info_rect.x + 14, info_rect.y + 14))
        status2 = "FIRE" if fired else "HOLD (NULL)"
        screen.blit(font_b.render(status2, True, GREEN if fired else YELLOW), (info_rect.right - 200, info_rect.y + 14))
        screen.blit(font.render(f"y_true: {label(y_true)}", True, TEXT), (info_rect.x + 14, info_rect.y + 62))
        screen.blit(font.render(f"ens_pred: {label(y_pred)}", True, TEXT), (info_rect.x + 14, info_rect.y + 90))
        screen.blit(font.render(f"max_prob: {max_prob:.3f}", True, TEXT), (info_rect.x + 14, info_rect.y + 118))
        screen.blit(font.render(f"coverage: {coverage:.3f}", True, TEXT_DIM), (info_rect.x + 320, info_rect.y + 62))
        screen.blit(font.render(f"conf_acc: {acc_conf:.3f}" if np.isfinite(acc_conf) else "conf_acc: NA", True, TEXT_DIM), (info_rect.x + 320, info_rect.y + 90))
        screen.blit(font.render(f"all_acc: {acc_all:.3f}" if np.isfinite(acc_all) else "all_acc: NA", True, TEXT_DIM), (info_rect.x + 320, info_rect.y + 118))

        bar = pygame.Rect(info_rect.x + 14, info_rect.y + 160, info_rect.width - 28, 18)
        _draw_bar(pygame, screen, bar, 0.0 if not np.isfinite(max_prob) else float(max_prob), fg=CYAN)
        x_thr = bar.x + int(bar.width * _clamp(threshold, 0.0, 1.0))
        pygame.draw.line(screen, MAGENTA, (x_thr, bar.y - 4), (x_thr, bar.bottom + 4), 2)

        win = int(window)
        start = max(0, trial - win + 1)
        hist = df.iloc[start : trial + 1]
        ys = hist["ensemble_max_prob"].to_numpy(dtype=float)
        ys = [float(v) if np.isfinite(v) else 0.0 for v in ys.tolist()]
        screen.blit(font.render("Max prob history", True, TEXT_DIM), (chart_rect.x + 14, chart_rect.y + 14))
        plot = pygame.Rect(chart_rect.x + 14, chart_rect.y + 44, chart_rect.width - 28, chart_rect.height - 60)
        _draw_line_chart(pygame, screen, plot, ys, threshold=threshold)

        screen.blit(font.render("Ensemble class probabilities", True, TEXT_DIM), (probs_rect.x + 14, probs_rect.y + 14))
        y0 = probs_rect.y + 48
        max_show = min(len(run.prob_cols), 6)
        for i in range(max_show):
            col = run.prob_cols[i]
            cls_i = _safe_int(col.replace("p_ens_c", "")) or i
            p = float(row[col])
            name = _fit_text(font_s, label(cls_i), 150)
            c = _color_for_class(cls_i)
            is_top = cls_i == pred_cls
            screen.blit(font_s.render(name, True, c if is_top else TEXT_DIM), (probs_rect.x + 14, y0 + i * 42))
            r = pygame.Rect(probs_rect.x + 180, y0 + i * 42 + 6, probs_rect.width - 240, 16)
            _draw_bar(pygame, screen, r, p, fg=c)
            screen.blit(font_s.render(f"{p:.3f}", True, TEXT), (r.right + 10, r.y - 2))

        _rounded_panel(pygame, screen, right_rect, fill=PANEL, border=CYAN, border_alpha=90, radius=0)
        gesture = label(pred_cls).upper()
        _draw_hand(pygame, screen, hand_rect, gesture=gesture, fired=fired)
        _rounded_panel(pygame, screen, stats_rect, fill=PANEL_2, border=CYAN, border_alpha=120, radius=8)
        model_pred_cols = [c for c in df.columns if c.endswith("_pred") and c != "ensemble_pred"]

        if tabs.active == "LOGS":
            screen.blit(font.render("[ LOGS ]", True, CYAN), (stats_rect.x + 14, stats_rect.y + 14))
            lines = log_lines[-18:]
            y = stats_rect.y + 48
            for ln in lines:
                screen.blit(font_s.render(_fit_text(font_s, ln, stats_rect.width - 28), True, TEXT_DIM), (stats_rect.x + 14, y))
                y += 20
        else:
            title = "[ DATA ]" if tabs.active == "DATA" else "[ OVERVIEW ]"
            screen.blit(font.render(title, True, CYAN), (stats_rect.x + 14, stats_rect.y + 14))
            cards_rect = pygame.Rect(stats_rect.x + 10, stats_rect.y + 44, stats_rect.width - 20, 220)
            _draw_model_cards(pygame, screen, cards_rect, df=df, trial=trial, class_names=class_names, model_pred_cols=model_pred_cols)
            y = cards_rect.bottom + 12
            if tabs.active == "DATA":
                screen.blit(font.render(f"trial: {trial+1}/{len(df)}", True, TEXT_DIM), (stats_rect.x + 14, y))
                y += 24
                screen.blit(font.render(f"threshold: {threshold:.2f}", True, TEXT_DIM), (stats_rect.x + 14, y))
                y += 24
                screen.blit(font.render(f"coverage (so far): {coverage:.3f}", True, TEXT_DIM), (stats_rect.x + 14, y))
            else:
                screen.blit(font.render("Ensemble gate: HOLD when unsure.", True, TEXT_DIM), (stats_rect.x + 14, y))
                y += 24
                screen.blit(font.render("Adjust threshold live (accuracy vs coverage).", True, TEXT_DIM), (stats_rect.x + 14, y))

        screen.blit(scan, (0, 0))
        pygame.display.flip()
        clock.tick(fps_target)


if __name__ == "__main__":
    main()
