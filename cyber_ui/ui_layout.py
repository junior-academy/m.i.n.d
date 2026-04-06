from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import math
import pygame
import pygame_gui
from pygame_gui.elements import UIButton, UIDropDownMenu, UIHorizontalSlider, UILabel, UIPanel, UITextEntryLine

from chart import TelemetryChart


CYBER_BG = (10, 10, 15)  # #0a0a0f
CYAN = (0, 245, 255)  # #00f5ff
MAGENTA = (255, 0, 170)  # #ff00aa
PANEL_BG = (13, 17, 23)  # #0d1117
TEXT = (232, 244, 248)  # #e8f4f8
TEXT_DIM = (122, 155, 181)  # #7a9bb5
GREEN = (80, 220, 140)
RED = (230, 85, 85)


def _glow_rect(surf: pygame.Surface, rect: pygame.Rect, color: Tuple[int, int, int], radius: int = 6) -> None:
    # Simple "glow" by drawing multiple outlines.
    for w, a in [(6, 30), (3, 60), (1, 140)]:
        glow = pygame.Surface((rect.width + 2 * w, rect.height + 2 * w), pygame.SRCALPHA)
        c = (*color, a)
        pygame.draw.rect(
            glow,
            c,
            pygame.Rect(0, 0, glow.get_width(), glow.get_height()),
            width=1,
            border_radius=radius + w,
        )
        surf.blit(glow, (rect.x - w, rect.y - w))


def _scanlines(size: Tuple[int, int]) -> pygame.Surface:
    w, h = size
    overlay = pygame.Surface((w, h), pygame.SRCALPHA)
    for y in range(0, h, 3):
        overlay.fill((0, 0, 0, 24), pygame.Rect(0, y, w, 1))
    return overlay


def _dot_grid(size: Tuple[int, int], spacing: int = 18) -> pygame.Surface:
    w, h = size
    grid = pygame.Surface((w, h), pygame.SRCALPHA)
    dot = pygame.Surface((2, 2), pygame.SRCALPHA)
    dot.fill((0, 245, 255, 14))
    for y in range(0, h, spacing):
        for x in range(0, w, spacing):
            grid.blit(dot, (x, y))
    return grid


@dataclass
class ToggleState:
    on: bool


class NavBar:
    def __init__(self, manager: pygame_gui.UIManager, rect: pygame.Rect):
        self.rect = rect
        self.manager = manager

        # pygame_gui 0.6.x does not support `starting_layer_height`; keep it version-agnostic.
        self.panel = UIPanel(relative_rect=rect, manager=manager, object_id="#nav_bar")
        self.title = UILabel(
            relative_rect=pygame.Rect(14, 10, 260, 28),
            text="SIM_CORE v1.0",
            manager=manager,
            container=self.panel,
            object_id="#nav_title",
        )

        x = 320
        self.tabs = []
        for txt in ["OVERVIEW", "CONTROLS", "DATA", "LOGS"]:
            btn = UIButton(
                relative_rect=pygame.Rect(x, 8, 120, 32),
                text=f"[{txt}]",
                manager=manager,
                container=self.panel,
                object_id="#nav_tab",
            )
            self.tabs.append(btn)
            x += 128

        self.running = False
        self._pulse_t = 0.0
        self.active_tab = "OVERVIEW"

    def set_running(self, running: bool) -> None:
        self.running = bool(running)

    def set_active_tab(self, tab: str) -> None:
        tab = tab.strip().upper()
        if tab:
            self.active_tab = tab

    def update(self, dt: float) -> None:
        self._pulse_t += dt

    def draw(self, screen: pygame.Surface) -> None:
        # Bottom glow line
        pygame.draw.line(screen, (*CYAN, 180), (0, self.rect.bottom - 1), (screen.get_width(), self.rect.bottom - 1), 1)
        _glow_rect(screen, pygame.Rect(0, self.rect.bottom - 2, screen.get_width(), 2), CYAN, radius=0)

        # Status badge (manual draw for pulse)
        badge_w = 180
        badge_h = 30
        x = self.rect.right - badge_w - 14
        y = self.rect.y + (self.rect.height - badge_h) // 2
        badge = pygame.Rect(x, y, badge_w, badge_h)

        alpha = int(120 + 80 * (0.5 + 0.5 * math.sin(self._pulse_t * 3.2)))
        if self.running:
            label = "● RUNNING"
            c = GREEN
        else:
            label = "■ IDLE"
            c = RED

        bg = pygame.Surface((badge.width, badge.height), pygame.SRCALPHA)
        pygame.draw.rect(bg, (20, 24, 30, 230), bg.get_rect(), border_radius=6)
        pygame.draw.rect(bg, (*c, alpha), bg.get_rect(), width=1, border_radius=6)
        screen.blit(bg, badge.topleft)

        font = pygame.font.SysFont("consolas", 18)
        text = font.render(label, True, c)
        screen.blit(text, (badge.x + 12, badge.y + 6))

        # Active tab hint (subtle underline glow)
        for btn in self.tabs:
            txt = btn.text.replace("[", "").replace("]", "").strip().upper()
            if txt == self.active_tab:
                r = btn.get_abs_rect()
                pygame.draw.line(screen, (*MAGENTA, 220), (r.x + 8, r.bottom - 4), (r.right - 8, r.bottom - 4), 2)


class SimulationUI:
    def __init__(self, *, screen: pygame.Surface, manager: pygame_gui.UIManager, window_size: Tuple[int, int]):
        self.screen = screen
        self.manager = manager
        self.window_size = window_size

        w, h = window_size
        nav_h = 48
        side_w = 260
        pad = 12

        self.nav = NavBar(manager, pygame.Rect(0, 0, w, nav_h))

        content_top = nav_h
        content_h = h - nav_h

        self.left_panel = UIPanel(
            relative_rect=pygame.Rect(0, content_top, side_w, content_h),
            manager=manager,
            object_id="#left_sidebar",
        )

        self.right_panel = UIPanel(
            relative_rect=pygame.Rect(w - side_w, content_top, side_w, content_h),
            manager=manager,
            object_id="#right_panel",
        )

        main_x = side_w
        main_w = w - (side_w * 2)
        self.main_rect = pygame.Rect(main_x + pad, content_top + pad, main_w - 2 * pad, content_h - 2 * pad)

        # Left sidebar controls
        UILabel(
            relative_rect=pygame.Rect(16, 14, side_w - 32, 24),
            text="PARAMETERS",
            manager=manager,
            container=self.left_panel,
            object_id="#section_header",
        )

        y = 56
        self.speed_label = UILabel(
            relative_rect=pygame.Rect(16, y, side_w - 32, 20),
            text="Simulation Speed",
            manager=manager,
            container=self.left_panel,
            object_id="#field_label",
        )
        y += 24
        self.speed_slider = UIHorizontalSlider(
            relative_rect=pygame.Rect(16, y, side_w - 32, 24),
            start_value=1.0,
            value_range=(0.1, 10.0),
            manager=manager,
            container=self.left_panel,
            object_id="#slider",
        )
        y += 46

        UILabel(
            relative_rect=pygame.Rect(16, y, side_w - 32, 20),
            text="Mode",
            manager=manager,
            container=self.left_panel,
            object_id="#field_label",
        )
        y += 24
        self.mode_dropdown = UIDropDownMenu(
            options_list=["Passive", "Active", "Aggressive"],
            starting_option="Passive",
            relative_rect=pygame.Rect(16, y, side_w - 32, 32),
            manager=manager,
            container=self.left_panel,
            object_id="#dropdown",
        )
        y += 52

        UILabel(
            relative_rect=pygame.Rect(16, y, side_w - 32, 20),
            text="Seed / ID",
            manager=manager,
            container=self.left_panel,
            object_id="#field_label",
        )
        y += 24
        self.seed_entry = UITextEntryLine(
            relative_rect=pygame.Rect(16, y, side_w - 32, 32),
            manager=manager,
            container=self.left_panel,
            object_id="#text_entry",
        )
        self.seed_entry.set_text("6857")
        y += 56

        UILabel(
            relative_rect=pygame.Rect(16, y, side_w - 32, 20),
            text="Toggles",
            manager=manager,
            container=self.left_panel,
            object_id="#field_label",
        )
        y += 28

        self.toggles: Dict[str, ToggleState] = {
            "GRAVITY": ToggleState(True),
            "COLLISIONS": ToggleState(True),
            "DECAY": ToggleState(False),
        }
        self.toggle_buttons: Dict[str, UIButton] = {}
        for label in ["GRAVITY", "COLLISIONS", "DECAY"]:
            btn = UIButton(
                relative_rect=pygame.Rect(16, y, side_w - 32, 34),
                text=f"{label} " + ("[ON]" if self.toggles[label].on else "[OFF]"),
                manager=manager,
                container=self.left_panel,
                object_id="#toggle_button",
            )
            self.toggle_buttons[label] = btn
            y += 42

        y += 8
        self.run_button = UIButton(
            relative_rect=pygame.Rect(16, y, side_w - 32, 52),
            text="▶ RUN SIMULATION",
            manager=manager,
            container=self.left_panel,
            object_id="#primary_button",
        )
        y += 64

        half = (side_w - 40) // 2
        self.reset_button = UIButton(
            relative_rect=pygame.Rect(16, y, half, 40),
            text="⟳ RESET",
            manager=manager,
            container=self.left_panel,
            object_id="#secondary_button",
        )
        self.pause_button = UIButton(
            relative_rect=pygame.Rect(24 + half, y, half, 40),
            text="⏸ PAUSE",
            manager=manager,
            container=self.left_panel,
            object_id="#secondary_button",
        )
        self.debug_label = UILabel(
            relative_rect=pygame.Rect(16, content_h - 34, side_w - 32, 20),
            text="UI: ready",
            manager=manager,
            container=self.left_panel,
            object_id="#field_label",
        )

        # Right telemetry
        UILabel(
            relative_rect=pygame.Rect(16, 14, side_w - 32, 24),
            text="TELEMETRY",
            manager=manager,
            container=self.right_panel,
            object_id="#section_header",
        )

        self.chart_rect = pygame.Rect(w - side_w + 16, content_top + 52, side_w - 32, 260)
        self.chart = TelemetryChart(rect=self.chart_rect, series_a_color=CYAN, series_b_color=MAGENTA)

        # Stat boxes (manual draw)
        self.stat_rects = []
        stat_top = self.chart_rect.bottom + 18
        bw = (side_w - 32 - 12) // 2
        bh = 72
        x0 = w - side_w + 16
        y0 = content_top + (stat_top - content_top)
        self.stat_rects.append(("TICK", pygame.Rect(x0, y0, bw, bh)))
        self.stat_rects.append(("ENTITIES", pygame.Rect(x0 + bw + 12, y0, bw, bh)))
        self.stat_rects.append(("AVG ENERGY", pygame.Rect(x0, y0 + bh + 12, bw, bh)))
        self.stat_rects.append(("EVENTS", pygame.Rect(x0 + bw + 12, y0 + bh + 12, bw, bh)))

        self.tick = 0
        self.entities = 120
        self.avg_energy = 0.66
        self.events = 0
        self.fps = 60.0

        self.sim_running = False
        self.sim_paused = False

        self.logs = ["[boot] SIM_CORE initialized", "[boot] UI online"]

        self.world = SimulationWorld(rect=self.main_rect.inflate(-24, -64))
        self.world.reset(seed_text=self.seed_entry.get_text())

        self._scanline = _scanlines(window_size)
        self._grid = _dot_grid(window_size, spacing=18)

    def _toggle(self, key: str) -> None:
        st = self.toggles[key]
        st.on = not st.on
        state_txt = "[ON]" if st.on else "[OFF]"
        self.toggle_buttons[key].set_text(f"{key} {state_txt}")
        self.logs.append(f"[toggle] {key} -> {'ON' if st.on else 'OFF'}")

    def process_event(self, event: pygame.event.Event) -> None:
        # pygame_gui event APIs vary by version:
        # - some versions: event.type == pygame_gui.UI_BUTTON_PRESSED
        # - others: event.type == pygame.USEREVENT and event.user_type == pygame_gui.UI_BUTTON_PRESSED
        is_button_pressed = False
        if getattr(event, "type", None) == getattr(pygame_gui, "UI_BUTTON_PRESSED", object()):
            is_button_pressed = True
        if getattr(event, "type", None) == pygame.USEREVENT and getattr(event, "user_type", None) == pygame_gui.UI_BUTTON_PRESSED:
            is_button_pressed = True

        if is_button_pressed:
            if event.ui_element == self.run_button:
                self.sim_running = True
                self.sim_paused = False
                self.nav.set_running(True)
                self.debug_label.set_text("UI: RUN pressed")
                self.logs.append("[sim] RUN")
            elif event.ui_element == self.pause_button:
                if self.sim_running:
                    self.sim_paused = not self.sim_paused
                    self.debug_label.set_text("UI: PAUSE toggled")
                    self.logs.append(f"[sim] {'PAUSE' if self.sim_paused else 'RESUME'}")
            elif event.ui_element == self.reset_button:
                self.sim_running = False
                self.sim_paused = False
                self.tick = 0
                self.events = 0
                self.entities = 120
                self.avg_energy = 0.66
                self.nav.set_running(False)
                self.debug_label.set_text("UI: RESET pressed")
                self.world.reset(seed_text=self.seed_entry.get_text())
                self.logs.append("[sim] RESET")
            else:
                # Nav tabs
                for btn in self.nav.tabs:
                    if event.ui_element == btn:
                        txt = btn.text
                        # "[OVERVIEW]" -> "OVERVIEW"
                        txt = txt.replace("[", "").replace("]", "").strip()
                        self.nav.set_active_tab(txt)
                        self.debug_label.set_text(f"UI: tab {txt}")
                        self.logs.append(f"[nav] {txt}")
                        break
                for k, btn in self.toggle_buttons.items():
                    if event.ui_element == btn:
                        self._toggle(k)
                        self.debug_label.set_text(f"UI: toggle {k}")
                        break

    def update(self, dt: float) -> None:
        self.nav.update(dt)

        speed = float(self.speed_slider.get_current_value())
        self.chart.update(dt * speed)
        self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(1e-6, dt))

        if self.sim_running and not self.sim_paused:
            self.tick += 1
            # world dynamics
            mode = self.mode_dropdown.selected_option if hasattr(self.mode_dropdown, "selected_option") else "Passive"
            self.world.step(
                dt=dt * speed,
                mode=str(mode),
                gravity_on=self.toggles["GRAVITY"].on,
                collisions_on=self.toggles["COLLISIONS"].on,
                decay_on=self.toggles["DECAY"].on,
            )
            self.entities = int(self.world.n_entities)
            self.avg_energy = float(self.world.avg_energy)
            self.events = int(self.world.events_last_tick)

    def draw(self) -> None:
        # Background
        self.screen.fill(CYBER_BG)
        self.screen.blit(self._grid, (0, 0))

        # Main canvas panel
        panel = pygame.Surface((self.main_rect.width, self.main_rect.height), pygame.SRCALPHA)
        panel.fill((*PANEL_BG, 210))
        pygame.draw.rect(panel, (*CYAN, 90), panel.get_rect(), width=1, border_radius=6)
        self.screen.blit(panel, self.main_rect.topleft)
        _glow_rect(self.screen, self.main_rect, CYAN, radius=6)

        # Viewport header text
        font = pygame.font.SysFont("consolas", 22)
        state = "RUNNING" if (self.sim_running and not self.sim_paused) else ("PAUSED" if self.sim_paused else "IDLE")
        txt = font.render(f"[ SIMULATION VIEWPORT | {self.nav.active_tab} | {state} ]", True, (0, 245, 255))
        self.screen.blit(
            txt,
            (self.main_rect.x + 18, self.main_rect.y + 16),
        )

        # Subtle hex-ish dots overlay inside viewport
        dots = pygame.Surface((self.main_rect.width, self.main_rect.height), pygame.SRCALPHA)
        for y in range(20, self.main_rect.height, 28):
            for x in range(20, self.main_rect.width, 28):
                pygame.draw.circle(dots, (0, 245, 255, 14), (x, y), 1)
        self.screen.blit(dots, self.main_rect.topleft)

        self._draw_viewport_contents()

        # Telemetry chart (manual draw)
        self.chart.draw(self.screen)

        # Stat boxes (manual draw)
        self._draw_stats()

    def _draw_stats(self) -> None:
        font_label = pygame.font.SysFont("consolas", 14)
        font_value = pygame.font.SysFont("consolas", 22)

        values = {
            "TICK": str(self.tick),
            "ENTITIES": str(self.entities),
            "AVG ENERGY": f"{self.avg_energy:.3f}",
            "EVENTS": str(self.events),
        }
        for label, rect in self.stat_rects:
            # Box bg
            box = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
            box.fill((*PANEL_BG, 220))
            pygame.draw.rect(box, (*CYAN, 90), box.get_rect(), width=1, border_radius=6)
            self.screen.blit(box, rect.topleft)

            # Label
            t1 = font_label.render(label, True, CYAN)
            self.screen.blit(t1, (rect.x + 10, rect.y + 10))

            # Value
            t2 = font_value.render(values[label], True, TEXT)
            self.screen.blit(t2, (rect.x + 10, rect.y + 34))

    def draw_overlays(self) -> None:
        self.screen.blit(self._scanline, (0, 0))

    def _draw_progress_bars(self, rect: pygame.Rect) -> None:
        # Bars: entities, energy, tick cycle
        bar_w = 320
        bar_h = 14
        x = rect.x + 18
        y = rect.y + 46
        font = pygame.font.SysFont("consolas", 14)

        def bar(label: str, value: float, color: Tuple[int, int, int]) -> None:
            nonlocal y
            self.screen.blit(font.render(label, True, TEXT_DIM), (x, y - 16))
            bg = pygame.Rect(x, y, bar_w, bar_h)
            pygame.draw.rect(self.screen, (30, 32, 40), bg, border_radius=6)
            fill = pygame.Rect(x, y, int(bar_w * max(0.0, min(1.0, value))), bar_h)
            pygame.draw.rect(self.screen, (*color, 255), fill, border_radius=6)
            pygame.draw.rect(self.screen, (*CYAN, 120), bg, width=1, border_radius=6)
            y += 34

        # Normalize entities to [0,1] around a target of 200
        bar("ENTITIES", min(1.0, self.entities / 200.0), CYAN)
        bar("AVG ENERGY", max(0.0, min(1.0, self.avg_energy)), MAGENTA)
        bar("TICK (cycle)", (self.tick % 1000) / 1000.0, (240, 210, 120))
        self.screen.blit(font.render(f"FPS: {self.fps:5.1f}", True, TEXT_DIM), (x + 350, rect.y + 46))

    def _draw_viewport_contents(self) -> None:
        inner = self.world.rect
        # Always show top-left bars
        self._draw_progress_bars(self.main_rect)

        tab = self.nav.active_tab.upper()
        if tab == "OVERVIEW":
            self.world.draw(self.screen)
            # Minimal legend
            font = pygame.font.SysFont("consolas", 14)
            self.screen.blit(font.render("cyan: entities", True, CYAN), (inner.x, inner.bottom + 10))
            self.screen.blit(font.render("magenta: higher energy", True, MAGENTA), (inner.x + 160, inner.bottom + 10))
        elif tab == "DATA":
            self._draw_data_view()
        elif tab == "CONTROLS":
            self._draw_controls_view()
        elif tab == "LOGS":
            self._draw_logs_view()
        else:
            self.world.draw(self.screen)

    def _draw_controls_view(self) -> None:
        inner = self.world.rect
        pygame.draw.rect(self.screen, (20, 22, 28), inner, border_radius=8)
        pygame.draw.rect(self.screen, (*CYAN, 120), inner, width=1, border_radius=8)
        font = pygame.font.SysFont("consolas", 16)
        lines = [
            "UI CONTROLS",
            "",
            "- ▶ RUN SIMULATION: start ticking + animation",
            "- ⏸ PAUSE: freeze world",
            "- ⟳ RESET: reset world to seed",
            "",
            "PARAMETERS",
            f"- Speed: {float(self.speed_slider.get_current_value()):.2f}x",
            f"- Mode: {self.mode_dropdown.selected_option if hasattr(self.mode_dropdown, 'selected_option') else 'Passive'}",
            f"- Seed/ID: {self.seed_entry.get_text()}",
            f"- GRAVITY: {'ON' if self.toggles['GRAVITY'].on else 'OFF'}",
            f"- COLLISIONS: {'ON' if self.toggles['COLLISIONS'].on else 'OFF'}",
            f"- DECAY: {'ON' if self.toggles['DECAY'].on else 'OFF'}",
        ]
        y = inner.y + 18
        for ln in lines:
            col = CYAN if ln and ln == ln.upper() else TEXT
            img = font.render(ln, True, col)
            self.screen.blit(img, (inner.x + 18, y))
            y += 24

    def _draw_logs_view(self) -> None:
        inner = self.world.rect
        pygame.draw.rect(self.screen, (20, 22, 28), inner, border_radius=8)
        pygame.draw.rect(self.screen, (*CYAN, 120), inner, width=1, border_radius=8)
        font = pygame.font.SysFont("consolas", 16)
        title = font.render("EVENT LOG", True, CYAN)
        self.screen.blit(title, (inner.x + 18, inner.y + 14))

        lines = self.logs[-18:]
        y = inner.y + 50
        for ln in lines:
            img = font.render(ln, True, TEXT)
            self.screen.blit(img, (inner.x + 18, y))
            y += 22

    def _draw_data_view(self) -> None:
        inner = self.world.rect
        pygame.draw.rect(self.screen, (20, 22, 28), inner, border_radius=8)
        pygame.draw.rect(self.screen, (*CYAN, 120), inner, width=1, border_radius=8)
        font = pygame.font.SysFont("consolas", 16)
        title = font.render("DATA SNAPSHOT", True, CYAN)
        self.screen.blit(title, (inner.x + 18, inner.y + 14))

        # Show rolling energy series (simple sparkline)
        series = self.world.energy_history[-min(160, len(self.world.energy_history)) :]
        if len(series) >= 2:
            plot = pygame.Rect(inner.x + 18, inner.y + 56, inner.width - 36, 180)
            pygame.draw.rect(self.screen, (15, 18, 24), plot, border_radius=8)
            pygame.draw.rect(self.screen, (*CYAN, 90), plot, width=1, border_radius=8)
            pts = []
            for i, v in enumerate(series):
                x = plot.x + int((i / (len(series) - 1)) * (plot.width - 1))
                y = plot.y + int((1.0 - max(0.0, min(1.0, float(v)))) * (plot.height - 1))
                pts.append((x, y))
            pygame.draw.lines(self.screen, MAGENTA, False, pts, 2)
            self.screen.blit(font.render("AVG ENERGY (rolling)", True, TEXT_DIM), (plot.x, plot.y - 22))

        # Readouts
        y0 = inner.y + 260
        self.screen.blit(font.render(f"TICK: {self.tick}", True, TEXT), (inner.x + 18, y0))
        self.screen.blit(font.render(f"ENTITIES: {self.entities}", True, TEXT), (inner.x + 18, y0 + 26))
        self.screen.blit(font.render(f"AVG ENERGY: {self.avg_energy:.3f}", True, TEXT), (inner.x + 18, y0 + 52))
        self.screen.blit(font.render(f"EVENTS (tick): {self.events}", True, TEXT), (inner.x + 18, y0 + 78))


class SimulationWorld:
    def __init__(self, rect: pygame.Rect):
        self.rect = rect
        self.rng = np.random.default_rng(6857)
        self.pos = np.zeros((0, 2), dtype=float)
        self.vel = np.zeros((0, 2), dtype=float)
        self.energy = np.zeros((0,), dtype=float)
        self.events_last_tick = 0
        self.energy_history: list[float] = []

    def reset(self, seed_text: str) -> None:
        try:
            seed = int(seed_text.strip()) if seed_text.strip() else 6857
        except Exception:
            seed = 6857
        self.rng = np.random.default_rng(seed)
        n = 140
        x = self.rng.uniform(self.rect.x + 10, self.rect.right - 10, size=n)
        y = self.rng.uniform(self.rect.y + 10, self.rect.bottom - 10, size=n)
        self.pos = np.stack([x, y], axis=1)
        self.vel = self.rng.normal(0, 60, size=(n, 2))
        self.energy = np.clip(self.rng.normal(0.65, 0.12, size=(n,)), 0.05, 1.0)
        self.events_last_tick = 0
        self.energy_history = [float(self.energy.mean())]

    @property
    def n_entities(self) -> int:
        return int(self.pos.shape[0])

    @property
    def avg_energy(self) -> float:
        return float(self.energy.mean()) if self.energy.size else 0.0

    def step(self, *, dt: float, mode: str, gravity_on: bool, collisions_on: bool, decay_on: bool) -> None:
        if self.pos.size == 0:
            return
        self.events_last_tick = 0
        mode = mode.strip().lower()

        # mode affects acceleration/noise
        if mode == "aggressive":
            noise = 40.0
            accel = 30.0
        elif mode == "active":
            noise = 20.0
            accel = 18.0
        else:
            noise = 10.0
            accel = 10.0

        # drift/noise
        self.vel += self.rng.normal(0.0, noise, size=self.vel.shape) * dt

        # gravity
        if gravity_on:
            self.vel[:, 1] += 180.0 * dt

        # mild centering force
        cx, cy = self.rect.center
        to_center = np.stack([cx - self.pos[:, 0], cy - self.pos[:, 1]], axis=1)
        self.vel += (to_center / (np.linalg.norm(to_center, axis=1, keepdims=True) + 1e-6)) * (accel * dt)

        # integrate
        self.pos += self.vel * dt

        # collisions with bounds
        if collisions_on:
            left = self.pos[:, 0] < self.rect.x + 6
            right = self.pos[:, 0] > self.rect.right - 6
            top = self.pos[:, 1] < self.rect.y + 6
            bottom = self.pos[:, 1] > self.rect.bottom - 6
            bounce = left | right | top | bottom
            if bounce.any():
                self.events_last_tick += int(bounce.sum())
            self.vel[left | right, 0] *= -0.9
            self.vel[top | bottom, 1] *= -0.9
            self.pos[:, 0] = np.clip(self.pos[:, 0], self.rect.x + 6, self.rect.right - 6)
            self.pos[:, 1] = np.clip(self.pos[:, 1], self.rect.y + 6, self.rect.bottom - 6)

        # decay reduces energy + damp velocity
        if decay_on:
            self.energy *= (1.0 - 0.18 * dt)
            self.vel *= (1.0 - 0.10 * dt)

            # occasionally drop very low-energy entities (creates visible "coverage" drop)
            drop = self.energy < 0.08
            if drop.any():
                self.events_last_tick += int(drop.sum())
                keep = ~drop
                self.pos = self.pos[keep]
                self.vel = self.vel[keep]
                self.energy = self.energy[keep]

        # keep energy in range
        self.energy = np.clip(self.energy, 0.02, 1.0)
        self.energy_history.append(float(self.avg_energy))
        if len(self.energy_history) > 600:
            self.energy_history = self.energy_history[-600:]

    def draw(self, screen: pygame.Surface) -> None:
        # background
        pygame.draw.rect(screen, (16, 18, 24), self.rect, border_radius=10)
        pygame.draw.rect(screen, (*CYAN, 90), self.rect, width=1, border_radius=10)

        # entities
        if self.pos.size == 0:
            return
        # color based on energy
        for (x, y), e in zip(self.pos, self.energy):
            c = (
                int(CYAN[0] * (1.0 - e) + MAGENTA[0] * e),
                int(CYAN[1] * (1.0 - e) + MAGENTA[1] * e),
                int(CYAN[2] * (1.0 - e) + MAGENTA[2] * e),
            )
            pygame.draw.circle(screen, c, (int(x), int(y)), 3)
