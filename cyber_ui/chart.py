from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import pygame
import numpy as np


PANEL_BG = (13, 17, 23)  # #0d1117
TEXT_DIM = (122, 155, 181)
OUTLINE = (0, 245, 255)


@dataclass
class TelemetryChart:
    rect: pygame.Rect
    series_a_color: Tuple[int, int, int]
    series_b_color: Tuple[int, int, int]
    capacity: int = 120
    t: float = 0.0
    series_a: List[float] = field(default_factory=list)
    series_b: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.series_a = [0.5 for _ in range(self.capacity)]
        self.series_b = [0.5 for _ in range(self.capacity)]

    def update(self, dt: float) -> None:
        self.t += dt

        # Mock "entity count" and "energy level"
        a = 0.55 + 0.35 * (0.5 + 0.5 * np.sin(self.t * 1.3)) + 0.03 * np.sin(self.t * 5.7)
        b = 0.45 + 0.40 * (0.5 + 0.5 * np.sin(self.t * 0.9 + 1.2)) + 0.02 * np.sin(self.t * 7.3)
        a = float(np.clip(a, 0.0, 1.0))
        b = float(np.clip(b, 0.0, 1.0))

        self.series_a.append(a)
        self.series_b.append(b)
        if len(self.series_a) > self.capacity:
            self.series_a.pop(0)
        if len(self.series_b) > self.capacity:
            self.series_b.pop(0)

    def draw(self, screen: pygame.Surface) -> None:
        panel = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
        panel.fill((*PANEL_BG, 220))
        pygame.draw.rect(panel, (*OUTLINE, 90), panel.get_rect(), width=1, border_radius=6)

        # Axis lines (dim)
        axis_color = (80, 90, 100)
        pygame.draw.line(panel, axis_color, (36, self.rect.height - 26), (self.rect.width - 14, self.rect.height - 26), 1)
        pygame.draw.line(panel, axis_color, (36, 16), (36, self.rect.height - 26), 1)

        # Labels
        font = pygame.font.SysFont("consolas", 14)
        panel.blit(font.render("ENTITY COUNT", True, self.series_a_color), (14, 10))
        panel.blit(font.render("ENERGY LEVEL", True, self.series_b_color), (14, 28))

        # Plot area
        plot = pygame.Rect(40, 18, self.rect.width - 56, self.rect.height - 52)

        def to_xy(i: int, v: float) -> Tuple[int, int]:
            x = plot.x + int((i / max(1, self.capacity - 1)) * plot.width)
            y = plot.y + int((1.0 - v) * plot.height)
            return x, y

        pts_a = [to_xy(i, v) for i, v in enumerate(self.series_a)]
        pts_b = [to_xy(i, v) for i, v in enumerate(self.series_b)]

        if len(pts_a) >= 2:
            pygame.draw.lines(panel, self.series_a_color, False, pts_a, 2)
        if len(pts_b) >= 2:
            pygame.draw.lines(panel, self.series_b_color, False, pts_b, 2)

        # Minimal tick marks
        for frac, label in [(0.0, "0"), (0.5, "0.5"), (1.0, "1")]:
            y = plot.y + int((1.0 - frac) * plot.height)
            pygame.draw.line(panel, (65, 70, 78), (plot.x - 4, y), (plot.x, y), 1)
            panel.blit(font.render(label, True, TEXT_DIM), (6, y - 8))

        screen.blit(panel, self.rect.topleft)


