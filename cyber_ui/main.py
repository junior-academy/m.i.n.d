from __future__ import annotations

import sys
from pathlib import Path

import pygame
import pygame_gui

from ui_layout import SimulationUI


def main() -> None:
    pygame.init()

    window_size = (1280, 800)
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption("SIM_CORE v1.0")

    clock = pygame.time.Clock()

    base_dir = Path(__file__).resolve().parent
    theme_path = base_dir / "theme.json"

    # Create the manager first without a theme so we can register fonts before loading theme styles.
    manager = pygame_gui.UIManager(window_size)

    # Register a monospace font dynamically (keeps repo lightweight).
    mono_candidates = [
        "consolas",
        "couriernew",
        "menlo",
        "dejavusansmono",
        "liberationmono",
        "monospace",
    ]
    mono_path = None
    for name in mono_candidates:
        p = pygame.font.match_font(name)
        if p:
            mono_path = p
            break
    if mono_path is None:
        mono_path = pygame.font.match_font(pygame.font.get_default_font()) or pygame.font.get_default_font()

    manager.add_font_paths("mono", mono_path)
    manager.preload_fonts(
        [
            {"name": "mono", "point_size": 14, "style": "regular"},
            {"name": "mono", "point_size": 16, "style": "regular"},
            {"name": "mono", "point_size": 18, "style": "regular"},
            {"name": "mono", "point_size": 22, "style": "regular"},
        ]
    )

    # Load theme after fonts are available (pygame_gui versions differ slightly in API).
    try:
        manager.get_theme().load_theme(str(theme_path))
    except Exception:
        try:
            manager.ui_theme.load_theme(str(theme_path))  # type: ignore[attr-defined]
        except Exception as e:
            raise SystemExit(f"Failed to load pygame_gui theme: {theme_path}\n{e}") from e

    ui = SimulationUI(screen=screen, manager=manager, window_size=window_size)

    running = True
    while running:
        time_delta = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                continue

            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
                continue

            manager.process_events(event)
            ui.process_event(event)

        ui.update(time_delta)
        manager.update(time_delta)

        ui.draw()
        manager.draw_ui(screen)
        ui.draw_overlays()

        pygame.display.flip()

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
