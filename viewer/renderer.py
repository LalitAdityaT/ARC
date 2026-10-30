"""
ARC Viewer — Pygame Rendering Functions
=========================================
All draw_* helpers and the main render loop components.
Fonts are module-level globals initialised by the viewer entry point
after pygame.init() has been called.
"""

import pygame

from viewer.constants import (
    C_BG, C_GRID, C_GREEN, C_GREEN_DIM, C_CYAN, C_ORANGE,
    C_WHITE, C_GRAY, C_DARKGRAY, C_PANEL,
)

# These globals are set by arc_skeleton_viewer.main() after pygame.init()
FONT_TINY  = None
FONT_SMALL = None
FONT_MED   = None
FONT_LARGE = None
FONT_TITLE = None


def init_fonts():
    """Initialise module-level font globals.  Call once after pygame.init()."""
    global FONT_TINY, FONT_SMALL, FONT_MED, FONT_LARGE, FONT_TITLE
    try:
        FONT_TINY  = pygame.font.SysFont("Consolas", 11)
        FONT_SMALL = pygame.font.SysFont("Consolas", 13)
        FONT_MED   = pygame.font.SysFont("Consolas", 16, bold=True)
        FONT_LARGE = pygame.font.SysFont("Consolas", 24, bold=True)
        FONT_TITLE = pygame.font.SysFont("Consolas", 28, bold=True)
    except Exception:
        FONT_TINY  = pygame.font.Font(None, 16)
        FONT_SMALL = pygame.font.Font(None, 18)
        FONT_MED   = pygame.font.Font(None, 22)
        FONT_LARGE = pygame.font.Font(None, 30)
        FONT_TITLE = pygame.font.Font(None, 36)


# ── Panel / UI helpers ────────────────────────────────────────────────────────

def draw_panel(surf, x, y, w, h, title=None) -> int:
    """Draw a bordered panel and return the y offset for content."""
    pygame.draw.rect(surf, C_PANEL,    (x, y, w, h))
    pygame.draw.rect(surf, C_GREEN_DIM,(x, y, w, h), 1)
    pygame.draw.rect(surf, C_GREEN,    (x, y, 3, h))   # left accent bar
    if title:
        tf = FONT_TINY.render(title, True, C_GREEN_DIM)
        surf.blit(tf, (x + 10, y + 6))
    return y + (22 if title else 8)


def draw_stat_row(surf, x, y, label, value, val_color=C_WHITE) -> int:
    """Draw a label + value row and return the next y offset."""
    lf = FONT_TINY.render(label, True, C_GRAY)
    vf = FONT_SMALL.render(str(value), True, val_color)
    surf.blit(lf, (x, y))
    surf.blit(vf, (x + 140, y))
    return y + 18


def draw_badge(surf, x, y, text, color) -> int:
    """Draw a filled badge label and return the next x offset."""
    tw, th = FONT_TINY.size(text)
    pygame.draw.rect(surf, color, (x, y, tw + 10, th + 6))
    tf = FONT_TINY.render(text, True, C_BG)
    surf.blit(tf, (x + 5, y + 3))
    return x + tw + 16


def draw_grid(surf, cx, cy, size=400, step=50):
    """Draw a perspective floor grid."""
    cols = size // step
    for i in range(-cols, cols + 1):
        ox = cx + i * step
        p1 = (ox, cy - size // 4)
        p2 = (cx + (ox - cx) * 3, cy + size // 2)
        pygame.draw.line(surf, C_GRID, p1, p2, 1)
    for j in range(0, cols + 1):
        fy = cy - size // 4 + j * (size * 3 // 4) // cols
        lx = cx - size // 2 - j * size // (cols * 2)
        rx = cx + size // 2 + j * size // (cols * 2)
        pygame.draw.line(surf, C_GRID, (lx, fy), (rx, fy), 1)
