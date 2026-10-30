"""
ARC Viewer — 3D Projection Helpers
=====================================
Simple perspective projection from UE5 world-space coordinates to 2D screen.
"""

import math
from typing import Tuple


def project_3d(
    x: float, y: float, z: float,
    cx: int, cy: int,
    scale: float = 1.8,
    cam_angle_x: float = 0.3,
    cam_angle_y: float = 0.0,
) -> Tuple[int, int, float]:
    """
    Simple 3D → 2D projection with Y/X axis rotation.

    UE5 coords: x=depth, y=right, z=up.
    Screen display: y→screen_x, z→screen_y, x→depth_scale.

    Returns (screen_x, screen_y, perspective_scale).
    """
    # Rotate around Y axis (left/right pan)
    cos_y = math.cos(cam_angle_y)
    sin_y = math.sin(cam_angle_y)
    rx = x * cos_y - y * sin_y
    ry = x * sin_y + y * cos_y

    # Rotate around X axis (up/down tilt)
    cos_x = math.cos(cam_angle_x)
    sin_x = math.sin(cam_angle_x)
    rz    = z * cos_x - ry * sin_x
    depth = z * sin_x + ry * cos_x

    # Perspective divide
    fov   = 600
    persp = fov / (fov + depth * 0.3 + 200)

    sx = int(cx + rx * scale * persp)
    sy = int(cy - rz * scale * persp)
    return sx, sy, persp


def lerp_color(
    c1: Tuple[int, int, int],
    c2: Tuple[int, int, int],
    t: float,
) -> Tuple[int, int, int]:
    """Linear interpolate between two RGB tuples."""
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))
