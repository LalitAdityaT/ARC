"""
ARC Viewer — Visual Constants
================================
Colors, skeleton connection definitions, and key joint labels
used by the pygame 3D skeleton renderer.
"""

# ── Colors ────────────────────────────────────────────────────────────────────
C_BG         = (8,   12,  10)
C_GRID       = (0,   40,  20)
C_GREEN      = (0,   255, 120)
C_GREEN_DIM  = (0,   100, 50)
C_CYAN       = (0,   210, 255)
C_CYAN_DIM   = (0,   80,  100)
C_ORANGE     = (255, 130, 30)
C_ORANGE_DIM = (100, 50,  10)
C_WHITE      = (220, 240, 220)
C_GRAY       = (100, 120, 100)
C_DARKGRAY   = (30,  45,  30)
C_PANEL      = (10,  18,  12)
C_ACCENT     = (0,   255, 120)

# ── Skeleton connections — (joint_a, joint_b, color, thickness) ───────────────
CONNECTIONS = [
    # Spine / torso
    (11, 12, C_GREEN,  3),
    (11, 23, C_GREEN,  2),
    (12, 24, C_GREEN,  2),
    (23, 24, C_GREEN,  3),
    # Left arm
    (11, 13, C_CYAN,   2),
    (13, 15, C_CYAN,   2),
    # Right arm
    (12, 14, C_CYAN,   2),
    (14, 16, C_CYAN,   2),
    # Left leg
    (23, 25, C_ORANGE, 2),
    (25, 27, C_ORANGE, 2),
    # Right leg
    (24, 26, C_ORANGE, 2),
    (26, 28, C_ORANGE, 2),
    # Head
    (0,  11, C_GREEN,  1),
    (0,  12, C_GREEN,  1),
]

# ── Key joints — {id: (label, color)} ────────────────────────────────────────
KEY_JOINTS = {
    0:  ("HEAD",     C_GREEN),
    11: ("L.SHLDR",  C_CYAN),
    12: ("R.SHLDR",  C_CYAN),
    13: ("L.ELBOW",  C_CYAN),
    14: ("R.ELBOW",  C_CYAN),
    15: ("L.WRIST",  C_CYAN),
    16: ("R.WRIST",  C_CYAN),
    23: ("L.HIP",    C_ORANGE),
    24: ("R.HIP",    C_ORANGE),
    25: ("L.KNEE",   C_ORANGE),
    26: ("R.KNEE",   C_ORANGE),
    27: ("L.ANKLE",  C_ORANGE),
    28: ("R.ANKLE",  C_ORANGE),
}
