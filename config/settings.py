"""
ARC — Shared Configuration
===========================
Single source of truth for all constants used across the project.
"""

# ── MOCAP / PIPELINE CONFIG ───────────────────────────────────────────────────
CFG = {
    # Cameras
    'front_cam_id':      0,
    'back_cam_id':       2,
    'cam_width':         640,
    'cam_height':        480,
    'cam_fps':           60,

    # MediaPipe — complexity=2 for highest accuracy (needs GPU or fast CPU)
    # 0=Lite, 1=Full, 2=Heavy
    'model_complexity':  2,
    'min_detect_conf':   0.5,
    'min_track_conf':    0.5,
    'smooth_landmarks':  False,      # we do our own smoothing

    # One-Euro Filter parameters (tuned for game animation)
    'one_euro_min_cutoff': 1.0,      # Hz — lower = more smoothing at slow speed
    'one_euro_beta':       0.007,    # slope of cutoff vs speed — higher = less lag on fast moves
    'one_euro_d_cutoff':   1.0,      # derivative cutoff

    # Occlusion
    'confidence_threshold':   0.35,
    'max_prediction_frames':  12,
    'temporal_buffer':        10,

    # IK
    'ik_calibration_frames':  45,    # more frames = better bone length estimate

    # UDP streaming
    'stream_host':     '127.0.0.1',
    'stream_port':     8888,
    'use_tcp':         False,

    # Coordinate system
    'world_scale_cm':  200.0,        # 2m capture volume

    # Camera fusion
    'back_cam_trust':  0.85,
    'cam_front_angle': 0.0,
    'cam_back_angle':  180.0,
}

# ── VIEWER / PYGAME CONFIG ────────────────────────────────────────────────────
UDP_PORT   = CFG['stream_port']
WIDTH      = 800
HEIGHT     = 600
TARGET_FPS = 60
TITLE      = "ARC  //  Adaptive Realtime Capture  —  3D Skeleton Viewer"
