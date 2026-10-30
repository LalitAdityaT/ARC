"""
ARC — Camera Workers & Dual-Camera Fusion
==========================================
CameraWorker  : runs one OpenCV + MediaPipe PoseLandmarker in a background thread.
DualCameraFusion: blends front + back camera landmarks weighted by confidence.
CoordinateConverter: converts MediaPipe (Y-down, normalised) → UE5 (Z-up, cm).
"""

import os
import threading
import time
import urllib.request
from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision as _mp_vision
from mediapipe.tasks.python.core import base_options as _mp_base_options

from config.settings import CFG

# ── MediaPipe Tasks API aliases ───────────────────────────────────────────────
_PoseLandmarker        = _mp_vision.PoseLandmarker
_PoseLandmarkerOptions = _mp_vision.PoseLandmarkerOptions
_RunningMode           = _mp_vision.RunningMode
_BaseOptions           = _mp_base_options.BaseOptions
_IMAGE                 = mp.Image
_IMAGE_FORMAT          = mp.ImageFormat

# ── Landmark name lookup (BlazePose 33) ──────────────────────────────────────
_POSE_LANDMARK_NAMES = [
    "nose","left_eye_inner","left_eye","left_eye_outer",
    "right_eye_inner","right_eye","right_eye_outer",
    "left_ear","right_ear","mouth_left","mouth_right",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_pinky","right_pinky",
    "left_index","right_index","left_thumb","right_thumb",
    "left_hip","right_hip","left_knee","right_knee",
    "left_ankle","right_ankle","left_heel","right_heel",
    "left_foot_index","right_foot_index",
]

_MODEL_PATH = "pose_landmarker_heavy.task"


def _ensure_model(path: str):
    """Download pose_landmarker_heavy.task if not present."""
    if os.path.exists(path):
        return
    url = (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_heavy/float16/latest/"
        "pose_landmarker_heavy.task"
    )
    print(f"[MoCap] Downloading model → {path}  (once only, ~29 MB)...")
    urllib.request.urlretrieve(url, path)
    print(f"[MoCap] Model saved to {path}")


# ── Camera Worker ─────────────────────────────────────────────────────────────
class CameraWorker:
    """
    Runs one camera + MediaPipe PoseLandmarker (Tasks API) in a background thread.
    Works with MediaPipe 0.10.x which removed the legacy solutions API.
    """
    def __init__(self, cam_id: int, name: str = 'CAM'):
        self.cam_id  = cam_id
        self.name    = name
        self._lock   = threading.Lock()
        self._lm     = None
        self._frame  = None
        self._run    = False
        self._thread = None
        self._fps    = 0.0
        self._times: deque = deque(maxlen=30)

    def start(self) -> bool:
        cap = cv2.VideoCapture(self.cam_id)
        if not cap.isOpened():
            print(f"[{self.name}] ✗ Could not open camera {self.cam_id}")
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CFG['cam_width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG['cam_height'])
        cap.set(cv2.CAP_PROP_FPS,          CFG['cam_fps'])
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[{self.name}] ✓ {w}x{h}")
        self._cap    = cap
        self._run    = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def _loop(self):
        _ensure_model(_MODEL_PATH)

        options = _PoseLandmarkerOptions(
            base_options=_BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=_RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=CFG['min_detect_conf'],
            min_pose_presence_confidence=CFG['min_detect_conf'],
            min_tracking_confidence=CFG['min_track_conf'],
            output_segmentation_masks=False,
        )
        landmarker = _PoseLandmarker.create_from_options(options)
        frame_ms = 0

        while self._run:
            t0 = time.perf_counter()
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.005)
                continue

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = _IMAGE(image_format=_IMAGE_FORMAT.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_img, frame_ms)
            frame_ms += int((time.perf_counter() - t0) * 1000) + 1

            lm = None
            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                lm = [
                    {
                        'id':         idx,
                        'name':       _POSE_LANDMARK_NAMES[idx] if idx < len(_POSE_LANDMARK_NAMES) else f"joint_{idx}",
                        'x':          l.x,
                        'y':          l.y,
                        'z':          l.z,
                        'visibility': l.visibility if l.visibility is not None else 1.0,
                    }
                    for idx, l in enumerate(result.pose_landmarks[0])
                ]

            with self._lock:
                self._lm    = lm
                self._frame = frame

            dt = time.perf_counter() - t0
            self._times.append(dt)
            self._fps = 1.0 / np.mean(self._times) if self._times else 0.0

        landmarker.close()

    def get_latest(self) -> Tuple[Optional[List], Optional[np.ndarray]]:
        with self._lock:
            return self._lm, self._frame

    @property
    def fps(self) -> float:
        return self._fps

    def stop(self):
        self._run = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if hasattr(self, '_cap'):
            self._cap.release()


# ── Dual Camera Fusion ────────────────────────────────────────────────────────
class DualCameraFusion:
    def __init__(self):
        self.conf       = CFG['confidence_threshold']
        self.back_trust = CFG['back_cam_trust']
        self.frames_fused = 0
        self.both = 0
        self.front_only = 0
        self.back_only = 0

    def _mirror(self, j: Dict) -> Dict:
        m = j.copy()
        m['x'] = 1.0 - j['x']
        m['z'] = -j['z']
        return m

    def _blend(self, fj: Dict, bj: Dict) -> Dict:
        fw = fj['visibility']
        bw = bj['visibility'] * self.back_trust
        t  = fw + bw
        if t == 0:
            return fj
        wf, wb = fw / t, bw / t
        b = fj.copy()
        for ax in 'xyz':
            b[ax] = wf * fj[ax] + wb * bj[ax]
        b['visibility'] = max(fj['visibility'], bj['visibility'])
        b['fused'] = True
        return b

    def fuse(self, front: Optional[List], back: Optional[List]) -> Optional[List]:
        if front is None and back is None:
            return None
        if front is None:
            return [self._mirror(j) for j in back]
        if back is None:
            return front

        fm = {j['id']: j for j in front}
        bm = {j['id']: j for j in back}
        self.frames_fused += 1
        out = []
        for jid in sorted(set(list(fm) + list(bm))):
            fj = fm.get(jid)
            bj = bm.get(jid)
            fv = fj is not None and fj['visibility'] >= self.conf
            bv = bj is not None and bj['visibility'] >= self.conf
            if fv and bv:
                out.append(self._blend(fj, self._mirror(bj)))
                self.both += 1
            elif fv:
                r = fj.copy(); r['source'] = 'front'; out.append(r)
                self.front_only += 1
            elif bv:
                r = self._mirror(bj); r['source'] = 'back'; r['interpolated'] = True
                out.append(r)
                self.back_only += 1
            elif fj is not None:
                out.append(fj)
            elif bj is not None:
                out.append(self._mirror(bj))
        return out

    def stats(self) -> Dict:
        t = max(self.frames_fused * 33, 1)
        return {
            'both_pct':  round(self.both / t * 100, 1),
            'front_pct': round(self.front_only / t * 100, 1),
            'back_pct':  round(self.back_only / t * 100, 1),
        }


# ── Coordinate Converter ──────────────────────────────────────────────────────
class CoordinateConverter:
    """
    MediaPipe (right-hand, Y-down, normalised) → UE5 (left-hand, Z-up, cm).
    UE5 convention: X=Forward, Y=Right, Z=Up.
    """
    def __init__(self, world_scale: float = 200.0):
        self.scale = world_scale

    def to_ue5(self, mp_pos: Dict) -> Dict:
        x_n = mp_pos['x'] - 0.5
        y_n = mp_pos['y'] - 0.5
        z_n = mp_pos['z']
        return {
            'x': float( z_n * self.scale),   # depth   → X (forward)
            'y': float( x_n * self.scale),   # lateral → Y (right)
            'z': float(-y_n * self.scale),   # vert    → Z (up, flipped)
        }

    def skeleton_to_ue5(self, landmarks: List[Dict]) -> List[Dict]:
        out = []
        for lm in landmarks:
            l = lm.copy()
            l['ue5'] = self.to_ue5(lm)
            out.append(l)
        return out
