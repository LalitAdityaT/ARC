"""
ARC — IK Retargeter
====================
FABRIK inverse kinematics + anatomical joint angle constraints.
Calibrates bone lengths from the first N frames of visible, high-confidence
landmarks, then falls back to anthropometric defaults.
"""

import math
from typing import Dict, List

import numpy as np

from config.settings import CFG


# ── Joint angle constraints (min_deg, max_deg) ───────────────────────────────
JOINT_CONSTRAINTS = {
    13: (0.0, 145.0), 14: (0.0, 145.0),      # elbows
    25: (0.0, 160.0), 26: (0.0, 160.0),      # knees
    11: (-90.0, 180.0), 12: (-90.0, 180.0),  # shoulders
    23: (-30.0, 120.0), 24: (-30.0, 120.0),  # hips
}

KINEMATIC_CHAINS = [
    [11, 13, 15], [12, 14, 16],   # arms
    [23, 25, 27], [24, 26, 28],   # legs
]

FALLBACK_BONE_LENGTHS = {
    (11,13):0.13,(13,15):0.12,(12,14):0.13,(14,16):0.12,
    (23,25):0.20,(25,27):0.19,(24,26):0.20,(26,28):0.19,
}


# ── Math helpers ──────────────────────────────────────────────────────────────
def _vec(j):
    return np.array([j['x'], j['y'], j['z']], dtype=np.float64)


def _set_vec(j, v):
    r = j.copy()
    r['x'] = float(v[0])
    r['y'] = float(v[1])
    r['z'] = float(v[2])
    return r


def _angle_between(a, b, c):
    ba, bc = a - b, c - b
    n1, n2 = np.linalg.norm(ba), np.linalg.norm(bc)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    return math.degrees(math.acos(np.clip(np.dot(ba, bc) / (n1 * n2), -1.0, 1.0)))


def _fabrik(positions, bone_lengths, target, root, max_iter=10, tol=0.001):
    n = len(positions)
    p = [x.copy() for x in positions]
    if np.linalg.norm(target - p[0]) >= sum(bone_lengths):
        for i in range(n - 1):
            d = target - p[i]
            nm = np.linalg.norm(d)
            if nm > 1e-9:
                p[i+1] = p[i] + d / nm * bone_lengths[i]
        return p
    for _ in range(max_iter):
        p[-1] = target.copy()
        for i in range(n - 2, -1, -1):
            d = p[i] - p[i+1]
            nm = np.linalg.norm(d)
            if nm > 1e-9:
                p[i] = p[i+1] + d / nm * bone_lengths[i]
        p[0] = root.copy()
        for i in range(n - 1):
            d = p[i+1] - p[i]
            nm = np.linalg.norm(d)
            if nm > 1e-9:
                p[i+1] = p[i] + d / nm * bone_lengths[i]
        if np.linalg.norm(p[-1] - target) < tol:
            break
    return p


def _enforce_angle(root_pos, joint_pos, effector_pos, joint_id):
    if joint_id not in JOINT_CONSTRAINTS:
        return effector_pos
    mn, mx = JOINT_CONSTRAINTS[joint_id]
    angle = _angle_between(root_pos, joint_pos, effector_pos)
    clamped = max(mn, min(mx, angle))
    if abs(clamped - angle) < 0.5:
        return effector_pos
    v1 = root_pos - joint_pos
    v2 = effector_pos - joint_pos
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return effector_pos
    axis = np.cross(v1, v2)
    nm = np.linalg.norm(axis)
    if nm < 1e-9:
        return effector_pos
    axis /= nm
    delta = math.radians(clamped - angle)
    c, s = math.cos(delta), math.sin(delta)
    K = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    R = np.eye(3) + s * K + (1 - c) * (K @ K)
    return joint_pos + R @ v2


# ── IK Retargeter ─────────────────────────────────────────────────────────────
class IKRetargeter:
    def __init__(self, calibration_frames=45):
        self.cal_frames   = calibration_frames
        self.bone_lengths = dict(FALLBACK_BONE_LENGTHS)
        self._accum       = {k: [] for k in FALLBACK_BONE_LENGTHS}
        self._count       = 0
        self.calibrated   = False
        self.corrections  = 0
        self.frames       = 0

    def _calibrate(self, lm_map: Dict):
        for (a, b) in self._accum:
            if a in lm_map and b in lm_map:
                if lm_map[a].get('visibility', 0) > 0.7 and lm_map[b].get('visibility', 0) > 0.7:
                    self._accum[(a,b)].append(
                        float(np.linalg.norm(_vec(lm_map[b]) - _vec(lm_map[a])))
                    )
        self._count += 1
        if self._count >= self.cal_frames:
            for k, samples in self._accum.items():
                if len(samples) >= 5:
                    self.bone_lengths[k] = float(np.median(samples))
            self.calibrated = True
            print(f"[IK] Bone calibration locked ({self._count} frames).")

    def _bone_len(self, a, b):
        return self.bone_lengths.get((a, b), self.bone_lengths.get((b, a), 0.12))

    def apply(self, landmarks: List[Dict]) -> List[Dict]:
        if not landmarks:
            return landmarks
        self.frames += 1
        lm_map = {lm['id']: lm.copy() for lm in landmarks}

        if not self.calibrated:
            self._calibrate(lm_map)

        for chain in KINEMATIC_CHAINS:
            if not all(jid in lm_map for jid in chain):
                continue
            if lm_map[chain[0]].get('visibility', 0) < 0.4:
                continue
            if lm_map[chain[-1]].get('visibility', 0) < 0.4:
                continue

            positions    = [_vec(lm_map[jid]) for jid in chain]
            bone_lengths = [self._bone_len(chain[i], chain[i+1]) for i in range(len(chain)-1)]
            corrected    = _fabrik(positions, bone_lengths, positions[-1].copy(), positions[0].copy())

            for i in range(1, len(chain) - 1):
                new_eff = _enforce_angle(corrected[i-1], corrected[i], corrected[i+1], chain[i])
                if not np.allclose(new_eff, corrected[i+1], atol=1e-4):
                    corrected[i+1] = new_eff
                    self.corrections += 1
                    lm_map[chain[i+1]]['ik_corrected'] = True

            for k, jid in enumerate(chain):
                if not np.allclose(corrected[k], _vec(lm_map[jid]), atol=1e-5):
                    lm_map[jid] = _set_vec(lm_map[jid], corrected[k])
                    lm_map[jid].setdefault('ik_corrected', True)

        return list(lm_map.values())

    def stats(self) -> Dict:
        return {
            'frames':                self.frames,
            'calibrated':            self.calibrated,
            'corrections_total':     self.corrections,
            'corrections_per_frame': round(self.corrections / max(self.frames, 1), 2),
            'bone_lengths':          {f"{a}->{b}": round(l, 4) for (a,b), l in self.bone_lengths.items()},
        }
