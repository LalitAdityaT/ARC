"""
ARC — Adaptive Smoothing Filters
==================================
One-Euro Filter (Casiez et al. 2012) applied per-joint for 3D pose smoothing.
Outperforms basic EWMA — adaptive cutoff based on movement speed.
"""

import math
from typing import Dict, List

from config.settings import CFG


class _LowPassFilter:
    def __init__(self, alpha: float):
        self._alpha = alpha
        self._y = None

    def filter(self, x: float, alpha: float) -> float:
        if self._y is None:
            self._y = x
        else:
            self._y = alpha * x + (1.0 - alpha) * self._y
        return self._y

    @property
    def value(self):
        return self._y


class OneEuroFilter:
    """
    One-Euro Filter for scalar signals.
    Reference: Casiez, G., Roussel, N., Vogel, D. (2012). CHI.
    """
    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self._min_cutoff = min_cutoff
        self._beta       = beta
        self._d_cutoff   = d_cutoff
        self._x_filt     = _LowPassFilter(self._alpha(min_cutoff))
        self._dx_filt    = _LowPassFilter(self._alpha(d_cutoff))
        self._last_t     = None

    def _alpha(self, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau * 30.0)

    def _alpha_dt(self, cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def filter(self, x: float, dt: float) -> float:
        dx = (x - (self._x_filt.value if self._x_filt.value is not None else x)) / max(dt, 1e-6)
        edx = self._dx_filt.filter(dx, self._alpha_dt(self._d_cutoff, dt))
        cutoff = self._min_cutoff + self._beta * abs(edx)
        return self._x_filt.filter(x, self._alpha_dt(cutoff, dt))

    def reset(self):
        self._x_filt  = _LowPassFilter(self._alpha(self._min_cutoff))
        self._dx_filt = _LowPassFilter(self._alpha(self._d_cutoff))


class JointOneEuroFilter:
    """3-axis One-Euro filter for a single joint."""
    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.fx = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self.fy = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self.fz = OneEuroFilter(min_cutoff, beta, d_cutoff)

    def filter(self, pos: Dict, dt: float) -> Dict:
        return {
            'x': self.fx.filter(pos['x'], dt),
            'y': self.fy.filter(pos['y'], dt),
            'z': self.fz.filter(pos['z'], dt),
            'visibility': pos['visibility'],
        }

    def reset(self):
        self.fx.reset()
        self.fy.reset()
        self.fz.reset()


class AdaptiveFilterBank:
    """One-Euro filter bank for all 33 MediaPipe joints."""
    def __init__(self):
        self._filters: Dict[int, JointOneEuroFilter] = {}

    def _get(self, joint_id: int) -> JointOneEuroFilter:
        if joint_id not in self._filters:
            self._filters[joint_id] = JointOneEuroFilter(
                min_cutoff=CFG['one_euro_min_cutoff'],
                beta=CFG['one_euro_beta'],
                d_cutoff=CFG['one_euro_d_cutoff'],
            )
        return self._filters[joint_id]

    def filter_landmark(self, landmark: Dict, dt: float) -> Dict:
        filt = self._get(landmark['id'])
        smoothed = filt.filter(landmark, dt)
        smoothed['id']   = landmark['id']
        smoothed['name'] = landmark.get('name', f"joint_{landmark['id']}")
        for key in ('interpolated', 'predicted', 'source', 'ik_corrected', 'fused'):
            if key in landmark:
                smoothed[key] = landmark[key]
        return smoothed

    def filter_all(self, landmarks: List[Dict], dt: float) -> List[Dict]:
        return [self.filter_landmark(lm, dt) for lm in landmarks]

    def reset(self):
        for f in self._filters.values():
            f.reset()
