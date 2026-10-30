"""
ARC — Occlusion Handler
========================
Temporal prediction with velocity + acceleration (Kalman-lite).
Falls back to spatial interpolation from visible neighbours when prediction
window is exhausted.
"""

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from config.settings import CFG


class OcclusionHandler:
    """
    Temporal prediction with velocity AND acceleration terms.
    Falls back to spatial interpolation from visible neighbours.
    """
    SKELETON_GRAPH = [
        (11,12),(11,13),(13,15),(12,14),(14,16),
        (11,23),(12,24),(23,24),(23,25),(24,26),
        (25,27),(26,28),(27,29),(28,30),(29,31),(30,32),
        (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8)
    ]

    def __init__(self):
        self.conf_thresh    = CFG['confidence_threshold']
        self.max_pred       = CFG['max_prediction_frames']
        self.buf_size       = CFG['temporal_buffer']
        self.history: Dict[int, deque]   = {}
        self.counter: Dict[int, int]     = {}

        # Stats for paper
        self.total_joints   = 0
        self.occluded       = 0
        self.recovered      = 0
        self.failed         = 0

    def _history(self, jid: int) -> deque:
        if jid not in self.history:
            self.history[jid] = deque(maxlen=self.buf_size)
        return self.history[jid]

    def _predict(self, jid: int) -> Optional[Dict]:
        h = list(self._history(jid))
        if len(h) < 3:
            if len(h) >= 2:
                v = {ax: h[-1][ax] - h[-2][ax] for ax in 'xyz'}
                return {ax: h[-1][ax] + v[ax] for ax in 'xyz'} | {'visibility': 0.4}
            return None
        p2 = np.array([h[-3][ax] for ax in 'xyz'])
        p1 = np.array([h[-2][ax] for ax in 'xyz'])
        p0 = np.array([h[-1][ax] for ax in 'xyz'])
        vel  = p0 - p1
        acc  = (p0 - p1) - (p1 - p2)
        pred = p0 + vel + 0.5 * acc
        return {'x': float(pred[0]), 'y': float(pred[1]), 'z': float(pred[2]), 'visibility': 0.4}

    def _spatial(self, jid: int, skeleton: List[Dict]) -> Optional[Dict]:
        lm_map = {lm['id']: lm for lm in skeleton}
        neighbours = []
        for a, b in self.SKELETON_GRAPH:
            if a == jid and b in lm_map and lm_map[b]['visibility'] >= self.conf_thresh:
                neighbours.append(lm_map[b])
            elif b == jid and a in lm_map and lm_map[a]['visibility'] >= self.conf_thresh:
                neighbours.append(lm_map[a])
        if not neighbours:
            return None
        arr = np.array([[n['x'], n['y'], n['z']] for n in neighbours])
        m = arr.mean(axis=0)
        return {'x': float(m[0]), 'y': float(m[1]), 'z': float(m[2]), 'visibility': 0.35}

    def process(self, skeleton: List[Dict]) -> Tuple[List[Dict], Dict]:
        out = []
        for lm in skeleton:
            jid = lm['id']
            self.total_joints += 1

            if lm['visibility'] >= self.conf_thresh:
                self._history(jid).append(lm)
                self.counter[jid] = 0
                out.append(lm)
            else:
                self.occluded += 1
                cnt = self.counter.get(jid, 0) + 1
                self.counter[jid] = cnt

                recovered = None
                if cnt <= self.max_pred:
                    recovered = self._predict(jid)
                if recovered is None:
                    recovered = self._spatial(jid, skeleton)

                if recovered:
                    lm_out = lm.copy()
                    lm_out.update(recovered)
                    lm_out['interpolated'] = True
                    out.append(lm_out)
                    self.recovered += 1
                else:
                    out.append(lm)
                    self.failed += 1

        info = {
            'occluded':       self.occluded,
            'recovered':      self.recovered,
            'occlusion_rate': round(self.occluded / max(self.total_joints, 1) * 100, 1),
            'recovery_rate':  round(self.recovered / max(self.occluded, 1) * 100, 1),
        }
        return out, info
