"""
ARC — UE5 UDP/TCP Streamer (sender side)
=========================================
Streams skeleton JSON to Unreal Engine 5 via UDP (default) or TCP.
Packet format is compatible with a simple UE5 Python/Blueprint receiver
or the MocapLiveLink C++ plugin.

Max UDP packet ~65 KB; a 33-joint JSON is ~3 KB — well within limits.
"""

import json
import socket
import struct
import time
from collections import deque
from typing import Dict, List

import numpy as np

from config.settings import CFG


# MediaPipe index → UE5 Mannequin bone name
MP_TO_UE5_BONE = {
    0:  'head',
    11: 'clavicle_l',   12: 'clavicle_r',
    13: 'upperarm_l',   14: 'upperarm_r',
    15: 'lowerarm_l',   16: 'lowerarm_r',
    17: 'hand_l',       18: 'hand_r',
    23: 'thigh_l',      24: 'thigh_r',
    25: 'calf_l',       26: 'calf_r',
    27: 'foot_l',       28: 'foot_r',
    29: 'ball_l',       30: 'ball_r',
    # Face
    1:  'eye_l',        4:  'eye_r',
}


class UE5Streamer:
    """Streams skeleton data to UE5 via UDP (default) or TCP."""

    def __init__(self, host: str = '127.0.0.1', port: int = 8888, use_tcp: bool = False):
        self.host    = host
        self.port    = port
        self.use_tcp = use_tcp
        self.seq     = 0
        self.connected = False

        if use_tcp:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                self.sock.connect((host, port))
                self.connected = True
                print(f"[STREAM] TCP connected → {host}:{port}")
            except Exception as e:
                print(f"[STREAM] TCP connection failed: {e}")
        else:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.connected = True
            print(f"[STREAM] UDP ready → {host}:{port}")

        self.latencies: deque = deque(maxlen=120)
        self.sent = 0
        self.failed_sends = 0

    def send(self, landmarks: List[Dict], ts: float, metadata: Dict = None) -> bool:
        if not self.connected:
            return False

        t0 = time.perf_counter()

        bones = []
        for lm in landmarks:
            bone_name = MP_TO_UE5_BONE.get(lm['id'], f"mp_{lm['id']}")
            ue5_pos   = lm.get('ue5', {'x': 0.0, 'y': 0.0, 'z': 0.0})
            bones.append({
                'id':       lm['id'],
                'bone':     bone_name,
                'pos':      [round(ue5_pos['x'], 4), round(ue5_pos['y'], 4), round(ue5_pos['z'], 4)],
                'vis':      round(lm['visibility'], 3),
                'interp':   bool(lm.get('interpolated', False)),
                'ik_fixed': bool(lm.get('ik_corrected', False)),
            })

        packet = {
            'seq':   self.seq,
            'ts':    round(ts, 5),
            'bones': bones,
            'meta':  metadata or {},
        }

        try:
            data = json.dumps(packet, separators=(',', ':')).encode('utf-8')
            if self.use_tcp:
                self.sock.sendall(struct.pack('!I', len(data)) + data)
            else:
                self.sock.sendto(data, (self.host, self.port))
            self.seq  += 1
            self.sent += 1
            self.latencies.append((time.perf_counter() - t0) * 1000)
            return True
        except Exception:
            self.failed_sends += 1
            return False

    def avg_latency_ms(self) -> float:
        return float(np.mean(self.latencies)) if self.latencies else 0.0

    def close(self):
        self.sock.close()
