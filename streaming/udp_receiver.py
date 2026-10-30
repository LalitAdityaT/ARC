"""
ARC — UDP Receiver (viewer side)
==================================
Listens for skeleton JSON packets on a UDP port and provides a thread-safe
get_state() interface for the pygame renderer.

Supports both packet formats:
  • OLD (week5/week6): key='joints', pos='position', vis='visibility'
  • NEW (mocap_ue5_final): key='bones',  pos='pos',      vis='vis'
"""

import json
import socket
import threading
import time
from collections import deque
from typing import Dict

from config.settings import CFG


class UDPReceiver:
    def __init__(self, port: int = None):
        self.port = port or CFG['stream_port']
        self.joints: Dict  = {}
        self.sequence: int = 0
        self.packet_times  = deque(maxlen=60)
        self.connected     = False
        self.last_packet_time = 0.0
        self._lock    = threading.Lock()
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(0.5)
        try:
            sock.bind(('0.0.0.0', self.port))
        except OSError as e:
            print(f"[UDP] Cannot bind port {self.port}: {e}")
            print(f"      Make sure mocap_ue5_final.py is running first")
            return

        print(f"[UDP] Listening on port {self.port}...")

        while self._running:
            try:
                data, _ = sock.recvfrom(65535)
                packet  = json.loads(data.decode('utf-8'))

                joints_dict: Dict = {}

                raw_list = packet.get('joints') or packet.get('bones') or []
                for j in raw_list:
                    jid = j['id']
                    # Normalise new-format keys to old names the renderer expects
                    if 'pos' in j and 'position' not in j:
                        ue5 = j['pos']
                        j['position']        = ue5
                        j['unreal_position'] = {'x': ue5[0], 'y': ue5[1], 'z': ue5[2]}
                    if 'vis' in j and 'visibility' not in j:
                        j['visibility'] = j['vis']
                    if 'interp' in j and 'interpolated' not in j:
                        j['interpolated'] = j['interp']
                    if 'ik_fixed' in j and 'ik_corrected' not in j:
                        j['ik_corrected'] = j['ik_fixed']
                    joints_dict[jid] = j

                seq_num = packet.get('sequence', packet.get('seq', 0))
                now = time.time()
                with self._lock:
                    self.joints           = joints_dict
                    self.sequence         = seq_num
                    self.connected        = True
                    self.last_packet_time = now
                    self.packet_times.append(now)

            except socket.timeout:
                if time.time() - self.last_packet_time > 2.0:
                    with self._lock:
                        self.connected = False
            except Exception:
                pass

        sock.close()

    def get_state(self) -> Dict:
        with self._lock:
            if len(self.packet_times) >= 2:
                span = self.packet_times[-1] - self.packet_times[0]
                pps  = (len(self.packet_times) - 1) / max(span, 0.001)
            else:
                pps = 0.0
            return {
                'joints':    dict(self.joints),
                'sequence':  self.sequence,
                'connected': self.connected,
                'pps':       pps,
            }

    def stop(self):
        self._running = False
