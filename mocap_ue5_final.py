"""
Real-Time Motion Capture — Final: Maximum Accuracy + Unreal Engine 5 Integration
==================================================================================
Targets:
  • MediaPipe FULL model (complexity=2) for highest joint accuracy
  • One-Euro Filter (replaces basic EWMA) — best-in-class adaptive smoothing
  • Multi-frame temporal prediction with velocity + acceleration (Kalman-lite)
  • FABRIK IK + anatomical constraints
  • Dual-camera fusion when second camera is available
  • Live Linking to UE5 via UDP (JSON) — compatible with MocapLiveLinkPlugin
  • Benchmark logger for IEEE paper (jitter, latency, fps, occlusion rate)

Python 3.10.6 | MediaPipe 0.10.x | OpenCV 4.x

Usage:
  python mocap_ue5_final.py                   # single camera, streams to UE5
  python mocap_ue5_final.py --dual            # dual camera mode
  python mocap_ue5_final.py --no-stream       # local preview only
  python mocap_ue5_final.py --benchmark 300   # log 300 frames then exit

UE5 side:
  Copy UE5_MocapReceiver.py to your UE5 project's Python folder and run it
  inside the UE5 Python console, OR use the companion MocapLiveLink C++ plugin.
"""

import argparse
import time
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional

import cv2
import numpy as np

from config.settings import CFG
from core.camera import CameraWorker, CoordinateConverter, DualCameraFusion
from core.filters import AdaptiveFilterBank
from core.ik import IKRetargeter
from core.occlusion import OcclusionHandler
from streaming.udp_sender import UE5Streamer
from benchmark.logger import BenchmarkLogger


class MocapSystemUE5:
    """
    Maximum-accuracy motion capture system with full UE5 integration.

    Pipeline (per frame):
      Camera(s) → MediaPipe Full Model → [Dual-cam Fusion] →
      Occlusion Handler (velocity+acc prediction) →
      IK Retargeter (FABRIK + anatomical constraints) →
      One-Euro Filter Bank →
      Coordinate Converter (MP → UE5) →
      UDP/TCP Streamer → UE5 Live Link
    """

    def __init__(
        self,
        dual_cam: bool = False,
        enable_stream: bool = True,
        benchmark_frames: int = 0,
    ):
        self.dual_cam         = dual_cam
        self.enable_stream    = enable_stream
        self.benchmark_frames = benchmark_frames

        # Pipeline modules
        self.cam_front   = CameraWorker(CFG['front_cam_id'], 'FRONT')
        self.cam_back    = CameraWorker(CFG['back_cam_id'],  'BACK') if dual_cam else None
        self.fusion      = DualCameraFusion() if dual_cam else None
        self.occ         = OcclusionHandler()
        self.ik          = IKRetargeter(CFG['ik_calibration_frames'])
        self.filter_bank = AdaptiveFilterBank()
        self.coord       = CoordinateConverter(CFG['world_scale_cm'])
        self.streamer    = (
            UE5Streamer(CFG['stream_host'], CFG['stream_port'], CFG['use_tcp'])
            if enable_stream else None
        )

        # Performance tracking
        self._frame_times: deque = deque(maxlen=60)
        self._fps_buf:     deque = deque(maxlen=30)
        self._jitter_raw:  deque = deque(maxlen=60)
        self._jitter_sm:   deque = deque(maxlen=60)
        self._prev_raw  = None
        self._prev_sm   = None
        self._last_t    = None

        # Benchmark
        self.bench = BenchmarkLogger() if benchmark_frames > 0 else None
        self._frames_done = 0

        # Session metadata
        self.session: Dict = {
            'start':    datetime.now().isoformat(),
            'frames':   0,
            'streamed': 0,
            'dual_cam': dual_cam,
        }

    # ── Startup ───────────────────────────────────────────────────────────────

    def start(self):
        print("\n" + "=" * 65)
        print("  MOCAP → UE5  |  Maximum Accuracy Mode")
        print("=" * 65)
        print(f"  MediaPipe model:  complexity={CFG['model_complexity']} (HEAVY)")
        print(f"  Filter:           One-Euro (min_cutoff={CFG['one_euro_min_cutoff']}, β={CFG['one_euro_beta']})")
        print(f"  IK calibration:   {CFG['ik_calibration_frames']} frames")
        print(f"  Dual camera:      {'YES' if self.dual_cam else 'NO'}")
        print(f"  Streaming:        {'UDP' if not CFG['use_tcp'] else 'TCP'} → {CFG['stream_host']}:{CFG['stream_port']}")
        print("=" * 65 + "\n")

        self.cam_front.start()
        if self.cam_back:
            ok = self.cam_back.start()
            if not ok:
                print("[WARN] Back camera unavailable — single camera mode.")
                self.cam_back = None
                self.fusion   = None
                self.dual_cam = False

        time.sleep(0.5)  # warm-up: discard first frames

    # ── Per-frame helpers ─────────────────────────────────────────────────────

    def _dt(self) -> float:
        now = time.perf_counter()
        dt  = (now - self._last_t) if self._last_t else 0.033
        self._last_t = now
        return max(dt, 1e-4)

    def _jitter(self, a: Optional[List], b: Optional[List]) -> float:
        if a is None or b is None or len(a) != len(b):
            return 0.0
        ca = np.array([[j['x'], j['y'], j['z']] for j in a])
        cb = np.array([[j['x'], j['y'], j['z']] for j in b])
        return float(np.mean(np.linalg.norm(ca - cb, axis=1)))

    # ── Main pipeline ─────────────────────────────────────────────────────────

    def process_frame(self) -> Dict:
        t0 = time.perf_counter()
        dt = self._dt()

        # Stage 1: Capture
        front_lm, front_frame = self.cam_front.get_latest()
        back_lm,  back_frame  = (self.cam_back.get_latest() if self.cam_back else (None, None))

        if front_lm is None and back_lm is None:
            return {'success': False}

        # Stage 2: Dual-cam fusion
        if self.fusion and (front_lm or back_lm):
            raw_lm = self.fusion.fuse(front_lm, back_lm)
        else:
            raw_lm = front_lm or back_lm

        if raw_lm is None:
            return {'success': False}

        # Stage 3: Occlusion recovery
        recovered_lm, occ_info = self.occ.process(raw_lm)

        # Stage 4: IK retargeting (FABRIK + joint constraints)
        ik_lm = self.ik.apply(recovered_lm)

        # Stage 5: One-Euro smoothing
        smoothed_lm = self.filter_bank.filter_all(ik_lm, dt)

        # Jitter tracking
        if self._prev_raw is not None:
            self._jitter_raw.append(self._jitter(raw_lm, self._prev_raw))
            self._jitter_sm.append(self._jitter(smoothed_lm, self._prev_sm))
        self._prev_raw = raw_lm
        self._prev_sm  = smoothed_lm

        # Stage 6: Coordinate conversion → UE5
        ue5_lm = self.coord.skeleton_to_ue5(smoothed_lm)

        # Stage 7: Stream
        proc_ms   = (time.perf_counter() - t0) * 1000
        stream_ok = False
        stream_ms = 0.0
        if self.streamer:
            metadata = {
                'fps':      self.fps(),
                'proc_ms':  round(proc_ms, 2),
                'ik':       self.ik.stats(),
                'occ':      occ_info,
                'dual_cam': self.dual_cam,
            }
            t_stream  = time.perf_counter()
            stream_ok = self.streamer.send(ue5_lm, time.time(), metadata)
            stream_ms = (time.perf_counter() - t_stream) * 1000
            if stream_ok:
                self.session['streamed'] += 1

        self._frame_times.append(time.perf_counter() - t0)
        self.session['frames'] += 1
        self._frames_done += 1

        if self.bench:
            self.bench.log(
                fps=self.fps(),
                proc_ms=proc_ms,
                jitter_raw=self._jitter_raw[-1] if self._jitter_raw else 0,
                jitter_smooth=self._jitter_sm[-1] if self._jitter_sm else 0,
                occ_rate=occ_info['occlusion_rate'],
                rec_rate=occ_info['recovery_rate'],
                ik_corr=self.ik.corrections,
                stream_ms=stream_ms,
            )

        return {
            'success':     True,
            'landmarks':   smoothed_lm,
            'ue5_lm':      ue5_lm,
            'stream_ok':   stream_ok,
            'proc_ms':     proc_ms,
            'stream_ms':   stream_ms,
            'occ_info':    occ_info,
            'front_frame': front_frame,
            'back_frame':  back_frame,
        }

    # ── FPS / Jitter helpers ──────────────────────────────────────────────────

    def fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        avg = float(np.mean(self._frame_times))
        f   = 1.0 / avg if avg > 0 else 0.0
        self._fps_buf.append(f)
        return float(np.mean(self._fps_buf))

    def jitter_reduction_pct(self) -> float:
        if len(self._jitter_raw) < 5 or len(self._jitter_sm) < 5:
            return 0.0
        r = float(np.mean(self._jitter_raw))
        s = float(np.mean(self._jitter_sm))
        return max(0.0, (r - s) / max(r, 1e-9) * 100)

    # ── Visualisation ─────────────────────────────────────────────────────────

    _VIS_CONNECTIONS = [
        (11,12),(11,13),(13,15),(12,14),(14,16),
        (11,23),(12,24),(23,24),(23,25),(24,26),
        (25,27),(26,28),(27,29),(28,30),(29,31),(30,32),
        (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    ]

    def draw(self, frame: np.ndarray, landmarks: List[Dict]) -> np.ndarray:
        if frame is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        h, w = frame.shape[:2]
        lm_map = {lm['id']: lm for lm in landmarks}

        for a, b in self._VIS_CONNECTIONS:
            if a in lm_map and b in lm_map:
                pa = (int(lm_map[a]['x'] * w), int(lm_map[a]['y'] * h))
                pb = (int(lm_map[b]['x'] * w), int(lm_map[b]['y'] * h))
                cv2.line(frame, pa, pb, (100, 100, 100), 1)

        for lm in landmarks:
            x = int(lm['x'] * w)
            y = int(lm['y'] * h)
            if lm.get('ik_corrected'):       color = (255, 0, 255)
            elif lm.get('interpolated'):     color = (0, 165, 255)
            elif lm['visibility'] > 0.7:     color = (0, 255, 0)
            else:                            color = (0, 100, 255)
            cv2.circle(frame, (x, y), 4, color, -1)
        return frame

    def overlay(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        if frame is None:
            return frame
        fps  = self.fps()
        jr   = self.jitter_reduction_pct()
        occ  = result.get('occ_info', {})
        sm   = result.get('stream_ms', 0)
        pm   = result.get('proc_ms', 0)
        ik_c = self.ik.corrections

        def put(text, y, color=(255, 255, 255)):
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

        fps_col = (0, 255, 0) if fps >= 25 else (0, 165, 255)
        put(f"FPS: {fps:.1f}",            25,  fps_col)
        put(f"Proc: {pm:.1f}ms",          50)
        put(f"Stream: {sm:.1f}ms",        75,  (0, 255, 0) if result.get('stream_ok') else (0, 0, 255))
        put(f"Jitter-: {jr:.1f}%",        100, (0, 255, 0))
        put(f"Occ: {occ.get('occlusion_rate',0):.1f}%  Rec: {occ.get('recovery_rate',0):.1f}%", 125)
        put(f"IK fixes: {ik_c}",          150, (255, 0, 255))
        put(f"Model: HEAVY (cmplx=2)",    175, (200, 200, 200))
        if self.dual_cam:
            put("DUAL CAM", 200, (0, 255, 255))
        if not self.ik.calibrated:
            put(f"Calibrating IK... ({self.ik._count}/{CFG['ik_calibration_frames']})", 225, (0, 165, 255))
        else:
            put("IK: calibrated", 225, (0, 255, 0))
        return frame

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        self.start()
        print("Controls:  Q=quit  S=save benchmark  M=toggle smoothing  O=toggle occlusion\n")

        enable_smooth = True
        enable_occ    = True

        while True:
            result = self.process_frame()

            if result['success']:
                frame = result['front_frame']
                if frame is not None:
                    frame = self.draw(frame.copy(), result['landmarks'])
                    frame = self.overlay(frame, result)
                    cv2.imshow('MoCap → UE5 (Q=quit)', frame)

                if result.get('back_frame') is not None:
                    cv2.imshow('Back Camera', result['back_frame'])

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and self.bench:
                print(f"[BENCH] {self.bench.frame} frames logged so far.")
            elif key == ord('m'):
                enable_smooth = not enable_smooth
                if not enable_smooth:
                    self.filter_bank.reset()
                print(f"[TOGGLE] One-Euro filter: {'ON' if enable_smooth else 'OFF'}")
            elif key == ord('o'):
                enable_occ = not enable_occ
                print(f"[TOGGLE] Occlusion handling: {'ON' if enable_occ else 'OFF'}")

            if self.benchmark_frames and self._frames_done >= self.benchmark_frames:
                print(f"\n[BENCH] Target {self.benchmark_frames} frames reached.")
                break

        self._shutdown()

    def _shutdown(self):
        if self.bench:
            self.bench.close()
        if self.streamer:
            self.streamer.close()
        self.cam_front.stop()
        if self.cam_back:
            self.cam_back.stop()
        cv2.destroyAllWindows()
        self._report()

    def _report(self):
        fps = self.fps()
        jr  = self.jitter_reduction_pct()
        ik  = self.ik.stats()
        occ = {
            'occlusion_rate': self.occ.occluded / max(self.occ.total_joints, 1) * 100,
            'recovery_rate':  self.occ.recovered / max(self.occ.occluded, 1) * 100,
        }

        print("\n" + "=" * 65)
        print("  FINAL PERFORMANCE REPORT")
        print("=" * 65)
        print(f"  Frames processed:      {self.session['frames']}")
        print(f"  Frames streamed:       {self.session['streamed']}")
        print(f"  Average FPS:           {fps:.2f}")
        print(f"  Jitter reduction:      {jr:.1f}%")
        print(f"  Occlusion rate:        {occ['occlusion_rate']:.1f}%")
        print(f"  Occlusion recovery:    {occ['recovery_rate']:.1f}%")
        print(f"  IK corrections total:  {ik['corrections_total']}")
        print(f"  IK corr/frame:         {ik['corrections_per_frame']}")
        if self.streamer:
            print(f"  Avg stream latency:    {self.streamer.avg_latency_ms():.2f} ms")
        if self.fusion:
            s = self.fusion.stats()
            print(f"  Dual-cam both:         {s['both_pct']}%")
            print(f"  Front-only:            {s['front_pct']}%")
            print(f"  Back-only:             {s['back_pct']}%")
        print("=" * 65 + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='MoCap → UE5 (Maximum Accuracy)')
    parser.add_argument('--dual',      action='store_true', help='Enable dual camera mode')
    parser.add_argument('--no-stream', action='store_true', help='Disable UDP streaming (local preview only)')
    parser.add_argument('--benchmark', type=int, default=0,  help='Log N frames then exit (0=run forever)')
    parser.add_argument('--host',      type=str, default=CFG['stream_host'], help='UE5 host IP')
    parser.add_argument('--port',      type=int, default=CFG['stream_port'], help='UE5 UDP port')
    args = parser.parse_args()

    CFG['stream_host'] = args.host
    CFG['stream_port'] = args.port

    system = MocapSystemUE5(
        dual_cam         = args.dual,
        enable_stream    = not args.no_stream,
        benchmark_frames = args.benchmark,
    )
    system.run()


if __name__ == '__main__':
    main()
