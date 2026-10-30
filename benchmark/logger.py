"""
ARC — Benchmark Logger
========================
Logs per-frame metrics to CSV for statistical analysis (IEEE paper).

Columns: frame, timestamp, fps, proc_ms, jitter_raw, jitter_smoothed,
         occlusion_rate, recovery_rate, ik_corrections, stream_latency_ms
"""

import csv
import time


class BenchmarkLogger:
    def __init__(self, path: str = 'benchmark_log.csv'):
        self.path  = path
        self._f    = open(path, 'w', newline='')
        self._w    = csv.writer(self._f)
        self._w.writerow([
            'frame', 'ts', 'fps', 'proc_ms',
            'jitter_raw', 'jitter_smoothed',
            'occ_rate', 'rec_rate',
            'ik_corrections', 'stream_ms',
        ])
        self.frame = 0
        print(f"[BENCH] Logging to {path}")

    def log(
        self,
        fps: float,
        proc_ms: float,
        jitter_raw: float,
        jitter_smooth: float,
        occ_rate: float,
        rec_rate: float,
        ik_corr: int,
        stream_ms: float,
    ):
        self._w.writerow([
            self.frame,
            round(time.time(), 4),
            round(fps, 2),
            round(proc_ms, 3),
            round(jitter_raw, 6),
            round(jitter_smooth, 6),
            round(occ_rate, 2),
            round(rec_rate, 2),
            ik_corr,
            round(stream_ms, 3),
        ])
        self.frame += 1

    def close(self):
        self._f.close()
        print(f"[BENCH] Saved {self.frame} rows → {self.path}")
