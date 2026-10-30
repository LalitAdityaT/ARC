# ARC — Adaptive Realtime Capture

Real-time, camera-based motion capture with a 3D skeleton viewer and a JSON stream for Unreal Engine 5 (UE5). The capture pipeline uses MediaPipe Pose Landmarker, temporal filtering, occlusion handling, inverse kinematics, and optional two-camera fusion.

## Features

- Live body-pose capture from one or two cameras
- Smoothing, occlusion recovery, and IK-based skeleton correction
- UDP streaming to `127.0.0.1:8888` by default, with optional TCP
- Standalone Pygame viewer that displays the incoming skeleton
- Optional per-frame benchmark output in `benchmark_log.csv`

## Requirements

- Windows
- Python 3.10 or newer (the project was developed with Python 3.10)
- A webcam; a second camera is optional
- A compatible UE5 receiver/plugin if sending data to Unreal Engine

Install the Python dependencies from the project directory:

```powershell
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The MediaPipe model, [`pose_landmarker_heavy.task`](./pose_landmarker_heavy.task), is included. If it is missing, the capture program downloads it when it starts.

## Run

Start motion capture with the default camera and stream to UE5:

```powershell
python mocap_ue5_final.py
```

Run the standalone skeleton viewer in a **second terminal** to view the UDP stream:

```powershell
python arc_skeleton_viewer.py
```

The batch launchers [`run_mocap.bat`](./run_mocap.bat) and [`run_viewer.bat`](./run_viewer.bat) run these programs using `venv\Scripts\python.exe`.

### Capture options

```text
python mocap_ue5_final.py --dual
python mocap_ue5_final.py --no-stream
python mocap_ue5_final.py --benchmark 300
python mocap_ue5_final.py --host 192.168.1.50 --port 8888
```

- `--dual`: enable the second camera (camera IDs are configured in [`config/settings.py`](./config/settings.py)).
- `--no-stream`: show the local camera preview without sending stream data.
- `--benchmark N`: process `N` frames and save metrics to `benchmark_log.csv`.
- `--host` and `--port`: set the stream destination; defaults are `127.0.0.1` and `8888`.

The UE5 receiver is not included in this repository. Configure a compatible receiver to accept the stream at the same host and port. The sender transmits JSON skeleton packets over UDP by default.

### Controls

**Capture preview:** `Q` quits, `M` toggles smoothing, and `O` toggles occlusion handling.

**Skeleton viewer:** `Esc` quits, arrow keys rotate the view, `R` toggles auto-rotation, `Space` resets the view, `L` toggles joint labels, and `C` toggles coordinates.

## Configuration

Camera IDs, capture dimensions, detection confidence, stream settings, and viewer defaults are in [`config/settings.py`](./config/settings.py). The default front and back camera IDs are `0` and `2`.

## License

This project is distributed under the [MIT License](./LICENSE).
