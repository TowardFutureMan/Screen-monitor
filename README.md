# Screen Monitor (mac camera)

This tool monitors your head pose and face position from the mac camera. If your
gaze/body is outside the screen range for a continuous duration (default 10
seconds), it triggers an alert.

## Requirements

- macOS with camera access allowed for Terminal/Python
- Python 3.9+

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python monitor.py
```

Press `q` to quit the preview window.

## Useful options

```bash
python monitor.py \
  --offscreen-seconds 10 \
  --yaw-threshold 25 \
  --pitch-threshold 20 \
  --min-face-area 0.06 \
  --max-face-area 0.50 \
  --max-center-offset 0.22
```

To run without a preview window:

```bash
python monitor.py --no-preview
```

## How it works

The app uses MediaPipe face landmarks to estimate head pose (yaw/pitch/roll),
checks face size (distance from the camera), and verifies the face is centered.
If the face is missing or outside these thresholds for 10 seconds, it triggers
an alert using macOS notifications and voice.
