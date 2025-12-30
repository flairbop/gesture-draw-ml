# Gesture Draw ML

A premium, webcam-based paint application that tracks hand gestures to draw on screen. Powered by MediaPipe and Scikit-Learn with advanced stabilization.

## Features
- **Pinch to Draw**: Bring thumb and index finger together. Pressure (pinch tightness) modulates line width.
- **Smart Smoothing**: Uses 1Euro filters for jitter-free drawing and Robust Hysteresis for stable state switching.
- **Stroke Management**: Full Undo/Redo stack for both drawing and erasing.
- **Local Erase**: Fist gesture acts as a movable eraser brush (not just "clear all").
- **Robust ML**: Session-aware collection and training with probability calibration.

## Installation

1.  **Clone/Enter the repo**:
    ```bash
    cd gesture-draw-ml
    ```

2.  **Environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

## Workflow

### 1. Collect Data (`src.collect`)
Collect data for the 3 classes: DRAW, HOVER, ERASE.
```bash
python -m src.collect --fps 60
```
*   **Controls**:
    *   `1`: DRAW (Pinch). Record ~500-1000 frames.
    *   `2`: HOVER (Open Hand). Record ~500-1000 frames.
    *   `3`: ERASE (Fist). Record ~500-1000 frames.
    *   `SPACE`: Toggle recording.
    *   `N`: Start new session (rotate hand, move specific distance).
    *   `Q`: Quit.

### 2. Train Model (`src.train`)
Train the classifier (Logistic Regression by default).
```bash
python -m src.train --model logreg
```
*   Performs session-based validation split.
*   Calibrates probabilities.
*   Saves `models/gesture_model.pkl`.

### 3. Run App (`src.run_app`)
Start drawing.
```bash
python -m src.run_app
```
*   **Controls**:
    *   `Pinch`: Draw lines.
    *   `Open Hand`: Move cursor.
    *   `Fist`: Erase (Eraser Brush).
    *   `Z`: Undo last action (stroke or erasure).
    *   `[` / `]`: Decrease/Increase eraser size.
    *   `R`: Clear entire canvas.
    *   `S`: Save drawing to `outputs/`.
    *   `Q`: Quit.

## Technical Details
This project uses:
*   **Robust State Machine**: Requires sustained probability thresholding to switch states (e.g. 0.85 prob for >4 frames to enter Draw), eliminating flicker.
*   **Action Manager**: Stores every stroke and erase operation as a vector path, allowing infinite undo capability without bitrate loss.
*   **1Euro Filter**: Adapts smoothing based on speed—smooths jitter when slow, maintains low latency when fast.

## Troubleshooting
- **Jittery Lines?** Try enabling the filter (Key `F`) or ensure good lighting.
- **Can't Draw?** Ensure you are collecting data at 60FPS or consistent with your webcam speed.
- **Eraser too small?** Use `]` to increase the brush size.
