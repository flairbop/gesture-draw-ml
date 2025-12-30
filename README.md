# Gesture Draw ML

A premium, webcam-based paint application that tracks hand gestures to draw on screen. Powered by MediaPipe and Scikit-Learn with advanced stabilization.

## Features
- **Pinch to Draw**: Bring thumb and index finger together. Pressure (pinch tightness) modulates line width.
- **Smart Smoothing**: Uses Viterbi decoding and 1Euro filters for jitter-free drawing.
- **Stroke Management**: Undo (`Z`), Clear (`R`), and Save (`S`) functionality.
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
python -m src.collect
```
*   **Controls**:
    *   `1`: DRAW (Pinch). Record ~300 frames.
    *   `2`: HOVER (Open Hand). Record ~300 frames.
    *   `3`: ERASE (Fist). Record ~300 frames.
    *   `SPACE`: Toggle recording.
    *   `N`: Start new session (rotate hand, move specific distance).
    *   `Q`: Quit.

### 2. Train Model (`src.train`)
Train the classifier (Logistic Regression by default).
```bash
python -m src.train --model logreg
```
*   Performs session-based validation split.
*   Calibrates probabilities for better stability.
*   Saves `models/gesture_model.pkl`.

### 3. Run App (`src.run_app`)
Start drawing.
```bash
python -m src.run_app --window 15 --filter True
```
*   **Controls**:
    *   `Pinch`: Draw.
    *   `Open Hand`: Hover.
    *   `Fist`: Hold to clear canvas.
    *   `Z`: Undo last stroke.
    *   `R`: Clear canvas.
    *   `S`: Save drawing to `outputs/`.
    *   `F`: Toggle cursor smoothing.
    *   `Q`: Quit.

## Improved Stability
This project uses two layers of stabilization:
1.  **Viterbi Decoder**: Smooths the sequence of predicted states (Draw/Hover) to prevent flickering.
2.  **1Euro Filter**: Smooths the cursor coordinates to remove hand jitter while maintaining responsiveness.

## Troubleshooting
- **Jittery Lines?** Try enabling the filter (Key `F`) or ensure good lighting.
- **Wrong State?** Collect more data with the `N` key features (new session) to capture different angles.
