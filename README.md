# Gesture Draw ML

A local, webcam-based paint application that uses computer vision and machine learning logic to let you draw on your screen with hand gestures.

## Features
- **Pinch to Draw**: Bring thumb and index finger together to draw.
- **Open Hand to Hover**: Move the cursor without drawing.
- **Fist to Erase**: Hold a fist (or just tuck thumb) for >0.5s to clear the canvas.
- **Customizable**: Train the model on *your* specific hand gestures in under a minute.

## Installation

1.  **Clone/Enter the repo**:
    ```bash
    cd gesture-draw-ml
    ```

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Mac/Linux
    # .venv\Scripts\activate   # Windows
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Workflow

### 1. Collect Data
You need to teach the machine what your gestures look like.
```bash
./scripts/run_collect.sh
# or python -m src.collect
```
*   **Controls**:
    *   `1`: Set label to **DRAW** (Make a pinch gesture).
    *   `2`: Set label to **HOVER** (Relaxed open hand).
    *   `3`: Set label to **ERASE** (Fist).
    *   `SPACE`: Toggle recording.
    *   `Q`: Save and Quit.
*   **Tips**:
    *   Collect ~500-1000 frames per class.
    *   Move your hand around the screen, vary rotation slightly, move closer/further.
    *   Ensure good lighting.

### 2. Train Model
Train a Random Forest classifier on your collected data.
```bash
./scripts/run_train.sh
# or python -m src.train
```
This should take only a few seconds. It will save the model to `models/gesture_model.pkl`.

### 3. Run App
Start drawing!
```bash
./scripts/run_app.sh
# or python -m src.run_app
```
*   **Controls**:
    *   `Q`: Quit.
    *   `R`: Clear canvas immediately.
*   **Behavior**:
    *   The cursor follows your index finger.
    *   If you pinch, it draws.
    *   If you make a fist/erase gesture, a red indicator appears; hold it to wipe the screen.

## Troubleshooting
-   **Webcam not found**: Ensure no other app is using the camera. The app defaults to index `0`.
-   **Jittery drawing**: Collect more data! Ensure "Hover" data looks distinctly different from "Draw" (thumb separated vs touching).
-   **Model accuracy**: Check the output of the training script. If accuracy is low (< 90%), your gestures might be too ambiguous.
