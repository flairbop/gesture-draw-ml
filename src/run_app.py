import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
import time
import json
import argparse
from datetime import datetime
from collections import deque

from src.features import HandFeatureExtractor
from src.utils import FPS, StateMachine, ViterbiDecoder, PointFilter

# Paths
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "gesture_model.pkl")
LABEL_ORDER_PATH = os.path.join(MODELS_DIR, "label_order.json")
OUTPUT_DIR = "outputs"

CANVAS_COLOR = (255, 255, 255) # White
BASE_THICKNESS = 4

class StrokeManager:
    """Manages drawing strokes for undo/redo."""
    def __init__(self):
        self.strokes = [] # List of (list of points, thickness)
        self.current_stroke = []
        self.current_thicknesses = []
    
    def add_point(self, point, thickness):
        self.current_stroke.append(point)
        self.current_thicknesses.append(thickness)
        
    def end_stroke(self):
        if self.current_stroke:
            # Optimize: Simplify stroke or keep raw? keeping raw for now
            self.strokes.append(list(zip(self.current_stroke, self.current_thicknesses)))
            self.current_stroke = []
            self.current_thicknesses = []
            
    def undo(self):
        if self.current_stroke:
            self.current_stroke = [] # Cancel current
            self.current_thicknesses = []
        elif self.strokes:
            self.strokes.pop()

    def clear(self):
        self.strokes = []
        self.current_stroke = []
        self.current_thicknesses = []

    def draw(self, canvas):
        canvas[:] = 0 # Clear buffer
        
        # Draw saved strokes
        for stroke_data in self.strokes:
            if len(stroke_data) < 2: continue
            pts = np.array([p[0] for p in stroke_data], dtype=np.int32)
            # Drawing variable width lines is tricky in OpenCV (polylines is constant)
            # Simplification: Draw segments
            for i in range(len(stroke_data) - 1):
                p1 = stroke_data[i][0]
                p2 = stroke_data[i+1][0]
                th = int(stroke_data[i][1])
                cv2.line(canvas, p1, p2, CANVAS_COLOR, th)

        # Draw current stroke
        if len(self.current_stroke) > 1:
            for i in range(len(self.current_stroke) - 1):
                p1 = self.current_stroke[i]
                p2 = self.current_stroke[i+1]
                th = int(self.current_thicknesses[i])
                cv2.line(canvas, p1, p2, CANVAS_COLOR, th)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--mirror', type=bool, default=True)
    parser.add_argument('--window', type=int, default=15, help="Viterbi window size")
    parser.add_argument('--filter', type=bool, default=True, help="Enable 1Euro filter")
    args = parser.parse_args()

    # Load Model
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Run training first.")
        return

    try:
        pipeline = joblib.load(MODEL_PATH)
        with open(LABEL_ORDER_PATH, 'r') as f:
            labels = json.load(f)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Setup Components
    extractor = HandFeatureExtractor(history_len=5)
    viterbi = ViterbiDecoder(labels, window_size=args.window)
    state_machine = StateMachine()
    point_filter = PointFilter(min_cutoff=0.1, beta=5.0) if args.filter else None # Tuned for smoothness
    strokes = StrokeManager()
    fps_counter = FPS()

    # MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(args.camera)
    
    # Canvas setup
    ret, frame = cap.read()
    if not ret: return
    h, w, _ = frame.shape
    canvas_mask = np.zeros((h, w, 3), dtype=np.uint8)

    print("=== Gesture Draw ML ===")
    print("Controls: Q=Quit, Z=Undo, R=Clear, S=Save, F=Toggle Filter")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if args.mirror:
            frame = cv2.flip(frame, 1)
            
        h, w, _ = frame.shape
        if canvas_mask.shape[:2] != (h, w):
            canvas_mask = cv2.resize(canvas_mask, (w, h))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        # Defaults
        curr_state = state_machine.state # default to last
        confidence = 0.0
        cursor_pos = None
        pinch_dist = 1.0 # default
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0] # Max 1
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Predict
            feats, extras = extractor.extract(hand_landmarks.landmark)
            if feats is not None:
                # pipeline handles scaling
                probs = pipeline.predict_proba([feats])[0]
                prob_dict = {label: val for label, val in zip(labels, probs)}
                
                # Update Stabilizers
                viterbi.update(probs)
                v_state = viterbi.decode()
                if v_state:
                    curr_state = state_machine.update(v_state, prob_dict)
                    confidence = prob_dict.get(curr_state, 0.0)

                # Cursor Logic
                if 'index_tip' in extras:
                    norm_pos = extras['index_tip']
                    raw_x, raw_y = int(norm_pos[0]*w), int(norm_pos[1]*h)
                    
                    if point_filter:
                        cursor_pos = point_filter.update((raw_x, raw_y))
                        cursor_pos = (int(cursor_pos[0]), int(cursor_pos[1]))
                    else:
                        cursor_pos = (raw_x, raw_y)

                    # Dynamic Width
                    pd = extras.get('pinch_dist', 0.2)
                    # Mapping: tight pinch (0.05) -> thin (1), wide (0.15) -> thick (6)? 
                    # Usually tight pinch = draw efficiently. 
                    # Let's simple inverse: smaller pinch = slightly thinner?
                    # Or pressure simulation: smaller pinch = harder press = thicker?
                    # "Smaller pinch distance => slightly thinner line" per prompt
                    width_factor = max(1, min(10, int(pd * 20))) 
                    # Actually if pinch is really small (drawing), we want it stable.
                    # Let's just fix it to 'pressure' style
                    brush_width = max(1, 6 - int(pd * 20)) # 0.05 -> 5, 0.2 -> 2
                    if brush_width < 1: brush_width = 1

        else:
            # No hand
            pass

        # Execution
        if curr_state == "DRAW" and cursor_pos:
            strokes.add_point(cursor_pos, brush_width)
            # Visual feedback
            cv2.circle(frame, cursor_pos, 3, (0, 0, 255), -1)
            
        elif curr_state == "HOVER":
            strokes.end_stroke()
            if cursor_pos:
                cv2.circle(frame, cursor_pos, 5, (0, 255, 255), 2)
                
        elif curr_state == "ERASE":
            strokes.end_stroke()
            # If held for X time (handled by state machine transition), but visual effect:
            # State machine only enters ERASE after hold. So if we are HERE, we erase.
            # But the user spec said "hold... to clear". 
            # StateMachine enters ERASE after hold. When in ERASE, we clear?
            # Or does 'ERASE' mean 'Eraser Tool'? The spec said "clear canvas".
            # So if we are in ERASE state, we clear everything once.
            if len(strokes.strokes) > 0:
                strokes.clear()
            
            cv2.putText(frame, "ERASED", (w//2 - 50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

        else:
            strokes.end_stroke()

        # Render Strokes
        strokes.draw(canvas_mask)
        
        # Composite
        # Black out lines in frame
        gray = cv2.cvtColor(canvas_mask, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        inv_mask = cv2.bitwise_not(mask)
        
        frame_bg = cv2.bitwise_and(frame, frame, mask=inv_mask)
        final_frame = cv2.add(frame_bg, canvas_mask)
        
        # UI
        fps_counter.update()
        fps = int(fps_counter.get())
        
        cv2.putText(final_frame, f"{curr_state} ({confidence:.2f})", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(final_frame, f"FPS: {fps} | Filt: {'ON' if point_filter else 'OFF'}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(final_frame, "[Z]Undo [S]Save [R]Clear [Q]Quit", (15, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        cv2.imshow('Gesture Draw ML', final_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('z'):
            strokes.undo()
        elif key == ord('r'):
            strokes.clear()
        elif key == ord('f'):
            # Toggle filter
            if point_filter: 
                point_filter = None
            else:
                point_filter = PointFilter(min_cutoff=0.1, beta=5.0)
        elif key == ord('s'):
            # Save
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = os.path.join(OUTPUT_DIR, f"drawing_{ts}.png")
            cv2.imwrite(fname, canvas_mask)
            print(f"Saved to {fname}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
