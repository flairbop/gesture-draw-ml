import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
import time
from src.features import extract_features
from src.utils import FPS, Smoother

# Constants
MODEL_PATH = "models/gesture_model.pkl"
SCALER_PATH = "models/scaler.pkl"
CANVAS_COLOR = (255, 255, 255) # White drawing
THICKNESS = 4
ERASE_HOLD_TIME = 0.5

def main():
    # Load Model
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("Model not found. Please run 'python -m src.collect' then 'python -m src.train'.")
        return

    clf = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Setup MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    
    # Canvas
    ret, frame = cap.read()
    if not ret: return
    h, w, _ = frame.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    
    # State
    prev_point = None
    smoother = Smoother(window_size=8, threshold=0.7)
    fps_counter = FPS()
    
    erase_start_time = None
    current_state = "HOVER"
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Mirror
        frame = cv2.flip(frame, 1)
        # Keep canvas same size
        if frame.shape[:2] != canvas.shape[:2]:
            canvas = cv2.resize(canvas, (frame.shape[1], frame.shape[0]))
            
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = hands.process(rgb)
        
        predicted_label = "HOVER" # Default safe state
        confidence = 0.0
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0] # Max 1 hand
            
            # Draw Landmarks (Subtle)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Prediction
            features = extract_features(hand_landmarks.landmark)
            if features is not None:
                feat_scaled = scaler.transform([features])
                probs = clf.predict_proba(feat_scaled)[0]
                pred_idx = np.argmax(probs)
                raw_label = clf.classes_[pred_idx]
                raw_prob = probs[pred_idx]
                
                smoother.update(raw_label, raw_prob)
                stable_label, stable_prob = smoother.get_stable_prediction()
                
                if stable_label:
                    predicted_label = stable_label
                    confidence = stable_prob
            
            # Cursor Position (Index Tip)
            idx_tip = hand_landmarks.landmark[8]
            cx, cy = int(idx_tip.x * w), int(idx_tip.y * h)
            
            # State Machine
            if predicted_label == "DRAW":
                if prev_point is None:
                    prev_point = (cx, cy)
                
                cv2.line(canvas, prev_point, (cx, cy), CANVAS_COLOR, THICKNESS)
                prev_point = (cx, cy)
                erase_start_time = None
                
            elif predicted_label == "HOVER":
                prev_point = (cx, cy)
                erase_start_time = None
                # Visual cursor
                cv2.circle(frame, (cx, cy), 8, (0, 255, 255), 2)
                
            elif predicted_label == "ERASE":
                prev_point = None
                if erase_start_time is None:
                    erase_start_time = time.time()
                elif time.time() - erase_start_time > ERASE_HOLD_TIME:
                    canvas[:] = 0 # Clear
                    erase_start_time = None # Reset
                    
                # Visual Indicator for erase
                cv2.circle(frame, (cx, cy), 15, (0, 0, 255), 2)

            current_state = predicted_label

        else:
            # No hand
            prev_point = None
            erase_start_time = None
            current_state = "NO HAND"

        # Blend Canvas
        # Create mask of canvas to only overlay non-black pixels? 
        # Or just simple add (lines are white)
        gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_canvas, 10, 255, cv2.THRESH_BINARY)
        inv_mask = cv2.bitwise_not(mask)
        
        # Black out the lines in the frame
        frame_bg = cv2.bitwise_and(frame, frame, mask=inv_mask)
        # Take only the lines from canvas
        canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask)
        # Combine
        final_frame = cv2.add(frame_bg, canvas_fg)
        
        # Helper: Darken background slightly to pop the drawing? 
        # Optional: final_frame = cv2.addWeighted(final_frame, 0.8, canvas, 0.2, 0)
        
        # UI Overlay
        fps_counter.update()
        fps_val = fps_counter.get()
        
        cv2.putText(final_frame, f"STATE: {current_state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(final_frame, f"CONF: {confidence:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(final_frame, f"FPS: {int(fps_val)}", (w - 120, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(final_frame, "Q: Quit | R: Reset", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        
        cv2.imshow('Gesture Draw ML', final_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            canvas[:] = 0

    cap.release()
    cv2.destroyAllWindows()

# Fix typo in utils import if necessary (FPS class)
class FPS:
    def __init__(self, avg_len=30):
        self._prev_time = time.time()
        self._delays = list()
        self._avg_len = avg_len
    
    def update(self):
        curr_time = time.time()
        delay = curr_time - self._prev_time
        self._prev_time = curr_time
        self._delays.append(delay)
        if len(self._delays) > self._avg_len:
            self._delays.pop(0)
        
    def get(self):
        if not self._delays: return 0.0
        return 1.0 / (sum(self._delays) / len(self._delays))

if __name__ == "__main__":
    main()
