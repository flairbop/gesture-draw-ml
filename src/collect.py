import cv2
import mediapipe as mp
import csv
import time
import os
import numpy as np
from src.features import extract_features

# Configurations
DATA_FILE = "data/gestures.csv"
CLASSES = {
    '1': "DRAW",
    '2': "HOVER",
    '3': "ERASE"
}
HEADER = ["timestamp", "label"] + [f"f{i}" for i in range(12)] # 12 features from extract_features

def main():
    # Setup MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # State variables
    current_label = '2' # Default to HOVER
    is_recording = False
    data_buffer = []
    class_counts = {k: 0 for k in CLASSES.keys()}
    
    # Load existing counts if file exists
    if os.path.exists(DATA_FILE):
        try:
            pass # simplified re-counting unimplemented for now, just append mode
        except Exception:
            pass

    print("=== Gesture Data Collector ===")
    print("Controls:")
    print("  1: DRAW | 2: HOVER | 3: ERASE")
    print("  SPACE: Toggle Recording")
    print("  C: Clear Session Buffer")
    print("  Q: Quit and Save")

    last_record_time = 0
    record_interval = 1.0 / 15.0 # Max 15 FPS recording

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Mirror for natural feel
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Convert to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        features = None
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Draw index tip indicator
                idx_tip = hand_landmarks.landmark[8]
                cx, cy = int(idx_tip.x * w), int(idx_tip.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                
                features = extract_features(hand_landmarks.landmark)
                
                # Record data
                if is_recording and features is not None:
                    curr_time = time.time()
                    if curr_time - last_record_time > record_interval:
                        row = [curr_time, CLASSES[current_label]] + features.tolist()
                        data_buffer.append(row)
                        class_counts[current_label] += 1
                        last_record_time = curr_time

        # UI Overlay
        # Mode Status
        mode_color = (0, 0, 255) if is_recording else (0, 255, 255) # Red if recording, Yellow if idle
        mode_text = "RECORDING" if is_recording else "IDLE (Press SPACE)"
        cv2.putText(frame, f"STATUS: {mode_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        
        # Current Label
        label_text = f"TARGET: {CLASSES[current_label]} (Key {current_label})"
        cv2.putText(frame, label_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Counts
        y_off = 90
        for k, v in CLASSES.items():
            cnt = class_counts[k]
            sess_cnt = sum(1 for row in data_buffer if row[1] == v)
            txt = f"{v}: {sess_cnt} session"
            col = (0, 255, 0) if k == current_label else (200, 200, 200)
            cv2.putText(frame, txt, (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 1)
            y_off += 25
            
        cv2.imshow('Gesture Data Collector', frame)
        
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        elif key in [ord('1'), ord('2'), ord('3')]:
            current_label = chr(key)
        elif key == ord(' '):
            is_recording = not is_recording
        elif key == ord('c'):
            data_buffer = []
            class_counts = {k: 0 for k in CLASSES.keys()}
            print("Buffer cleared.")

    cap.release()
    cv2.destroyAllWindows()
    
    # Save Data
    if data_buffer:
        print(f"Saving {len(data_buffer)} samples to {DATA_FILE}...")
        file_exists = os.path.exists(DATA_FILE)
        
        with open(DATA_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(HEADER)
            writer.writerows(data_buffer)
        print("Done.")

if __name__ == "__main__":
    main()
