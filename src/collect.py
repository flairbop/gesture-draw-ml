import cv2
import mediapipe as mp
import csv
import time
import os
import uuid
import numpy as np
from src.features import HandFeatureExtractor

# Configurations
DATA_FILE = "data/gestures.csv"
CLASSES = {
    '1': "DRAW",
    '2': "HOVER",
    '3': "ERASE"
}

def get_dummy_features():
    # Helper to determine feature length dynamically
    ext = HandFeatureExtractor()
    # Create dummy landmarks
    from collections import namedtuple
    P = namedtuple('P', ['x', 'y'])
    lms = [P(0.5, 0.5)] * 21
    feats, _ = ext.extract(lms)
    return len(feats)

FEATURE_LEN = get_dummy_features()
HEADER = ["session_id", "timestamp", "label"] + [f"f{i}" for i in range(FEATURE_LEN)]

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

    # Initialization
    extractor = HandFeatureExtractor(history_len=5)
    
    current_label = '2'
    is_recording = False
    data_buffer = []
    
    # Session ID
    session_id = str(uuid.uuid4())[:8]
    
    # Timing
    last_record_time = 0
    record_interval = 1.0 / 15.0 # 15 FPS
    
    # Counts
    session_counts = {k: 0 for k in CLASSES.keys()}
    total_counts = {k: 0 for k in CLASSES.keys()}
    
    # Load totals if exists
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'r') as f:
                reader = csv.reader(f)
                next(reader, None) # header
                for row in reader:
                    lbl = row[2] # 3rd col is label name
                    # Reverse map
                    for k, v in CLASSES.items():
                        if v == lbl:
                            total_counts[k] += 1
        except:
            pass

    print("=== Gesture Data Collector ===")
    print(f"Session ID: {session_id}")
    print("Controls: 1=DRAW, 2=HOVER, 3=ERASE")
    print("SPACE=Toggle Record, N=New Session, Q=Quit")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = hands.process(rgb)
        
        feats = None
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            # Check handedness just for feature input (default 1.0/Right if unknown)
            # multi_handedness is available
            handedness = 1.0
            if results.multi_handedness:
                # Label is 'Left' or 'Right', score is confidence
                # We can encode Right=1, Left=0
                lbl = results.multi_handedness[0].classification[0].label
                handedness = 1.0 if lbl == 'Right' else 0.0

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract features (stateful)
            feats, extras = extractor.extract(hand_landmarks.landmark, handedness)
            
            # Draw cursor
            if 'index_tip' in extras and extras['index_tip']:
                ix, iy = extras['index_tip']
                cv2.circle(frame, (int(ix*w), int(iy*h)), 5, (0, 255, 0), -1)

            # Record
            if is_recording and feats is not None:
                curr_time = time.time()
                if curr_time - last_record_time > record_interval:
                    lbl_name = CLASSES[current_label]
                    row = [session_id, curr_time, lbl_name] + feats.tolist()
                    data_buffer.append(row)
                    session_counts[current_label] += 1
                    total_counts[current_label] += 1
                    last_record_time = curr_time

        # UI Overlay
        # Box background for text
        cv2.rectangle(frame, (0,0), (300, 160), (0,0,0), -1)
        
        # Status
        mode_txt = "REC" if is_recording else "IDLE"
        mode_col = (0, 0, 255) if is_recording else (100, 100, 100)
        cv2.putText(frame, f"[{mode_txt}] Sess: {session_id}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_col, 2)
        
        cv2.putText(frame, f"TARGET: {CLASSES[current_label]}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        y = 75
        for k, v in CLASSES.items():
            sc = session_counts[k]
            tc = total_counts[k]
            col = (0, 255, 0) if k == current_label else (150, 150, 150)
            cv2.putText(frame, f"{v}: {sc} (Tot: {tc})", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
            y += 20
            
        cv2.putText(frame, "Space:Rec N:New Q:Quit", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow('Gesture Data Collector', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in [ord('1'), ord('2'), ord('3')]:
            current_label = chr(key)
        elif key == ord(' '):
            is_recording = not is_recording
        elif key == ord('n'):
            # New session
            session_id = str(uuid.uuid4())[:8]
            session_counts = {k: 0 for k in CLASSES.keys()}
            # Also reset extractor history for clean break
            extractor = HandFeatureExtractor(history_len=5)
            print(f"--- New Session: {session_id} ---")

    cap.release()
    cv2.destroyAllWindows()
    
    # Save
    if data_buffer:
        print(f"Saving {len(data_buffer)} samples...")
        file_exists = os.path.exists(DATA_FILE)
        with open(DATA_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(HEADER)
            writer.writerows(data_buffer)
        print("Done.")

if __name__ == "__main__":
    main()
