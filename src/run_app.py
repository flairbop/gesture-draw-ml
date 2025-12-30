import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
import time
import json
import argparse
from datetime import datetime

from src.features import HandFeatureExtractor
from src.utils import FPS, StateMachine, ViterbiDecoder, PointFilter

MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "gesture_model.pkl")
LABEL_ORDER_PATH = os.path.join(MODELS_DIR, "label_order.json")
OUTPUT_DIR = "outputs"

CANVAS_COLOR = (255, 255, 255) # White drawing

class ActionManager:
    """Manages drawing actions (Strokes and Erasures) for undo/redo."""
    def __init__(self):
        self.actions = [] # List of dicts: {'type': 'stroke'|'erase', 'points': [(x,y)], 'sizes': [int]}
        self.current_action = None 
    
    def start_action(self, action_type):
        self.current_action = {'type': action_type, 'points': [], 'sizes': []}
        
    def add_point(self, point, size):
        if self.current_action:
            self.current_action['points'].append(point)
            self.current_action['sizes'].append(size)
            
    def end_action(self):
        if self.current_action and len(self.current_action['points']) > 0:
            self.actions.append(self.current_action)
        self.current_action = None

    def undo(self):
        if self.current_action:
            self.current_action = None
        elif self.actions:
            self.actions.pop()

    def clear(self):
        self.actions = []
        self.current_action = None

    def render(self, canvas):
        """
        Replays all actions onto the canvas.
        Canvas is expected to be a blank (black) image buffer.
        """
        canvas[:] = 0 
        
        # Helper to draw a set of points
        def draw_points(pts, sizes, is_erase):
            color = (0, 0, 0) if is_erase else CANVAS_COLOR
            # If erase, we draw thick circles or lines. Since actions may be fast, lines are better for continuity.
            # But 'erase' usually implies 'removing' so drawing black on the mask.
            
            if len(pts) < 2:
                # Single point
                if is_erase:
                    cv2.circle(canvas, pts[0], sizes[0], color, -1)
                else:
                    cv2.line(canvas, pts[0], pts[0], color, sizes[0]) # Dot
                return

            for i in range(len(pts) - 1):
                p1 = pts[i]
                p2 = pts[i+1]
                th = sizes[i]
                if is_erase:
                    # Erase needs to be a 'circle' swept along line
                    cv2.line(canvas, p1, p2, color, th*2) # Thicker line for erase coverage
                    cv2.circle(canvas, p1, th, color, -1) # Round caps
                else:
                    cv2.line(canvas, p1, p2, color, th)
                    
        # 1. Render history
        for action in self.actions:
            is_erase = (action['type'] == 'erase')
            draw_points(action['points'], action['sizes'], is_erase)

        # 2. Render current
        if self.current_action:
            is_erase = (self.current_action['type'] == 'erase')
            draw_points(self.current_action['points'], self.current_action['sizes'], is_erase)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--mirror', type=bool, default=True)
    parser.add_argument('--window', type=int, default=15)
    parser.add_argument('--filter', type=bool, default=True)
    args = parser.parse_args()

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

    extractor = HandFeatureExtractor(history_len=5)
    viterbi = ViterbiDecoder(labels, window_size=args.window)
    state_machine = StateMachine() # Uses new defaults
    point_filter = PointFilter(min_cutoff=0.1, beta=5.0) if args.filter else None
    
    actions = ActionManager()
    fps_counter = FPS()
    
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(args.camera)
    
    ret, frame = cap.read()
    if not ret: return
    h, w, _ = frame.shape
    canvas_mask = np.zeros((h, w, 3), dtype=np.uint8)

    eraser_radius = 20
    print("Controls: Z=Undo, [/]=Eraser Size, R=Clear All, S=Save, Q=Quit")

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
        
        curr_state = state_machine.state
        confidence = 0.0
        cursor_pos = None
        pinch_dist = 1.0
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            feats, extras = extractor.extract(hand_landmarks.landmark)
            if feats is not None:
                probs = pipeline.predict_proba([feats])[0]
                prob_dict = {label: val for label, val in zip(labels, probs)}
                
                viterbi.update(probs)
                v_state = viterbi.decode()
                if v_state:
                    curr_state = state_machine.update(prob_dict) # v_state implicit in robust machine update if needed? 
                    # Actually new StateMachine only takes prob_dict and tracks history. 
                    # It ignores viterbi unless we want to combine.
                    # The prompt said "improve state logic... hysteresis".
                    # My new StateMachine logic uses P(DRAW) history.
                    # We can use viterbi as a gate, but the sustained prob check is stronger.
                    # Let's trust the RobustStateMachine as the primary arbiter.
                    pass
                
                # Confidence implies current state prob
                confidence = prob_dict.get(curr_state, 0.0)

                if 'index_tip' in extras:
                    norm_pos = extras['index_tip']
                    raw_x, raw_y = int(norm_pos[0]*w), int(norm_pos[1]*h)
                    
                    if point_filter:
                        cursor_pos = point_filter.update((raw_x, raw_y))
                        cursor_pos = (int(cursor_pos[0]), int(cursor_pos[1]))
                    else:
                        cursor_pos = (raw_x, raw_y)
                    
                    pinch_dist = extras.get('pinch_dist', 0.2)

        # State Action Logic
        if curr_state == "DRAW" and cursor_pos:
            if not actions.current_action or actions.current_action['type'] != 'stroke':
                actions.end_action() # Finish prev if any
                actions.start_action('stroke')
            
            # Dynamic width
            brush_width = max(1, 6 - int(pinch_dist * 20)) 
            if brush_width < 1: brush_width = 1
            
            actions.add_point(cursor_pos, brush_width)
            cv2.circle(frame, cursor_pos, 3, (0, 0, 255), -1)

        elif curr_state == "ERASE" and cursor_pos:
            if not actions.current_action or actions.current_action['type'] != 'erase':
                actions.end_action()
                actions.start_action('erase')
                
            actions.add_point(cursor_pos, eraser_radius)
            # Visual cursor for eraser
            cv2.circle(frame, cursor_pos, eraser_radius, (0,0,255), 1)
            cv2.putText(frame, "ERASE", (cursor_pos[0]+10, cursor_pos[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)

        else:
            # Hover or Lost Hand
            actions.end_action()
            if cursor_pos:
                cv2.circle(frame, cursor_pos, 5, (0, 255, 255), 2)

        # Render
        actions.render(canvas_mask)

        # Composite
        gray = cv2.cvtColor(canvas_mask, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        inv_mask = cv2.bitwise_not(mask)
        frame_bg = cv2.bitwise_and(frame, frame, mask=inv_mask)
        final_frame = cv2.add(frame_bg, canvas_mask)
        
        # UI
        fps_counter.update()
        fps_val = int(fps_counter.get())
        
        cv2.putText(final_frame, f"{curr_state} ({confidence:.2f})", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(final_frame, f"FPS: {fps_val}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        hint = f"Z:Undo R:Clear [/]:Size({eraser_radius}) Q:Quit"
        cv2.putText(final_frame, hint, (15, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        cv2.imshow('Gesture Draw ML', final_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('z'):
            action_ended = True if not actions.current_action else False # Only undo if idle? Or cancel current?
            # User usually expects Z to undo last COMPLETED stroke.
            # If currently drawing, maybe cancel current?
            actions.undo()
        elif key == ord('r'):
            actions.clear()
        elif key == ord('s'):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = os.path.join(OUTPUT_DIR, f"drawing_{ts}.png")
            cv2.imwrite(fname, canvas_mask)
            print(f"Saved {fname}")
        elif key == ord('['):
            eraser_radius = max(5, eraser_radius - 5)
        elif key == ord(']'):
            eraser_radius = min(60, eraser_radius + 5)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
