import numpy as np
import math
from collections import deque

def get_dist(p1, p2):
    """Euclidean distance between two landmarks."""
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def get_angle(a, b, c):
    """
    Calculate angle at b given points a, b, c.
    Returns angle in degrees.
    """
    ang = math.degrees(math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x))
    return ang + 360 if ang < 0 else ang

class HandFeatureExtractor:
    def __init__(self, history_len=5):
        self.history_len = history_len
        # Store history of shape (N_scalar_features,)
        self.history = deque(maxlen=history_len)
        self.feature_names = [] 
        
    def extract(self, landmarks, handedness_score=1.0):
        """
        Extracts robust features from landmarks.
        landmarks: list of 21 normalized landmarks
        handedness_score: 0.0 for left, 1.0 for right (approx) or purely metric
        """
        if not landmarks or len(landmarks) < 21:
            return None, {}
            
        # 1. Basic Landmarks
        wrist = landmarks[0]
        thumb_cmc = landmarks[1]
        thumb_mcp = landmarks[2]
        thumb_tip = landmarks[4]
        index_mcp = landmarks[5]
        index_pip = landmarks[6]
        index_tip = landmarks[8]
        middle_mcp = landmarks[9]
        middle_pip = landmarks[10]
        middle_tip = landmarks[12]
        ring_mcp = landmarks[13]
        ring_pip = landmarks[14]
        ring_tip = landmarks[16]
        pinky_mcp = landmarks[17]
        pinky_pip = landmarks[18]
        pinky_tip = landmarks[20]

        # Scale Factor
        hand_scale = get_dist(wrist, middle_mcp)
        if hand_scale < 1e-6: hand_scale = 1.0

        feats = []
        
        # --- SCALAR BASE FEATURES ---
        
        # A. Distances (Normalized)
        # Pinch
        pinch_dist = get_dist(thumb_tip, index_tip) / hand_scale
        feats.append(pinch_dist)
        
        # Finger Curls (Tip to MCP)
        feats.append(get_dist(index_tip, index_mcp) / hand_scale)
        feats.append(get_dist(middle_tip, middle_mcp) / hand_scale)
        feats.append(get_dist(ring_tip, ring_mcp) / hand_scale)
        feats.append(get_dist(pinky_tip, pinky_mcp) / hand_scale)
        
        # Tips to Wrist
        feats.append(get_dist(thumb_tip, wrist) / hand_scale)
        feats.append(get_dist(index_tip, wrist) / hand_scale)
        feats.append(get_dist(middle_tip, wrist) / hand_scale)
        feats.append(get_dist(ring_tip, wrist) / hand_scale)
        feats.append(get_dist(pinky_tip, wrist) / hand_scale)
        
        # Palm Spread
        feats.append(get_dist(index_mcp, pinky_mcp) / hand_scale)
        
        # Internal Thumb Check
        feats.append(get_dist(thumb_tip, pinky_mcp) / hand_scale)

        # B. Angles (Degrees / 360) - roughly normalized 0-1
        # Thumb Angle (CMC, MCP, TIP) - usually straight when pinch, bent when fist
        feats.append(abs(get_angle(thumb_cmc, thumb_mcp, thumb_tip) - 180) / 180.0)
        
        # Finger Curl Angles (MCP, PIP, TIP)
        feats.append(abs(get_angle(index_mcp, index_pip, index_tip) - 180) / 180.0)
        feats.append(abs(get_angle(middle_mcp, middle_pip, middle_tip) - 180) / 180.0)
        feats.append(abs(get_angle(ring_mcp, ring_pip, ring_tip) - 180) / 180.0)
        feats.append(abs(get_angle(pinky_mcp, pinky_pip, pinky_tip) - 180) / 180.0)

        # C. Handedness (Binary-ish)
        feats.append(float(handedness_score))

        # --- TEMPORAL FEATURES ---
        
        current_vec = np.array(feats, dtype=np.float32)
        
        # Store for velocity calc
        # We track specific metrics for velocity: Pinch, Wrist X/Y
        metrics = np.array([pinch_dist, wrist.x, wrist.y], dtype=np.float32)
        
        if len(self.history) > 0:
            # Velocity: (Current - Oldest) / Steps
            # We compare against the oldest in buffer for smoother delta
            prev_metrics = self.history[0]
            steps = len(self.history)
            deltas = (metrics - prev_metrics) / steps
        else:
            deltas = np.zeros_like(metrics)
            
        self.history.append(metrics)
        
        # Combine
        final_features = np.concatenate([current_vec, deltas])
        
        # Extras for app usage
        extras = {
            'pinch_dist': pinch_dist,
            'hand_scale': hand_scale,
            'wrist': (wrist.x, wrist.y),
            'index_tip': (index_tip.x, index_tip.y)
        }
        
        return final_features, extras
