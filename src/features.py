import numpy as np
import math

def get_dist(p1, p2):
    """Euclidean distance between two landmarks (ignoring z if 2D)."""
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def extract_features(landmarks):
    """
    Extracts scale-invariant feature vector from MediaPipe hand landmarks.
    
    Args:
        landmarks: mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList
                   or list of objects with x, y attributes.
    
    Returns:
        np.array: Feature vector of shape (N,) or None if invalid.
    """
    if not landmarks or len(landmarks) < 21:
        return None

    # MediaPipe Landmark Indices:
    # 0: WRIST
    # 1-4: THUMB (CMC, MCP, IP, TIP)
    # 5-8: INDEX (MCP, PIP, DIP, TIP)
    # 9-12: MIDDLE
    # 13-16: RING
    # 17-20: PINKY
    
    wrist = landmarks[0]
    thumb_tip = landmarks[4]
    index_mcp = landmarks[5]
    index_tip = landmarks[8]
    middle_mcp = landmarks[9]
    middle_tip = landmarks[12]
    ring_mcp = landmarks[13]
    ring_tip = landmarks[16]
    pinky_mcp = landmarks[17]
    pinky_tip = landmarks[20]

    # Hand scale reference: Wrist to Middle MCP
    # This is a relatively stable metric for hand size in the frame.
    hand_scale = get_dist(wrist, middle_mcp)
    
    # Avoid division by zero
    if hand_scale < 1e-6:
        hand_scale = 1.0

    features = []

    # 1. Pinch distance (Thumb Tip <-> Index Tip)
    features.append(get_dist(thumb_tip, index_tip) / hand_scale)

    # 2. Finger Curls (Tip <-> MCP)
    # If tip is close to MCP, finger is curled.
    features.append(get_dist(index_tip, index_mcp) / hand_scale)
    features.append(get_dist(middle_tip, middle_mcp) / hand_scale)
    features.append(get_dist(ring_tip, ring_mcp) / hand_scale)
    features.append(get_dist(pinky_tip, pinky_mcp) / hand_scale)

    # 3. Fingertip to Wrist distances (General hand shape)
    features.append(get_dist(thumb_tip, wrist) / hand_scale)
    features.append(get_dist(index_tip, wrist) / hand_scale)
    features.append(get_dist(middle_tip, wrist) / hand_scale)
    features.append(get_dist(ring_tip, wrist) / hand_scale)
    features.append(get_dist(pinky_tip, wrist) / hand_scale)

    # 4. Palm Spread (Index MCP <-> Pinky MCP)
    features.append(get_dist(index_mcp, pinky_mcp) / hand_scale)

    # 5. Thumb Reach (Thumb Tip <-> Pinky Tip) - good for fist vs open
    features.append(get_dist(thumb_tip, pinky_tip) / hand_scale)
    
    return np.array(features, dtype=np.float32)
