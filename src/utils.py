import time
from collections import deque, Counter

class FPS:
    def __init__(self, avg_len=30):
        self._prev_time = time.time()
        self._delays = deque(maxlen=avg_len)
    
    def update(self):
        curr_time = time.time()
        delay = curr_time - self._prev_time
        self._prev_time = curr_time
        self._delays.append(delay)
        
    def get(self):
        if not self._delays:
            return 0.0
        return 1.0 / (sum(self._delays) / len(self._delays))

class Smoother:
    """
    Smoothing for classification predictions.
    Uses majority vote over a window AND requires confidence threshold.
    """
    def __init__(self, window_size=10, threshold=0.6):
        self.window_size = window_size
        self.threshold = threshold
        self.history = deque(maxlen=window_size)
    
    def update(self, prediction, probability):
        """
        Args:
            prediction (str/int): The predicted class label.
            probability (float): The confidence of the prediction.
        """
        self.history.append((prediction, probability))
    
    def get_stable_prediction(self):
        if not self.history:
            return None, 0.0
            
        # Unpack
        preds = [h[0] for h in self.history]
        probs = [h[1] for h in self.history]
        
        # Count votes
        counts = Counter(preds)
        most_common, count = counts.most_common(1)[0]
        
        # Check stability (e.g., > 50% of window agrees)
        # and check average confidence for that specific class
        if count > (self.window_size * 0.5):
            # Calc avg prob for the winner
            avg_prob = sum(p for p, l in zip(probs, preds) if l == most_common) / count
            
            if avg_prob >= self.threshold:
                return most_common, avg_prob
                
        # Fallback or unstable
        return None, 0.0
