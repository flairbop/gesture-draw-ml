import time
import math
import numpy as np
from collections import deque, Counter

# =============================================================================
# 1. FPS Counter
# =============================================================================

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

# =============================================================================
# 2. One Euro Filter (Cursor Stabilization)
# =============================================================================

class OneEuroFilter:
    def __init__(self, t0, x0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = float(x0)
        self.dx_prev = 0.0
        self.t_prev = float(t0)

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff
        return 1.0 / (1.0 + (1.0 / (r * t_e))) if t_e > 0 else 1.0

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1.0 - a) * x_prev

    def __call__(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 0.0: return self.x_prev
        
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)
        
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)
        
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

class PointFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0):
        self.f_x = None
        self.f_y = None
        self.min_cutoff = min_cutoff
        self.beta = beta

    def update(self, point):
        t = time.time()
        x, y = point
        if self.f_x is None:
            self.f_x = OneEuroFilter(t, x, min_cutoff=self.min_cutoff, beta=self.beta)
            self.f_y = OneEuroFilter(t, y, min_cutoff=self.min_cutoff, beta=self.beta)
            return point
        return (self.f_x(t, x), self.f_y(t, y))

# =============================================================================
# 3. Viterbi Decoder (State Smoothing)
# =============================================================================

class ViterbiDecoder:
    def __init__(self, labels, transition_prob=0.97, window_size=30):
        self.labels = labels
        self.n_states = len(labels)
        self.window_size = window_size
        self.trans_log = np.zeros((self.n_states, self.n_states))
        
        for i in range(self.n_states):
            for j in range(self.n_states):
                p = transition_prob if i == j else (1.0 - transition_prob) / (self.n_states - 1)
                self.trans_log[i, j] = np.log(p + 1e-9)
        self.obs_buffer = deque(maxlen=window_size)

    def update(self, probs):
        if isinstance(probs, dict):
            p_vec = [probs.get(l, 0.0) for l in self.labels]
        else:
            p_vec = probs
        self.obs_buffer.append(np.log(np.array(p_vec) + 1e-9))

    def decode(self):
        if not self.obs_buffer: return None
        T = len(self.obs_buffer)
        n = self.n_states
        dp = np.zeros((T, n))
        dp[0, :] = self.obs_buffer[0]
        
        for t in range(1, T):
            for s in range(n):
                prev_scores = dp[t-1, :] + self.trans_log[:, s]
                dp[t, s] = np.max(prev_scores) + self.obs_buffer[t][s]
                
        return self.labels[np.argmax(dp[T-1, :])]

# =============================================================================
# 4. Robust State Machine (Hysteresis & Debounce)
# =============================================================================

class StateMachine:
    def __init__(self, 
                 enter_draw_thresh=0.85, 
                 exit_draw_thresh=0.55, 
                 enter_erase_thresh=0.90,
                 min_draw_dwell=0.15):
        self.state = "HOVER"
        
        # Calibration
        self.enter_draw_thresh = enter_draw_thresh
        self.exit_draw_thresh = exit_draw_thresh
        self.enter_erase_thresh = enter_erase_thresh
        self.min_draw_dwell = min_draw_dwell
        
        # History buffers (frames)
        self.history_len = 15
        self.probs_history = deque(maxlen=self.history_len)
        
        # Dwell timing
        self.last_state_change = time.time()

    def _count_sustained(self, label, threshold, window=8):
        """Count how many frames in the last `window` exceeded `threshold` for `label`."""
        if len(self.probs_history) < 1: return 0
        
        # Iterate backwards
        count = 0
        limit = min(window, len(self.probs_history))
        for i in range(1, limit + 1):
            p = self.probs_history[-i].get(label, 0.0)
            if p >= threshold:
                count += 1
        return count

    def update(self, curr_probs):
        """
        curr_probs: dict {label: probability}
        Returns: current stable state
        """
        self.probs_history.append(curr_probs)
        now = time.time()
        time_gathering = (now - self.last_state_change)
        
        # Logic
        if self.state == "HOVER":
            # Rule: Enter DRAW if P(DRAW) >= 0.85 for 4 of last 6 frames
            draw_high_count = self._count_sustained("DRAW", self.enter_draw_thresh, window=6)
            if draw_high_count >= 4:
                self.state = "DRAW"
                self.last_state_change = now
                return self.state
            
            # Rule: Enter ERASE if P(ERASE) >= 0.90 for 10 of last 15 frames
            erase_high_count = self._count_sustained("ERASE", self.enter_erase_thresh, window=15)
            if erase_high_count >= 10:
                self.state = "ERASE"
                self.last_state_change = now
                return self.state

        elif self.state == "DRAW":
            # Minimum dwell check
            if time_gathering < self.min_draw_dwell:
                return self.state

            # Rule: Exit DRAW if P(DRAW) <= 0.55 for 6 of last 10 frames
            # Equivalent: Check if P(DRAW) was LOW
            # We count frames where P(DRAW) <= exit_thresh
            low_draw_count = 0
            check_win = 10
            limit = min(check_win, len(self.probs_history))
            for i in range(1, limit + 1):
                if self.probs_history[-i].get("DRAW", 0.0) <= self.exit_draw_thresh:
                    low_draw_count += 1
            
            if low_draw_count >= 6:
                self.state = "HOVER"
                self.last_state_change = now
                
        elif self.state == "ERASE":
            # Exit ERASE if P(ERASE) drops. 
            # Let's say if P(ERASE) < 0.80 for 5 of last 8 frames
            low_erase_count = 0
            check_win = 8
            limit = min(check_win, len(self.probs_history))
            for i in range(1, limit + 1):
                if self.probs_history[-i].get("ERASE", 0.0) < 0.8:
                    low_erase_count += 1
            
            if low_erase_count >= 5:
                self.state = "HOVER"
                self.last_state_change = now

        return self.state
