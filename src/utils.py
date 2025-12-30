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
        """
        min_cutoff: Min cutoff frequency in Hz (lower = more smoothing for slow mvmt)
        beta: Speed coefficient (higher = less lag for fast mvmt)
        d_cutoff: Cutoff for derivative
        """
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
        
        # Prevent division by zero or negative time
        if t_e <= 0.0:
            return self.x_prev

        # Calculate derivative (velocity)
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)
        
        # Calculate dynamic cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        
        # Filter signal
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)
        
        # Update state
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        
        return x_hat

class PointFilter:
    """Wrapper for 2D OneEuroFilter"""
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
        
        nx = self.f_x(t, x)
        ny = self.f_y(t, y)
        return (nx, ny)

# =============================================================================
# 3. Viterbi Decoder (State Smoothing)
# =============================================================================

class ViterbiDecoder:
    def __init__(self, labels, transition_prob=0.97, window_size=30):
        self.labels = labels  # list of state names e.g. ['DRAW', 'HOVER', 'ERASE']
        self.n_states = len(labels)
        self.map_idx = {name: i for i, name in enumerate(labels)}
        self.window_size = window_size
        
        # Transition Matrix (Log Domain)
        # Self loop = transition_prob
        # Others share remaining prob
        self.trans_log = np.zeros((self.n_states, self.n_states))
        
        for i in range(self.n_states):
            for j in range(self.n_states):
                if i == j:
                    p = transition_prob
                else:
                    # Distribute remainder
                    p = (1.0 - transition_prob) / (self.n_states - 1)
                
                # Custom overrides (Optional per spec)
                # DRAW<->HOVER ~0.02, ERASE rare
                # For simplicity we stick to uniform remaining, or user-tuned:
                if labels[i] == 'ERASE' and i != j:
                    # Exit erase is a bit easier? Or harder?
                    pass 
                
                self.trans_log[i, j] = np.log(p + 1e-9)
        
        # Buffer of observations (log probabilities)
        self.obs_buffer = deque(maxlen=window_size)

    def update(self, probs):
        """
        probs: dict {label: probability} or list matching self.labels order
        """
        if isinstance(probs, dict):
            # Sort by label index
            p_vec = [probs.get(l, 0.0) for l in self.labels]
        else:
            p_vec = probs
            
        # Convert to log domain
        p_vec = np.array(p_vec) + 1e-9
        log_p = np.log(p_vec)
        self.obs_buffer.append(log_p)

    def decode(self):
        """
        Runs Viterbi on the current window.
        Returns the likely state at the *end* of the window.
        """
        if not self.obs_buffer:
            return None
        
        T = len(self.obs_buffer)
        n = self.n_states
        
        # dpi[t][s] = max prob of path ending at state s at time t
        dp = np.zeros((T, n))
        path = np.zeros((T, n), dtype=int)
        
        # Initialize
        dp[0, :] = self.obs_buffer[0] # Assume uniform prior or use emission directly
        
        # Recurse
        for t in range(1, T):
            for s in range(n):
                # max over prev states
                # dp[t, s] = max(dp[t-1, prev] + trans[prev, s]) + emission[t, s]
                
                # Vectorized search for max_prev
                prev_scores = dp[t-1, :] + self.trans_log[:, s]
                best_prev = np.argmax(prev_scores)
                dp[t, s] = prev_scores[best_prev] + self.obs_buffer[t][s]
                path[t, s] = best_prev
                
        # Backtrack (we only need the last state usually, which is just argmax(dp[T-1]))
        # But if we were outputting the full sequence we'd backtrack.
        # The user wants "decode most likely sequence... and output the last state"
        # The last state of the most likely sequence IS argmax(dp[T-1]).
        
        last_state_idx = np.argmax(dp[T-1, :])
        return self.labels[last_state_idx]

# =============================================================================
# 4. State Hysteresis (Debouncing)
# =============================================================================

class StateMachine:
    def __init__(self, enter_draw=0.85, exit_draw=0.55, erase_hold=0.5):
        self.state = "HOVER"
        self.enter_draw = enter_draw
        self.exit_draw = exit_draw
        self.erase_hold = erase_hold
        
        self.history_probs = deque(maxlen=10) # Short window for threshold checks
        self.last_strong_erase = 0
        self.erase_start = None

    def update(self, viterbi_state, probs_dict):
        """
        Applies hysteresis rules on top of Viterbi state.
        probs_dict: {label: prob}
        """
        p_draw = probs_dict.get('DRAW', 0.0)
        p_erase = probs_dict.get('ERASE', 0.0)
        
        self.history_probs.append(probs_dict)
        
        # Average probabilities over short window
        avg_draw = sum(d.get('DRAW', 0.0) for d in self.history_probs) / len(self.history_probs)
        
        curr_time = time.time()
        
        # Logic
        if self.state == "HOVER":
            if viterbi_state == "DRAW" and avg_draw > self.enter_draw:
                self.state = "DRAW"
            elif viterbi_state == "ERASE":
                # Check hold time for erase
                if self.erase_start is None:
                    self.erase_start = curr_time
                elif (curr_time - self.erase_start) > self.erase_hold:
                    self.state = "ERASE"
            else:
                self.erase_start = None # Reset if we flicker away from Erase
                
        elif self.state == "DRAW":
            # Sticky exit
            if viterbi_state != "DRAW" and avg_draw < self.exit_draw:
                self.state = "HOVER"
            # Hard switch to erase? Usually unlikely from draw directly, pass thru hover
            
        elif self.state == "ERASE":
            # Exit erase easily once gesture stops
            if viterbi_state != "ERASE":
                self.state = "HOVER"
                self.erase_start = None

        return self.state
