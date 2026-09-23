import sys, os
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import random
import torch
from ai.neural_ai import NeuralAgent
from ai.neural.evaluate import play_one_game, generate_random_board

class VariedNeuralAgent(NeuralAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._parity_shift = random.randint(0, 1)

    def reset(self):
        super().reset()
        self._parity_shift = random.randint(0, 1)
        self._observation = self._build_obs()

    def _build_obs(self):
        # Override to ensure self._parity_shift is applied to Channel 5
        board = self.board_size
        obs = super()._build_obs()
        if not self._unsunk_hits:
            min_size = min(self.remaining_ship_sizes) if self.remaining_ship_sizes else 2
            channel5 = np.zeros((board, board), dtype=np.float32)
            blocking = self.misses | (self.hits - self._unsunk_hits)
            for r in range(board):
                for c in range(board):
                    if (r, c) in self.shots_taken:
                        continue
                    if (r + c + self._parity_shift) % min_size != 0:
                        continue
                    # Dead space check
                    max_len = 0
                    for dr, dc in [(0, 1), (1, 0)]:
                        length = 1
                        i = 1
                        while 0 <= r + dr*i < board and 0 <= c + dc*i < board and (r + dr*i, c + dc*i) not in blocking:
                            length += 1
                            i += 1
                        i = 1
                        while 0 <= r - dr*i < board and 0 <= c - dc*i < board and (r - dr*i, c - dc*i) not in blocking:
                            length += 1
                            i += 1
                        max_len = max(max_len, length)
                    if max_len >= min_size:
                        channel5[r, c] = 1.0
            obs[5] = channel5
        return obs

def run_bench():
    print("Running 200 games comparison...")
    for touch_prob in [0.0, 0.5]:
        s_std = []
        s_var = []
        for _ in range(200):
            b, p = generate_random_board(touch_probability=touch_prob)
            na_std = NeuralAgent()
            na_var = VariedNeuralAgent()
            _, s1 = play_one_game(na_std, b, p)
            _, s2 = play_one_game(na_var, b, p)
            s_std.append(s1)
            s_var.append(s2)
        mode = "Gapped (touch_prob=0.0)" if touch_prob == 0.0 else "Mixed (touch_prob=0.5)"
        print(f"\n--- {mode} ---")
        print(f"  Fixed Diagonal Neural:  avg = {np.mean(s_std):.2f}, med = {np.median(s_std):.1f}")
        print(f"  Varied De-predicted:   avg = {np.mean(s_var):.2f}, med = {np.median(s_var):.1f}")

if __name__ == "__main__":
    run_bench()
