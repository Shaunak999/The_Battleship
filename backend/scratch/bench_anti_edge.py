import random
import numpy as np
import torch
from ai.neural_ai import NeuralAgent, load_dqn_qnetwork, BOARD_SIZE, axis_extension_cells
from ai.neural.battleship_env import _place_ships_randomly, SHIP_DEFINITIONS

def random_edge_board():
    board = {}
    ships = [('Carrier', 5), ('Battleship', 4), ('Cruiser', 3), ('Submarine', 3), ('Destroyer', 2)]
    for name, size in ships:
        placed = False
        for _ in range(1000):
            orient = random.choice(['h', 'v'])
            if orient == 'h':
                r = random.choice([0, 1, 8, 9])
                c = random.randint(0, 10 - size)
                coords = [(r, c+i) for i in range(size)]
            else:
                r = random.randint(0, 10 - size)
                c = random.choice([0, 1, 8, 9])
                coords = [(r+i, c) for i in range(size)]
            if not any(co in board for co in coords):
                for co in coords: board[co] = (name, size)
                placed = True
                break
        if not placed: return random_edge_board()
    return board

class AdaptiveAntiCampAgent(NeuralAgent):
    def __init__(self, border_boost=0.5, miss_threshold=2, miss_weight=1.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.border_boost = border_boost
        self.miss_threshold = miss_threshold
        self.miss_weight = miss_weight

    def choose_move(self):
        if self._q_net is None:
            self._q_net = load_dqn_qnetwork(self._model_path, self.board_size)
            if self._q_net is None and not self._fallback_active:
                self._fallback_active = True
                if self._phase_random:
                    self._parity_shift = random.randint(0, 1)
                self._observation = self._build_obs()

        mask = np.ones(self.board_size * self.board_size, dtype=bool)
        for r, c in self.shots_taken:
            mask[r * self.board_size + c] = False

        if self._q_net is not None:
            in_c = 6
            obs_slice = self._observation[:in_c]
            obs_tensor = torch.from_numpy(obs_slice).unsqueeze(0)
            device = next(self._q_net.parameters()).device
            obs_tensor = obs_tensor.to(device)

            with torch.no_grad():
                q_values = self._q_net(obs_tensor).cpu().numpy().flatten()

            q_values[~mask] = -np.inf

            # Target mode: strict collinear extension
            if self._unsunk_hits:
                exts = axis_extension_cells(self._unsunk_hits, self.shots_taken, self.board_size)
                if exts:
                    ext_actions = [r * self.board_size + c for r, c in exts]
                    best_ext = max(ext_actions, key=lambda a: q_values[a])
                    return divmod(int(best_ext), self.board_size)
                action = int(np.argmax(q_values))
                return divmod(action, self.board_size)

            # Hunt mode
            heatmap_flat = self._observation[4].flatten()
            parity_flat = self._observation[5].flatten()
            dead_cells = (heatmap_flat == 0.0)
            q_values[dead_cells] = -np.inf

            # 1. Subtle baseline border parity compensation
            if self.border_boost > 0:
                for r in range(self.board_size):
                    for c in range(self.board_size):
                        idx = r * self.board_size + c
                        if parity_flat[idx] > 0.5:
                            if r in (0, 9) or c in (0, 9):
                                q_values[idx] += self.border_boost
                            elif r in (1, 8) or c in (1, 8):
                                q_values[idx] += self.border_boost * 0.5

            # 2. Adaptive Anti-Edge-Camping Reactive Shift:
            # Count misses in the central 6x6 core. When >= miss_threshold misses occur with 0 hits,
            # recognize that opponent is camping perimeter and aggressively boost border cells.
            center_misses = sum(1 for r, c in self.misses if 2 <= r <= 7 and 2 <= c <= 7)
            if center_misses >= self.miss_threshold and not self._unsunk_hits:
                excess_misses = center_misses - self.miss_threshold + 1
                for r in range(self.board_size):
                    for c in range(self.board_size):
                        idx = r * self.board_size + c
                        if parity_flat[idx] > 0.5:
                            if r in (0, 9) or c in (0, 9):
                                q_values[idx] += excess_misses * self.miss_weight
                            elif r in (1, 8) or c in (1, 8):
                                q_values[idx] += excess_misses * self.miss_weight * 0.7

            # 3. Dynamic Parity Scaling
            min_size = min(self.remaining_ship_sizes) if self.remaining_ship_sizes else 2
            if min_size >= 3:
                q_values += parity_flat * 5.0

            if np.all(np.isneginf(q_values)):
                q_values[~mask] = -np.inf
                action = int(np.argmax(q_values))
                return divmod(action, self.board_size)

            max_q = float(np.max(q_values))
            top_candidates = np.flatnonzero(q_values >= max_q - 1.2)
            if len(top_candidates) > 1:
                sub_q = q_values[top_candidates]
                exp_q = np.exp((sub_q - max_q) / 0.8)
                probs = exp_q / np.sum(exp_q)
                action = int(np.random.choice(top_candidates, p=probs))
            else:
                action = int(top_candidates[0])

            return divmod(action, self.board_size)

        return super().choose_move()

def evaluate(agent_factory, n=150):
    for btype in ["standard", "edge"]:
        shots = []
        for _ in range(n):
            agent = agent_factory()
            agent.reset()
            if btype == "standard":
                board_dict = _place_ships_randomly(10, SHIP_DEFINITIONS, touch_probability=0.0)
                board = {co: (name, next(s for n, s in SHIP_DEFINITIONS if n == name)) for co, (name, _) in board_dict.items()}
            else:
                board = random_edge_board()
            sh = 0
            while len(board) > 0 and sh < 100:
                r, c = agent.choose_move()
                sh += 1
                if (r, c) in board:
                    name, size = board.pop((r, c))
                    alive = any(s == name for s, sz in board.values())
                    if not alive:
                        agent.process_result(r, c, "sunk", name, size)
                    else:
                        agent.process_result(r, c, "hit")
                else:
                    agent.process_result(r, c, "miss")
            shots.append(sh)
        print(f"  {btype:<9s}: Avg {np.mean(shots):.1f}, Best {np.min(shots)}, Med {np.median(shots):.1f}, Worst {np.max(shots)}")

for (boost, thresh, weight) in [(0.3, 2, 1.5), (0.4, 3, 1.8), (0.5, 2, 1.2)]:
    print(f"\nTesting Config (border_boost={boost}, miss_threshold={thresh}, miss_weight={weight}):")
    evaluate(lambda: AdaptiveAntiCampAgent(border_boost=boost, miss_threshold=thresh, miss_weight=weight), 150)
