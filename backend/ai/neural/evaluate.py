"""
Evaluate the trained NeuralAgent against every baseline strategy.

Runs 1000 games and records:
    - Win rate
    - Average shots to win (or 100 if lost)

Usage:
    cd backend
    python -m ai.neural.evaluate [--games N] [--model PATH]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

# Ensure backend/ and repo root are on the path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
repo_dir = os.path.abspath(os.path.join(backend_dir, ".."))
for p in [backend_dir, repo_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from ai.base_ai import BaseAI, SHIP_DEFINITIONS, BOARD_SIZE
    from ai.random_ai import RandomAI
    from ai.hunt_target_ai import HuntTargetAI
    from ai.probability_ai import ProbabilityAI
except ImportError:
    from backend.ai.base_ai import BaseAI, SHIP_DEFINITIONS, BOARD_SIZE
    from backend.ai.random_ai import RandomAI
    from backend.ai.hunt_target_ai import HuntTargetAI
    from backend.ai.probability_ai import ProbabilityAI


# ── Simulation helper ──────────────────────────────────────────────────────

def generate_random_board(
    board_size: int = BOARD_SIZE,
    ships=None,
) -> Tuple[Dict[Tuple[int, int], str], List[Dict]]:
    """Place ships randomly. Returns (board_map, placements)."""
    if ships is None:
        ships = SHIP_DEFINITIONS

    occupied: Dict[Tuple[int, int], str] = {}
    placements = []

    for name, size in ships:
        placed = False
        for _ in range(10000):
            orient = "h" if np.random.random() < 0.5 else "v"
            if orient == "h":
                r = np.random.randint(0, board_size)
                c = np.random.randint(0, board_size - size + 1)
                coords = [(r, c + i) for i in range(size)]
            else:
                r = np.random.randint(0, board_size - size + 1)
                c = np.random.randint(0, board_size)
                coords = [(r + i, c) for i in range(size)]

            if not any(co in occupied for co in coords):
                for co in coords:
                    occupied[co] = name
                placements.append({"name": name, "size": size, "coordinates": coords})
                placed = True
                break
        if not placed:
            return generate_random_board(board_size, ships)

    return occupied, placements


def play_one_game(
    attacker: BaseAI,
    defender_board: Dict[Tuple[int, int], str],
    defender_placements: List[Dict],
    max_shots: int = 100,
) -> Tuple[bool, int]:
    """Play one game. Attacker tries to sink all ships.

    Returns (won, shots_taken).
    """
    attacker.reset()

    # Build ship HP tracking
    ship_hp: Dict[str, set] = {}
    for p in defender_placements:
        ship_hp[p["name"]] = set(tuple(c) for c in p["coordinates"])

    total_hp = sum(len(v) for v in ship_hp.values())
    hits = 0
    shots = 0

    while hits < total_hp and shots < max_shots:
        row, col = attacker.choose_move()
        shots += 1
        cell = (row, col)

        if cell in defender_board:
            ship_name = defender_board[cell]
            ship_hp[ship_name].discard(cell)
            hits += 1

            if len(ship_hp[ship_name]) == 0:
                ship_size = next(s for n, s in SHIP_DEFINITIONS if n == ship_name)
                attacker.process_result(row, col, "sunk", ship_name, ship_size)
            else:
                attacker.process_result(row, col, "hit")
        else:
            attacker.process_result(row, col, "miss")

    won = hits >= total_hp
    return won, shots


# ── Evaluation ─────────────────────────────────────────────────────────────

def evaluate(
    attacker: BaseAI,
    num_games: int = 1000,
    label: str = "",
    max_shots: int = 100,
) -> Dict[str, float]:
    """Run *num_games* and return stats."""
    wins = 0
    total_shots = 0
    win_shots: List[int] = []

    for i in range(num_games):
        if (i + 1) % max(1, num_games // 10) == 0:
            print(f"\r    Progress: {i + 1}/{num_games}", end="", flush=True)
        
        board, placements = generate_random_board()
        won, shots = play_one_game(attacker, board, placements, max_shots)
        if won:
            wins += 1
            win_shots.append(shots)
        total_shots += shots

    print("\r" + " " * 30 + "\r", end="", flush=True)  # Clear progress line
    
    win_rate = wins / num_games * 100
    avg_shots = total_shots / num_games
    avg_win_shots = np.mean(win_shots) if win_shots else float("nan")

    return {
        "label": label,
        "win_rate": win_rate,
        "avg_shots": avg_shots,
        "avg_win_shots": avg_win_shots,
        "wins": wins,
        "total": num_games,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate NeuralAgent vs baselines")
    parser.add_argument("--games", type=int, default=1000, help="Number of games")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained model (without .zip)")
    args = parser.parse_args()

    # Import here so the script works even if torch isn't installed yet
    try:
        from ai.neural_ai import NeuralAgent
    except ImportError:
        from backend.ai.neural_ai import NeuralAgent

    print("=" * 65)
    print("  BATTLESHIP AI EVALUATION")
    print("=" * 65)

    # Build list of competitors
    competitors = [
        (RandomAI(), "Random"),
        (HuntTargetAI(), "Hunt & Target"),
        (ProbabilityAI(), "Probability"),
    ]

    # Add NeuralAgent if model exists
    try:
        neural = NeuralAgent(model_path=args.model)
        neural.reset()  # triggers model load
        competitors.append((neural, "Neural"))
    except FileNotFoundError as e:
        print(f"\n  ⚠  Could not load NeuralAgent: {e}")
        print("     Training the model first: python -m ai.neural.train\n")

    print(f"\n  Games per matchup: {args.games}")
    print(f"  Max shots per game: 100\n")

    results = []
    for agent, label in competitors:
        print(f"  Evaluating {label}...", end=" ", flush=True)
        start = time.time()
        stats = evaluate(agent, num_games=args.games, label=label)
        elapsed = time.time() - start
        print(f"done ({elapsed:.1f}s)")
        results.append(stats)

    # ── Print results table ────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"  RESULTS ({args.games} games each)")
    print(f"{'=' * 65}")
    print(f"  {'Strategy':<20s}  {'Win%':>6s}  {'Avg Shots':>10s}  {'Avg Win Shots':>14s}")
    print(f"  {'-'*20}  {'-'*6}  {'-'*10}  {'-'*14}")

    for r in results:
        print(
            f"  {r['label']:<20s}  {r['win_rate']:5.1f}%  "
            f"{r['avg_shots']:10.1f}  {r['avg_win_shots']:14.1f}"
        )

    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
