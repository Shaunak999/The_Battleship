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
from collections import Counter
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

def _touches(
    coords: List[Tuple[int, int]],
    occupied: Dict[Tuple[int, int], str],
    board_size: int,
) -> bool:
    """True if any coordinate is adjacent (incl. diagonally) to an occupied cell."""
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < board_size and 0 <= nc < board_size and (nr, nc) in occupied:
                    return True
    return False


def generate_random_board(
    board_size: int = BOARD_SIZE,
    ships=None,
    touch_probability: float = 0.0,
) -> Tuple[Dict[Tuple[int, int], str], List[Dict]]:
    """Place ships randomly in one of the two real-world placement styles.

    AI defenders and the frontend's "Randomize" button always keep a 1-cell
    gap between ships; manual human placement only forbids overlap, so ships
    may touch. ``touch_probability`` is the chance that a given board uses the
    touching style. The default (0.0 = gapped) matches the training
    environment in :mod:`ai.neural.train`, so a model is always scored on the
    same distribution it was trained on. Pass 0.5 to mix both styles or 1.0
    for full touching boards.

    Returns (board_map, placements).
    """
    if ships is None:
        ships = SHIP_DEFINITIONS

    allow_touch = np.random.random() < touch_probability
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

            if not any(co in occupied for co in coords) and (
                allow_touch or not _touches(coords, occupied, board_size)
            ):
                for co in coords:
                    occupied[co] = name
                placements.append({"name": name, "size": size, "coordinates": coords})
                placed = True
                break
        if not placed:
            return generate_random_board(board_size, ships, touch_probability)

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
    fired_at: set = set()

    while hits < total_hp and shots < max_shots:
        row, col = attacker.choose_move()
        shots += 1
        cell = (row, col)

        # A strategy that repeats a cell makes no progress. Never count the
        # same cell twice, or `hits` can reach total_hp early and report a
        # win that never happened.
        if cell in fired_at:
            continue
        fired_at.add(cell)

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
    touch_probability: float = 0.0,
) -> Dict[str, float]:
    """Run *num_games* and return stats."""
    wins = 0
    total_shots = 0
    win_shots: List[int] = []

    for i in range(num_games):
        if (i + 1) % max(1, num_games // 10) == 0:
            print(f"\r    Progress: {i + 1}/{num_games}", end="", flush=True)
        
        board, placements = generate_random_board(touch_probability=touch_probability)
        won, shots = play_one_game(attacker, board, placements, max_shots)
        if won:
            wins += 1
            win_shots.append(shots)
        total_shots += shots

    print("\r" + " " * 30 + "\r", end="", flush=True)  # Clear progress line
    
    win_rate = wins / num_games * 100
    avg_shots = total_shots / num_games
    avg_win_shots = np.mean(win_shots) if win_shots else float("nan")
    min_shots = int(np.min(win_shots)) if win_shots else 0
    med_shots = float(np.median(win_shots)) if win_shots else float("nan")
    max_shots = int(np.max(win_shots)) if win_shots else 0
    mode_shots = Counter(win_shots).most_common(1)[0][0] if win_shots else 0

    return {
        "label": label,
        "win_rate": win_rate,
        "avg_shots": avg_shots,
        "avg_win_shots": avg_win_shots,
        "min_shots": min_shots,
        "med_shots": med_shots,
        "mode_shots": mode_shots,
        "max_shots": max_shots,
        "wins": wins,
        "total": num_games,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate NeuralAgent vs baselines")
    parser.add_argument("--games", type=int, default=1000, help="Number of games")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained model (without .zip)")
    parser.add_argument("--touch-prob", type=float, default=0.0,
                        help="Board placement style: 0.0 = gapped standard (default, matches training), "
                             "0.5 = mixed, 1.0 = ships may touch")
    args = parser.parse_args()

    # Import here so the script works even if torch isn't installed yet
    try:
        from ai.neural_ai import NeuralAgent
    except ImportError:
        from backend.ai.neural_ai import NeuralAgent

    print("=" * 80)
    print("  BATTLESHIP AI EVALUATION")
    print("=" * 80)

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
        print(f"\n  \u26a0  Could not load NeuralAgent: {e}")
        print("     Training the model first: python -m ai.neural.train\n")

    touch_label = "Pure Gapped (Standard)" if args.touch_prob == 0.0 else ("Dense Touching" if args.touch_prob == 1.0 else "50/50 Mixed")
    print(f"\n  Games per matchup : {args.games}")
    print(f"  Board style       : {touch_label} (touch_prob={args.touch_prob})")
    print(f"  Max shots per game: 100\n")

    results = []
    for agent, label in competitors:
        print(f"  Evaluating {label}...", end=" ", flush=True)
        start = time.time()
        stats = evaluate(agent, num_games=args.games, label=label, touch_probability=args.touch_prob)
        elapsed = time.time() - start
        print(f"done ({elapsed:.1f}s)")
        results.append(stats)

    # ── Print results table ────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print(f"  RESULTS ({args.games} games each, {touch_label})")
    print(f"{'=' * 80}")
    print(f"  {'Strategy':<16s}  {'Win%':>5s}  {'Best':>5s}  {'Mode':>6s}  {'Avg':>7s}  {'Worst':>6s}")
    print(f"  {'-'*16}  {'-'*5}  {'-'*5}  {'-'*6}  {'-'*7}  {'-'*6}")

    for r in results:
        print(
            f"  {r['label']:<16s}  {r['win_rate']:4.0f}%  "
            f"{r['min_shots']:5d}  {r['mode_shots']:6d}  "
            f"{r['avg_shots']:7.1f}  {r['max_shots']:6d}"
        )

    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
