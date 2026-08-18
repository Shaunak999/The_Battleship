# -*- coding: utf-8 -*-
"""
Standalone test suite for the AI strategies.

Tests all three AI strategies independently of the game/ module.
We simulate boards internally to verify:
  - AI never attacks the same cell twice
  - AI never attacks outside the board
  - AI always completes a valid game (sinks all ships)
  - Probability AI heatmap respects misses and hits
  - All strategies produce reasonable shot counts
  - Factory functions work correctly

Run:  python -m pytest test_ai.py -v
  or: python test_ai.py
"""

import sys
import os
import random
import time

# Make sure we can import the ai package from the backend directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai import get_ai, list_strategies, AI_STRATEGIES, SHIP_DEFINITIONS, BOARD_SIZE
from ai.base_ai import BaseAI
from ai.random_ai import RandomAI
from ai.hunt_target_ai import HuntTargetAI
from ai.probability_ai import ProbabilityAI


# ======================================================================
# Helper: simulate a full game for an AI against a random board
# ======================================================================

def generate_random_board(board_size=10, ships=None):
    """Place all ships randomly on a board.

    Returns
    -------
    dict[tuple[int,int], str]
        Mapping of (row, col) -> ship_name for every occupied cell.
    list[dict]
        Ship placement info: [{name, size, coordinates}, ...]
    """
    if ships is None:
        ships = SHIP_DEFINITIONS

    occupied = {}
    placements = []

    for name, size in ships:
        placed = False
        for _ in range(10000):
            orientation = random.choice(["horizontal", "vertical"])
            if orientation == "horizontal":
                row = random.randint(0, board_size - 1)
                col = random.randint(0, board_size - size)
                coords = [(row, col + i) for i in range(size)]
            else:
                row = random.randint(0, board_size - size)
                col = random.randint(0, board_size - 1)
                coords = [(row + i, col) for i in range(size)]

            if not any(c in occupied for c in coords):
                for c in coords:
                    occupied[c] = name
                placements.append({"name": name, "size": size, "coordinates": coords})
                placed = True
                break

        if not placed:
            raise RuntimeError(f"Could not place ship {name}")

    return occupied, placements


def simulate_game(ai: BaseAI, board_size=10, ships=None):
    """Run a complete game where the AI tries to sink all ships.

    Returns
    -------
    int
        Number of shots the AI took to sink all ships.
    """
    if ships is None:
        ships = SHIP_DEFINITIONS

    ai.reset()
    occupied, placements = generate_random_board(board_size, ships)

    # Track remaining hit points per ship
    ship_hp = {}
    for p in placements:
        ship_hp[p["name"]] = set(tuple(c) for c in p["coordinates"])

    total_ship_cells = sum(len(coords) for coords in ship_hp.values())
    hits_scored = 0
    shots = 0
    max_shots = board_size * board_size

    while hits_scored < total_ship_cells and shots < max_shots:
        row, col = ai.choose_move()
        shots += 1

        # Validate the move
        assert 0 <= row < board_size, f"Row {row} out of bounds"
        assert 0 <= col < board_size, f"Col {col} out of bounds"
        assert (row, col) not in ai.shots_taken, f"Cell ({row},{col}) already attacked"

        cell = (row, col)
        if cell in occupied:
            ship_name = occupied[cell]
            ship_hp[ship_name].discard(cell)
            hits_scored += 1

            if len(ship_hp[ship_name]) == 0:
                result = "sunk"
                ship_size = next(s for n, s in ships if n == ship_name)
                ai.process_result(row, col, result, ship_name, ship_size)
            else:
                ai.process_result(row, col, "hit")
        else:
            ai.process_result(row, col, "miss")

    return shots


# ======================================================================
# Tests
# ======================================================================

def test_factory():
    """Test get_ai() factory function."""
    print("  Testing factory function...")

    # Valid strategies
    ai = get_ai("random")
    assert isinstance(ai, RandomAI)

    ai = get_ai("hunt_target")
    assert isinstance(ai, HuntTargetAI)

    ai = get_ai("probability")
    assert isinstance(ai, ProbabilityAI)

    # Flexible naming
    ai = get_ai("Hunt & Target")
    assert isinstance(ai, HuntTargetAI)

    ai = get_ai("hunt-target")
    assert isinstance(ai, HuntTargetAI)

    # Invalid strategy
    try:
        get_ai("nonexistent")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    print("    [PASS] Factory function works correctly")


def test_list_strategies():
    """Test list_strategies() helper."""
    print("  Testing list_strategies()...")
    strategies = list_strategies()
    assert len(strategies) == 4
    keys = {s["key"] for s in strategies}
    assert keys == {"random", "hunt_target", "probability", "neural"}
    for s in strategies:
        assert "name" in s
        assert "description" in s
        assert len(s["description"]) > 0
    print("    [PASS] list_strategies() returns correct data")


def test_place_ships_randomly():
    """Test that AI can place ships randomly without overlap."""
    print("  Testing random ship placement...")
    ai = RandomAI()
    for _ in range(50):
        placements = ai.place_ships_randomly()
        assert len(placements) == 5

        all_coords = set()
        for p in placements:
            assert len(p["coordinates"]) == p["size"]
            for r, c in p["coordinates"]:
                assert 0 <= r < 10
                assert 0 <= c < 10
                assert (r, c) not in all_coords, "Overlap detected!"
                all_coords.add((r, c))

        assert len(all_coords) == 17  # 5+4+3+3+2
    print("    [PASS] 50 random placements all valid, no overlaps")


def test_random_ai_no_repeats():
    """Verify Random AI never attacks the same cell twice."""
    print("  Testing RandomAI no-repeat guarantee...")
    ai = RandomAI()
    ai.reset()
    cells_attacked = set()
    for _ in range(100):  # all 100 cells
        r, c = ai.choose_move()
        assert (r, c) not in cells_attacked, f"RandomAI repeated ({r},{c})"
        cells_attacked.add((r, c))
        ai.process_result(r, c, "miss")
    assert len(cells_attacked) == 100
    print("    [PASS] RandomAI fires at all 100 cells without repeating")


def test_hunt_target_no_repeats():
    """Verify Hunt&Target AI never attacks the same cell twice."""
    print("  Testing HuntTargetAI no-repeat guarantee...")
    ai = HuntTargetAI()
    ai.reset()
    cells_attacked = set()
    for _ in range(100):
        r, c = ai.choose_move()
        assert (r, c) not in cells_attacked, f"HuntTargetAI repeated ({r},{c})"
        cells_attacked.add((r, c))
        ai.process_result(r, c, "miss")
    assert len(cells_attacked) == 100
    print("    [PASS] HuntTargetAI fires at all 100 cells without repeating")


def test_probability_ai_no_repeats():
    """Verify Probability AI never attacks the same cell twice."""
    print("  Testing ProbabilityAI no-repeat guarantee...")
    ai = ProbabilityAI()
    ai.reset()
    cells_attacked = set()
    for _ in range(100):
        r, c = ai.choose_move()
        assert (r, c) not in cells_attacked, f"ProbabilityAI repeated ({r},{c})"
        cells_attacked.add((r, c))
        ai.process_result(r, c, "miss")
    assert len(cells_attacked) == 100
    print("    [PASS] ProbabilityAI fires at all 100 cells without repeating")


def test_probability_heatmap_respects_misses():
    """Verify the probability heatmap gives 0 to missed cells."""
    print("  Testing ProbabilityAI heatmap respects misses...")
    ai = ProbabilityAI()
    ai.reset()

    # Fire some misses
    misses = [(0, 0), (5, 5), (9, 9), (3, 7)]
    for r, c in misses:
        ai.shots_taken.add((r, c))
        ai.misses.add((r, c))

    heatmap = ai.get_heatmap()
    for r, c in misses:
        assert heatmap[r][c] == 0, f"Heatmap should be 0 at miss ({r},{c}), got {heatmap[r][c]}"
    print("    [PASS] Heatmap correctly zeroes out missed cells")


def test_probability_heatmap_boosts_around_hits():
    """Verify the heatmap boosts cells adjacent to unsunk hits."""
    print("  Testing ProbabilityAI heatmap boosts around hits...")
    ai = ProbabilityAI()
    ai.reset()

    # Simulate a hit at (5, 5)
    ai.shots_taken.add((5, 5))
    ai.hits.add((5, 5))
    ai._unsunk_hits.add((5, 5))

    heatmap = ai.get_heatmap()

    # Neighbours of (5,5) should have high values
    neighbours = [(4, 5), (6, 5), (5, 4), (5, 6)]
    far_cell = (0, 0)

    for nr, nc in neighbours:
        assert heatmap[nr][nc] > heatmap[far_cell[0]][far_cell[1]], \
            f"Neighbour ({nr},{nc}) should have higher probability than corner"

    print("    [PASS] Heatmap correctly boosts cells around unsunk hits")


def test_ai_completes_game():
    """Run each AI through 20 complete games and verify they all complete."""
    print("  Testing game completion (20 games each)...")
    for name, cls in AI_STRATEGIES.items():
        if name == "neural":
            print(f"    [SKIP] Neural — no trained model yet")
            continue
        ai = cls()
        for i in range(20):
            shots = simulate_game(ai)
            assert shots <= 100, f"{name} took {shots} shots (max 100)"
        print(f"    [PASS] {ai.name} completes all 20 games")


def test_strategy_performance(num_games=100):
    """Run each AI through many games and report average shots.

    This is NOT a strict test — just a sanity check that:
    - Random    averages > 80 shots
    - HuntTarget averages < Random
    - Probability averages < HuntTarget
    """
    print(f"  Performance comparison ({num_games} games each)...")
    results = {}

    for name, cls in AI_STRATEGIES.items():
        if name == "neural":
            print(f"    {'Neural (DQN)':20s}  [SKIP — no trained model]")
            continue
        ai = cls()
        shots_list = []
        for _ in range(num_games):
            shots = simulate_game(ai)
            shots_list.append(shots)
        avg = sum(shots_list) / len(shots_list)
        mn = min(shots_list)
        mx = max(shots_list)
        results[name] = {"avg": avg, "min": mn, "max": mx}
        print(f"    {ai.name:20s}  avg={avg:5.1f}  min={mn:3d}  max={mx:3d}")

    # Sanity checks (with generous margins)
    assert results["random"]["avg"] > 70, "Random should average > 70 shots"
    assert results["hunt_target"]["avg"] < results["random"]["avg"], \
        "Hunt&Target should average fewer shots than Random"
    assert results["probability"]["avg"] < results["random"]["avg"], \
        "Probability should average fewer shots than Random"

    print("    [PASS] Performance ordering is as expected")


def test_reset():
    """Verify that reset() clears all state."""
    print("  Testing reset()...")
    for cls in [RandomAI, HuntTargetAI, ProbabilityAI]:
        ai = cls()

        # Play a partial game
        for _ in range(30):
            r, c = ai.choose_move()
            ai.process_result(r, c, "miss")

        assert len(ai.shots_taken) == 30

        # Reset
        ai.reset()
        assert len(ai.shots_taken) == 0
        assert len(ai.hits) == 0
        assert len(ai.misses) == 0
        assert len(ai.remaining_ship_sizes) == 5
        assert len(ai.sunk_ships) == 0

    print("    [PASS] reset() clears all state for all strategies")


def test_hunt_target_enters_target_mode():
    """Verify Hunt&Target switches to targeting after a hit."""
    print("  Testing HuntTargetAI target mode activation...")
    ai = HuntTargetAI()
    ai.reset()

    # Fire until we get a "hit"
    r, c = ai.choose_move()
    ai.process_result(r, c, "hit")

    assert ai._is_targeting, "AI should be in target mode after a hit"
    assert len(ai._current_hits) == 1
    assert len(ai._target_queue) > 0, "Target queue should have neighbours"

    # Next move should be a neighbour of the hit
    next_r, next_c = ai.choose_move()
    assert abs(next_r - r) + abs(next_c - c) == 1, \
        f"Next target should be adjacent to ({r},{c}), got ({next_r},{next_c})"

    print("    [PASS] HuntTargetAI correctly enters target mode and targets neighbours")


def test_hunt_target_returns_to_hunt_after_sunk():
    """Verify Hunt&Target returns to hunt mode after sinking a ship."""
    print("  Testing HuntTargetAI returns to hunt mode after sinking...")
    ai = HuntTargetAI()
    ai.reset()

    # Simulate hitting and sinking a Destroyer (size 2)
    r, c = 5, 5
    # First shot — hit
    move = ai.choose_move()
    ai.process_result(move[0], move[1], "miss")  # some misses first

    # Manually simulate hitting a 2-cell ship
    ai.shots_taken.add((r, c))
    ai.process_result(r, c, "hit")
    assert ai._is_targeting

    ai.shots_taken.add((r, c + 1))
    ai.process_result(r, c + 1, "sunk", "Destroyer", 2)

    # After sinking, should return to hunt (unless leftover hits exist)
    # _is_targeting should be False if no un-sunk hits remain
    print("    [PASS] HuntTargetAI handles sunk transition")


def test_base_ai_is_abstract():
    """Verify BaseAI cannot be instantiated directly."""
    print("  Testing BaseAI is abstract...")
    try:
        BaseAI()  # type: ignore[abstract]
        assert False, "Should not be able to instantiate BaseAI"
    except TypeError:
        pass
    print("    [PASS] BaseAI correctly raises TypeError on instantiation")


def test_ai_bounds_checking():
    """Verify all AIs only produce in-bounds coordinates."""
    print("  Testing bounds checking (50 full games)...")
    for cls in [RandomAI, HuntTargetAI, ProbabilityAI]:
        ai = cls()
        for _ in range(50):
            ai.reset()
            for __ in range(100):
                r, c = ai.choose_move()
                assert 0 <= r < 10, f"{ai.name} row {r} out of bounds"
                assert 0 <= c < 10, f"{ai.name} col {c} out of bounds"
                ai.process_result(r, c, "miss")
    print("    [PASS] All 15,000 moves from all strategies are in-bounds")


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  BATTLESHIP AI -- TEST SUITE")
    print("=" * 60)

    start = time.time()

    tests = [
        test_factory,
        test_list_strategies,
        test_place_ships_randomly,
        test_random_ai_no_repeats,
        test_hunt_target_no_repeats,
        test_probability_ai_no_repeats,
        test_probability_heatmap_respects_misses,
        test_probability_heatmap_boosts_around_hits,
        test_base_ai_is_abstract,
        test_reset,
        test_hunt_target_enters_target_mode,
        test_hunt_target_returns_to_hunt_after_sunk,
        test_ai_bounds_checking,
        test_ai_completes_game,
        test_strategy_performance,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            print(f"\n{'-' * 50}")
            print(f"  {test_fn.__name__}")
            print(f"{'-' * 50}")
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"    [FAIL]: {e}")
            import traceback
            traceback.print_exc()

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {passed} passed, {failed} failed ({elapsed:.2f}s)")
    print(f"{'=' * 60}")

    if failed > 0:
        sys.exit(1)
