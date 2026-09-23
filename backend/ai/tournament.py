"""
Round-robin AI-vs-AI tournament.

Every AI strategy plays a head-to-head Battleship match against every other
strategy. In each match the two sides defendindependently generated random boards (gapped placement, the style the Neural
agent is trained against) and take turns
firing at each other, exactly like the engine's "ai_vs_ai" mode. The first
side to sink the opponent's whole fleet wins. Starting player alternates
between games so the first-mover advantage cancels out.

Usage (from the backend/ directory):

    python -m ai.tournament                      # 500 games per pair
    python -m ai.tournament --games 200          # fewer games
    python -m ai.tournament --report my.html     # custom report file

Output
------
Writes an interactive HTML dashboard (--report, default ai_vs_ai_results.html)
with head-to-head heatmaps, an overall ranking, matchup breakdowns and the
per-game shot distributions. Requires plotly (see requirements.txt); skip
it with --no-report.

A won/lost breakdown with win percentages is also printed to the console.
"""

import argparse
import os
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

# Make sure backend/ is importable when run as a plain script too
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai.hunt_target_ai import HuntTargetAI
from ai.neural_ai import NeuralAgent
from ai.neural.evaluate import generate_random_board
from ai.probability_ai import ProbabilityAI
from ai.random_ai import RandomAI

# Ordered by strength, used for display and the round-robin pair list.
STRATEGIES: Dict[str, Callable[[], object]] = {
    "random": RandomAI,
    "hunt_target": HuntTargetAI,
    "probability": ProbabilityAI,
    "neural": NeuralAgent,
}

DISPLAY_NAMES = {
    "random": "Random",
    "hunt_target": "Hunt&Target",
    "probability": "Probability",
    "neural": "Neural",
}

MAX_SHOTS_PER_PLAYER = 100  # same cap as the rest of the codebase
MAX_TURNS = 2 * MAX_SHOTS_PER_PLAYER + 20  # hard stop so a game always ends

_SIZE_BY_NAME = {
    "Carrier": 5,
    "Battleship": 4,
    "Cruiser": 3,
    "Submarine": 3,
    "Destroyer": 2,
}


# ----------------------------------------------------------------------
# Head-to-head match
# ----------------------------------------------------------------------

def _fire(
    attacker,
    defender_board: Dict[Tuple[int, int], str],
    ship_hp: Dict[str, set],
    ship_cells_left: int,
) -> Tuple[str, int]:
    """One AI fires at the defender's board.

    Returns (outcome, ship_cells_left_after) where outcome is
    "miss" / "hit" / "sunk" / "win" (the last one sinks the final ship).
    """
    row, col = attacker.choose_move()
    cell = (row, col)

    if cell not in defender_board:
        attacker.process_result(row, col, "miss")
        return "miss", ship_cells_left

    name = defender_board[cell]
    ship_hp[name].discard(cell)
    ship_cells_left -= 1

    if ship_cells_left == 0:
        attacker.process_result(row, col, "sunk", name, _SIZE_BY_NAME[name])
        return "win", 0

    if not ship_hp[name]:
        attacker.process_result(row, col, "sunk", name, _SIZE_BY_NAME[name])
    else:
        attacker.process_result(row, col, "hit")
    return "hit", ship_cells_left


def play_match(
    make_a: Callable[[], object],
    make_b: Callable[[], object],
    a_starts: bool,
) -> Tuple[Optional[str], int, int]:
    """Play one head-to-head match.

    Player A defends ``board_a`` (attacked by B) and attacks ``board_b``.
    Returns (winner, shots_a, shots_b) where winner is "a", "b" or None (draw).
    """
    # Gapped boards (0.0): the placement style the Neural agent is trained on.
    board_a, place_a = generate_random_board(touch_probability=0.0)
    board_b, place_b = generate_random_board(touch_probability=0.0)

    ai_a = make_a()
    ai_b = make_b()

    hp_a = {p["name"]: set(tuple(c) for c in p["coordinates"]) for p in place_a}
    hp_b = {p["name"]: set(tuple(c) for c in p["coordinates"]) for p in place_b}
    cells_a = sum(len(v) for v in hp_a.values())  # 17
    cells_b = sum(len(v) for v in hp_b.values())

    shots_a = shots_b = 0

    for turn in range(MAX_TURNS):
        a_fires = (turn % 2 == 0) == a_starts

        if a_fires:
            shots_a += 1
            outcome, cells_b = _fire(ai_a, board_b, hp_b, cells_b)
            if outcome == "win":
                return "a", shots_a, shots_b
        else:
            shots_b += 1
            outcome, cells_a = _fire(ai_b, board_a, hp_a, cells_a)
            if outcome == "win":
                return "b", shots_a, shots_b

        if shots_a >= MAX_SHOTS_PER_PLAYER and shots_b >= MAX_SHOTS_PER_PLAYER:
            break

    return None, shots_a, shots_b  # draw (nobody sank the fleet in time)


# ----------------------------------------------------------------------
# Tournament
# ----------------------------------------------------------------------

def run_tournament(games_per_pair: int) -> Dict[str, Dict]:
    """Run every unordered pair of strategies.

    Returns results[a][b] = dict with keys a_wins, b_wins, draws,
    shots_a (list), shots_b (list), winner_seq (list of "a"/"b"/None,
    one entry per game, from a's perspective) for the a-vs-b session.
    """
    keys = list(STRATEGIES.keys())
    results: Dict[str, Dict[str, Dict]] = {
        a: {b: {"a_wins": 0, "b_wins": 0, "draws": 0, "shots_a": [], "shots_b": [], "winner_seq": []} for b in keys}
        for a in keys
    }

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            stats = results[a][b]
            t0 = time.time()

            for g in range(games_per_pair):
                # Alternate the starter so the first-mover advantage is balanced.
                winner, shots_a, shots_b = play_match(
                    STRATEGIES[a], STRATEGIES[b], a_starts=(g % 2 == 0)
                )
                if winner == "a":
                    stats["a_wins"] += 1
                elif winner == "b":
                    stats["b_wins"] += 1
                else:
                    stats["draws"] += 1
                stats["winner_seq"].append(winner)
                stats["shots_a"].append(shots_a)
                stats["shots_b"].append(shots_b)

            # Mirror for the reverse direction: b vs a is the complement.
            # "a"/"b"/None flip roles; a draw stays a draw (never counted as
            # a win for either side).
            mirrored = results[b][a]
            mirrored["a_wins"], mirrored["b_wins"] = stats["b_wins"], stats["a_wins"]
            mirrored["draws"] = stats["draws"]
            mirrored["shots_a"] = stats["shots_b"]
            mirrored["shots_b"] = stats["shots_a"]
            mirrored["winner_seq"] = [
                {"a": "b", "b": "a"}.get(w) for w in stats["winner_seq"]
            ]

            print(
                f"  {DISPLAY_NAMES[a]:<12s} vs {DISPLAY_NAMES[b]:<12s} "
                f"({games_per_pair} games, {time.time() - t0:.0f}s) "
                f"-> {DISPLAY_NAMES[a]} {stats['a_wins']} / "
                f"{DISPLAY_NAMES[b]} {stats['b_wins']} / draws {stats['draws']}",
                flush=True,
            )

    return results


# ----------------------------------------------------------------------
# Console report
# ----------------------------------------------------------------------

def print_records(results: Dict[str, Dict], games_per_pair: int) -> None:
    """Print how many games each AI won and lost, with win percentages.

    For every AI it shows the record against each opponent plus the overall
    record across all its games.
    """
    keys = list(STRATEGIES.keys())
    per_ai = []  # (total_wins, total_losses, name, lines, draw_txt, overall_pct)
    for a in keys:
        lines = []
        total_wins = total_losses = total_draws = 0
        for b in keys:
            if a == b:
                continue
            stats = results[a][b]
            wins, losses = stats["a_wins"], stats["b_wins"]
            draws = stats["draws"]
            total_wins += wins
            total_losses += losses
            total_draws += draws
            decided = wins + losses
            pct = 100.0 * wins / max(decided, 1)
            draw_txt = f", drawn {draws}" if draws else ""
            lines.append(
                f"    vs {DISPLAY_NAMES[b]:<12s}: "
                f"won {wins:>4}  lost {losses:>4}{draw_txt}  "
                f"({pct:5.1f}%)"
            )
        total = games_per_pair * (len(keys) - 1)
        decided = total - total_draws
        overall_pct = 100.0 * total_wins / max(decided, 1)
        draw_txt = f", drawn {total_draws}" if total_draws else ""
        per_ai.append((total_wins, total_losses, a, lines, draw_txt, overall_pct))

    print(f"\n  Won/lost record per AI ({games_per_pair} games per match-up)\n")
    for total_wins, total_losses, a, lines, draw_txt, overall_pct in sorted(
        per_ai, key=lambda x: -x[0]
    ):
        print(f"  {DISPLAY_NAMES[a]}:")
        for line in lines:
            print(line)
        print(
            f"    {'TOTAL':<12s}: won {total_wins:>4}  lost {total_losses:>4}"
            f"{draw_txt}  ({overall_pct:5.1f}%)"
        )
        print()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Round-robin AI-vs-AI tournament")
    parser.add_argument(
        "--games", type=int, default=500,
        help="Games per match-up (default: 500)",
    )
    parser.add_argument(
        "--report", type=str, default="ai_vs_ai_results.html",
        help="Interactive HTML report to write "
             "(default: ai_vs_ai_results.html)",
    )
    parser.add_argument(
        "--no-report", action="store_true",
        help="Skip the interactive HTML report",
    )
    parser.add_argument(
        "--no-open", action="store_true",
        help="Do not automatically open the HTML report in default browser",
    )
    args = parser.parse_args()

    print("=" * 62)
    print("  AI VS AI — ROUND ROBIN TOURNAMENT")
    print("=" * 62)
    print(f"  Strategies : {', '.join(DISPLAY_NAMES[k] for k in STRATEGIES)}")
    print(f"  Games/pair : {args.games}  (starting player alternates)")
    print(f"  Pairs      : {len(STRATEGIES) * (len(STRATEGIES) - 1) // 2}")
    print()

    t0 = time.time()
    results = run_tournament(args.games)
    print(f"\n  Tournament finished in {time.time() - t0:.0f}s")
    print_records(results, args.games)
    if not args.no_report:
        try:
            from ai.report import write_report
            write_report(results, args.games, args.report, auto_open=not args.no_open)
        except ImportError as exc:
            print(
                f"\n  [!] HTML report skipped: plotly is not installed ({exc}).\n"
                "      Add 'plotly' to requirements.txt and reinstall, "
                "or rerun with --no-report."
            )


if __name__ == "__main__":
    main()
