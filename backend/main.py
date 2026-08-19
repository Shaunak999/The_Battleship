import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

# Ensure backend directory is in sys.path for direct or module execution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from ai import AI_STRATEGY_NAMES, get_ai, list_strategies
    from ai.base_ai import BOARD_SIZE, SHIP_DEFINITIONS
    from game.game import Game
    from game.ship import Ship
except ImportError:
    from backend.ai import AI_STRATEGY_NAMES, get_ai, list_strategies
    from backend.ai.base_ai import BOARD_SIZE, SHIP_DEFINITIONS
    from backend.game.game import Game
    from backend.game.ship import Ship

# Precomputed once instead of linear-scanning SHIP_DEFINITIONS on every sunk event
SHIP_SIZES: Dict[str, int] = dict(SHIP_DEFINITIONS)


app = FastAPI(
    title="Battleship AI Backend",
    description="Local Battleship game engine with Human vs Human and Human vs AI.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

games: Dict[str, Dict[str, Any]] = {}


class GameCreateRequest(BaseModel):
    mode: str = "human_vs_human"
    player1_name: Optional[str] = "Player 1"
    player2_name: Optional[str] = "Player 2"
    ai_strategy: Optional[str] = None
    ai_strategy_2: Optional[str] = None  # for ai_vs_ai mode


class PlaceShipRequest(BaseModel):
    player_index: int
    ship_name: str
    coordinate: str
    orientation: str


class AttackRequest(BaseModel):
    player_index: int
    coordinate: str


def parse_coordinate(coordinate: str) -> Tuple[int, int]:
    match = re.fullmatch(r"\s*([A-Ja-j])\s*([1-9]|10)\s*", coordinate)
    if not match:
        raise ValueError("Invalid coordinate. Use A1 through J10.")

    row = ord(match.group(1).upper()) - ord("A")
    col = int(match.group(2)) - 1
    return row, col


def validate_orientation(orientation: str) -> str:
    normalized = orientation.strip().lower()
    if normalized not in {"horizontal", "vertical"}:
        raise ValueError("Orientation must be 'horizontal' or 'vertical'.")
    return normalized


def get_sunk_ship_groups(player) -> List[Dict[str, Any]]:
    """
    Exact cell groups for this player's SUNK ships only — e.g.
    [{"name": "Destroyer", "cells": [[3, 4], [3, 5]]}, ...].

    Safe to expose for the opponent too: every cell in a sunk ship is
    already visible as a hit ("X") regardless of hide_ships. This just
    tells the frontend which adjacent hit cells belong to the same ship,
    so it can draw a real boundary between two ships sunk next to each
    other instead of guessing from cell proximity (which breaks exactly
    when ships are placed touching one another).
    """
    return [
        {"name": ship.name, "cells": [[r, c] for (r, c) in ship.positions]}
        for ship in player.board.ships
        if ship.is_sunk()
    ]


def build_game_state(game_data: Dict[str, Any], viewer_index: Optional[int] = None) -> Dict[str, Any]:
    game = game_data["game"]
    if viewer_index is None or viewer_index not in (0, 1):
        viewer_index = game.current_player

    viewer = game.players[viewer_index]
    opponent = game.players[1 - viewer_index]

    return {
        "game_id": game_data["game_id"],
        "mode": game_data["mode"],
        "ai_strategy": game_data.get("ai_strategy"),
        "current_player": game.get_current_player().name,
        "viewer_index": viewer_index,
        "your_player": {
            "name": viewer.name,
            "board": viewer.board.get_board(hide_ships=False),
            "remaining_ships": viewer.board.remaining_ships(),
            "sunk_ships": get_sunk_ship_groups(viewer),
        },
        "opponent_player": {
            "name": opponent.name,
            "board": opponent.board.get_board(hide_ships=True),
            "remaining_ships": opponent.board.remaining_ships(),
            "sunk_ships": get_sunk_ship_groups(opponent),
        },
        "game_over": game.game_over,
        "winner": game.winner,
        "players_ready": game.both_players_ready(),
    }


def get_game_data(game_id: str) -> Dict[str, Any]:
    game_data = games.get(game_id)
    if game_data is None:
        raise HTTPException(status_code=404, detail="Game not found.")
    return game_data


def _place_ai_ships(game_data: Dict[str, Any], ai_instance, player_index: int) -> None:
    """Place a random fleet for an AI on the given player's board."""
    if ai_instance is None:
        return
    player = game_data["game"].players[player_index]
    for placement in ai_instance.place_ships_randomly():
        ship = Ship(placement["name"], placement["size"])
        ship.place(placement["coordinates"])
        player.board.ships.append(ship)


def place_ai_ships_if_needed(game_data: Dict[str, Any]) -> None:
    """
    Place a full random fleet for the AI player.

    Builds Ship objects directly from the AI's already-computed,
    already-validated coordinate lists instead of routing them back
    through Game.place_ship()/Board.get_positions(), which would
    redundantly re-derive coordinates the AI already produced and
    re-run an overlap check place_ships_randomly() already guarantees.
    """
    if game_data["mode"] != "human_vs_ai" or game_data.get("ai_ships_placed"):
        return

    ai = game_data.get("ai")
    if ai is None:
        return

    _place_ai_ships(game_data, ai, 1)
    game_data["ai_ships_placed"] = True


@app.post("/game/create")
def create_game(request: GameCreateRequest) -> Dict[str, Any]:
    mode = request.mode.strip().lower()
    if mode not in {"human_vs_human", "human_vs_ai", "ai_vs_ai"}:
        raise HTTPException(
            status_code=400,
            detail="mode must be 'human_vs_human', 'human_vs_ai', or 'ai_vs_ai'.",
        )

    ai_strategy = None
    ai_instance = None
    ai_strategy_2 = None
    ai_instance_2 = None

    if mode == "human_vs_ai":
        ai_strategy = request.ai_strategy or "random"
        try:
            ai_instance = get_ai(ai_strategy, board_size=BOARD_SIZE)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    if mode == "ai_vs_ai":
        ai_strategy = request.ai_strategy or "random"
        ai_strategy_2 = request.ai_strategy_2 or "hunt_target"
        try:
            ai_instance = get_ai(ai_strategy, board_size=BOARD_SIZE)
            ai_instance_2 = get_ai(ai_strategy_2, board_size=BOARD_SIZE)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    p1_name = request.player1_name or "Player 1"
    p2_name = request.player2_name or "Player 2"

    if mode == "human_vs_ai":
        p2_name = AI_STRATEGY_NAMES.get(ai_strategy, p2_name)
    elif mode == "ai_vs_ai":
        p1_name = AI_STRATEGY_NAMES.get(ai_strategy, p1_name)
        p2_name = AI_STRATEGY_NAMES.get(ai_strategy_2, p2_name)
        if p1_name == p2_name:
            p1_name = f"{p1_name} 1"
            p2_name = f"{p2_name} 2"

    game = Game(p1_name, p2_name)

    game_id = str(uuid4())
    game_data: Dict[str, Any] = {
        "game_id": game_id,
        "game": game,
        "mode": mode,
        "ai_strategy": ai_strategy,
        "ai": ai_instance,
        "ai_strategy_2": ai_strategy_2,
        "ai_2": ai_instance_2,
        "ai_ships_placed": False,
    }
    games[game_id] = game_data

    # For ai_vs_ai, auto-place ships for both players immediately
    if mode == "ai_vs_ai":
        _place_ai_ships(game_data, ai_instance, 0)
        _place_ai_ships(game_data, ai_instance_2, 1)

    return {"game_id": game_id, "state": build_game_state(game_data)}


@app.post("/game/{game_id}/place")
def place_ship(game_id: str, request: PlaceShipRequest) -> Dict[str, Any]:
    game_data = get_game_data(game_id)
    game = game_data["game"]

    if request.player_index not in (0, 1):
        raise HTTPException(status_code=400, detail="player_index must be 0 or 1.")

    try:
        row, col = parse_coordinate(request.coordinate)
        orientation = validate_orientation(request.orientation)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    success = game.place_ship(request.player_index, request.ship_name, row, col, orientation)
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Invalid ship placement: overlap, out of bounds, or duplicate ship.",
        )

    if game_data["mode"] == "human_vs_ai" and request.player_index == 0:
        if game.players[0].has_finished_placing():
            place_ai_ships_if_needed(game_data)

    return {"success": True, "state": build_game_state(game_data)}


@app.post("/game/{game_id}/attack")
def attack_cell(game_id: str, request: AttackRequest) -> Dict[str, Any]:
    game_data = get_game_data(game_id)
    game = game_data["game"]

    if request.player_index not in (0, 1):
        raise HTTPException(status_code=400, detail="player_index must be 0 or 1.")

    if game.current_player != request.player_index:
        raise HTTPException(status_code=400, detail="It is not this player's turn.")

    try:
        row, col = parse_coordinate(request.coordinate)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    result = game.attack(row, col)
    return {"result": result, "state": build_game_state(game_data)}


@app.post("/game/{game_id}/ai-move")
def ai_move(game_id: str) -> Dict[str, Any]:
    game_data = get_game_data(game_id)
    game = game_data["game"]

    if game_data["mode"] != "human_vs_ai" or game_data.get("ai") is None:
        raise HTTPException(status_code=400, detail="AI moves are only available in human_vs_ai mode.")

    ai_player_index = 1
    if game.current_player != ai_player_index:
        raise HTTPException(status_code=400, detail="It is not the AI's turn.")

    ai = game_data["ai"]
    row, col = ai.choose_move()
    result = game.attack(row, col)

    if result.get("valid"):
        is_sunk = result["result"] == "sunk"
        ship_name = result.get("ship") if is_sunk else None
        ship_size = SHIP_SIZES.get(ship_name) if is_sunk else None
        ai.process_result(row, col, result["result"], ship_name, ship_size)

    return {
        "move": {"row": row, "col": col, "coordinate": f"{chr(ord('A') + row)}{col + 1}"},
        "result": result,
        "state": build_game_state(game_data),
    }


@app.post("/game/{game_id}/ai-step")
def ai_step(game_id: str) -> Dict[str, Any]:
    """Execute one AI move for whichever player's turn it is (ai_vs_ai mode)."""
    game_data = get_game_data(game_id)
    game = game_data["game"]

    if game_data["mode"] != "ai_vs_ai":
        raise HTTPException(status_code=400, detail="ai-step is only available in ai_vs_ai mode.")

    if game.game_over:
        raise HTTPException(status_code=400, detail="Game is already over.")

    current_idx = game.current_player
    ai = game_data["ai"] if current_idx == 0 else game_data.get("ai_2")

    if ai is None:
        raise HTTPException(status_code=400, detail=f"No AI instance for player {current_idx}.")

    row, col = ai.choose_move()
    result = game.attack(row, col)

    if result.get("valid"):
        is_sunk = result["result"] == "sunk"
        ship_name = result.get("ship") if is_sunk else None
        ship_size = SHIP_SIZES.get(ship_name) if is_sunk else None
        ai.process_result(row, col, result["result"], ship_name, ship_size)

    return {
        "move": {"row": row, "col": col, "coordinate": f"{chr(ord('A') + row)}{col + 1}"},
        "result": result,
        "state": build_game_state(game_data),
    }


@app.get("/game/{game_id}")
def get_game(game_id: str, viewer_index: Optional[int] = Query(None, ge=0, le=1)) -> Dict[str, Any]:
    return build_game_state(get_game_data(game_id), viewer_index)


@app.get("/ai/strategies")
def ai_strategies() -> List[Dict[str, str]]:
    return list_strategies()
@app.get("/")
def root():
    return {"message": "Battleship AI Backend is running"}

