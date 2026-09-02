import os
import re
import sys
import json
import asyncio
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from ai import AI_STRATEGY_NAMES, get_ai, list_strategies
    from ai.base_ai import BOARD_SIZE, SHIP_DEFINITIONS
    from game.game import Game
    from game.ship import Ship
    from multiplayer.room import Room, Role
    from multiplayer.game_manager import GameManager
except ImportError:
    from backend.ai import AI_STRATEGY_NAMES, get_ai, list_strategies
    from backend.ai.base_ai import BOARD_SIZE, SHIP_DEFINITIONS
    from backend.game.game import Game
    from backend.game.ship import Ship
    from backend.multiplayer.room import Room, Role
    from backend.multiplayer.game_manager import GameManager

SHIP_SIZES: Dict[str, int] = dict(SHIP_DEFINITIONS)

app = FastAPI(
    title="Battleship AI Backend",
    description="Local Battleship game engine with Human vs Human, Human vs AI, and LAN Multiplayer.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# ── In-memory state ────────────────────────────────────────────────────
games: Dict[str, Dict[str, Any]] = {}
mp_manager = GameManager()


# ── Request models ─────────────────────────────────────────────────────

class GameCreateRequest(BaseModel):
    mode: str = "human_vs_human"
    player1_name: Optional[str] = "Player 1"
    player2_name: Optional[str] = "Player 2"
    ai_strategy: Optional[str] = None
    ai_strategy_2: Optional[str] = None


class PlaceShipRequest(BaseModel):
    player_index: int
    ship_name: str
    coordinate: str
    orientation: str


class AttackRequest(BaseModel):
    player_index: int
    coordinate: str


class MpCreateRequest(BaseModel):
    player_name: Optional[str] = "Player 1"


class MpJoinRequest(BaseModel):
    game_id: str
    player_name: Optional[str] = "Player 2"


class MpSpectateRequest(BaseModel):
    game_id: str


# ── Helpers ────────────────────────────────────────────────────────────

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
        "current_player_index": game.current_player,
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


def build_spectator_state(room: Room) -> Dict[str, Any]:
    """Build a full God-view state for spectators."""
    game = room.game
    if game is None:
        return {"game_id": room.game_id, "status": "waiting"}

    p1 = game.players[0]
    p2 = game.players[1]

    return {
        "game_id": room.game_id,
        "status": "playing" if not game.game_over else "over",
        "current_player": game.current_player,
        "current_player_name": game.get_current_player().name,
        "winner": game.winner,
        "player1": {
            "name": p1.name,
            "board": p1.board.get_board(hide_ships=False),
            "ships": [
                {
                    "name": s.name,
                    "size": s.size,
                    "positions": [[r, c] for r, c in s.positions],
                    "hits": [[r, c] for r, c in s.hits],
                    "sunk": s.is_sunk(),
                }
                for s in p1.board.ships
            ],
            "remaining_ships": p1.board.remaining_ships(),
            "sunk_ships": get_sunk_ship_groups(p1),
        },
        "player2": {
            "name": p2.name,
            "board": p2.board.get_board(hide_ships=False),
            "ships": [
                {
                    "name": s.name,
                    "size": s.size,
                    "positions": [[r, c] for r, c in s.positions],
                    "hits": [[r, c] for r, c in s.hits],
                    "sunk": s.is_sunk(),
                }
                for s in p2.board.ships
            ],
            "remaining_ships": p2.board.remaining_ships(),
            "sunk_ships": get_sunk_ship_groups(p2),
        },
        "player1_ready": room.player1_ready,
        "player2_ready": room.player2_ready,
        "statistics": room.get_statistics(),
        "event_log": room.event_log[-50:],
    }


def build_player_state(room: Room, player_index: int) -> Dict[str, Any]:
    """Build a player-view state hiding opponent ships."""
    game = room.game
    if game is None:
        return {"game_id": room.game_id, "status": "waiting"}

    viewer = game.players[player_index]
    opponent = game.players[1 - player_index]

    return {
        "game_id": room.game_id,
        "status": "playing" if not game.game_over else "over",
        "current_player": game.current_player,
        "current_player_name": game.get_current_player().name,
        "viewer_index": player_index,
        "winner": game.winner,
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
        "player1_ready": room.player1_ready,
        "player2_ready": room.player2_ready,
    }


def get_game_data(game_id: str) -> Dict[str, Any]:
    game_data = games.get(game_id)
    if game_data is None:
        raise HTTPException(status_code=404, detail="Game not found.")
    return game_data


def _place_ai_ships(game_data: Dict[str, Any], ai_instance, player_index: int) -> None:
    if ai_instance is None:
        return
    player = game_data["game"].players[player_index]
    for placement in ai_instance.place_ships_randomly():
        ship = Ship(placement["name"], placement["size"])
        ship.place(placement["coordinates"])
        player.board.ships.append(ship)


def place_ai_ships_if_needed(game_data: Dict[str, Any]) -> None:
    if game_data["mode"] != "human_vs_ai" or game_data.get("ai_ships_placed"):
        return
    ai = game_data.get("ai")
    if ai is None:
        return
    _place_ai_ships(game_data, ai, 1)
    game_data["ai_ships_placed"] = True


# ── Broadcast helpers for multiplayer ──────────────────────────────────

async def broadcast_to_room(room: Room, message: Dict[str, Any]) -> None:
    """Send a JSON message to every connected client in the room."""
    data = json.dumps(message)
    for role in Role:
        for ws in list(room.connections[role]):
            try:
                await ws.send_text(data)
            except Exception:
                room.remove_connection(ws)


async def send_to_role(room: Room, role: Role, message: Dict[str, Any]) -> None:
    """Send a JSON message to all connections of a specific role."""
    data = json.dumps(message)
    for ws in list(room.connections[role]):
        try:
            await ws.send_text(data)
        except Exception:
            room.remove_connection(ws)


async def send_to_ws(ws: WebSocket, message: Dict[str, Any]) -> None:
    """Send a JSON message to a single WebSocket."""
    try:
        await ws.send_text(json.dumps(message))
    except Exception:
        pass


# ======================================================================
# EXISTING REST API — preserved unchanged
# ======================================================================

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


# ======================================================================
# MULTIPLAYER REST ENDPOINTS
# ======================================================================

@app.post("/mp/create")
def mp_create(request: MpCreateRequest) -> Dict[str, Any]:
    """Create a new LAN multiplayer room."""
    room = mp_manager.create_room(request.player_name or "Player 1")
    return {
        "game_id": room.game_id,
        "player_name": room.player_names[Role.PLAYER_1],
    }


@app.post("/mp/join")
def mp_join(request: MpJoinRequest) -> Dict[str, Any]:
    """Check if a multiplayer room exists and has a player slot."""
    room = mp_manager.get_room(request.game_id.upper())
    if room is None:
        raise HTTPException(status_code=404, detail="Game not found.")
    if room.player1_connected and room.player2_connected:
        raise HTTPException(status_code=400, detail="Game is full.")
    return {
        "game_id": room.game_id,
        "player_name": request.player_name or "Player 2",
    }


@app.post("/mp/spectate")
def mp_spectate(request: MpSpectateRequest) -> Dict[str, Any]:
    """Check if a multiplayer room exists (for spectating)."""
    room = mp_manager.get_room(request.game_id.upper())
    if room is None:
        raise HTTPException(status_code=404, detail="Game not found.")
    return {"game_id": room.game_id}


@app.get("/mp/rooms")
def mp_list_rooms() -> List[Dict[str, Any]]:
    """List active multiplayer rooms."""
    return [
        {
            "game_id": r.game_id,
            "player1_connected": r.player1_connected,
            "player2_connected": r.player2_connected,
            "spectator_count": r.spectator_count,
        }
        for r in mp_manager.rooms.values()
    ]


# ======================================================================
# MULTIPLAYER WEBSOCKET ENDPOINT
# ======================================================================

@app.websocket("/ws/{game_id}/{role}")
async def websocket_endpoint(websocket: WebSocket, game_id: str, role: str, player_name: Optional[str] = Query(None)):
    """
    WebSocket endpoint for multiplayer.

    Roles: "player1", "player2", "spectator"
    """
    await websocket.accept()
    msg_player_name = player_name or ""

    # Validate role
    try:
        ws_role = Role(role)
    except ValueError:
        await send_to_ws(websocket, {"type": "error", "message": f"Invalid role: {role}"})
        await websocket.close()
        return

    # Get or wait for room
    room = mp_manager.get_room(game_id.upper())
    if room is None:
        # For player1: room will be created via REST first
        # For player2/spectator: room must exist
        if ws_role == Role.PLAYER_1:
            # Create room on-demand if player1 connects first
            room = mp_manager.create_room()
            # Overwrite the game_id to match
            mp_manager.rooms.pop(room.game_id, None)
            room.game_id = game_id.upper()
            mp_manager.rooms[room.game_id] = room
        else:
            await send_to_ws(websocket, {"type": "error", "message": "Game not found."})
            await websocket.close()
            return

    # Re-connection handling / slot replacement:
    # If a player slot already has an existing connection (e.g. React StrictMode re-mount or page refresh),
    # close old connections for that role so the new connection takes over seamlessly.
    if ws_role in (Role.PLAYER_1, Role.PLAYER_2):
        existing_connections = list(room.connections[ws_role])
        for old_ws in existing_connections:
            room.remove_connection(old_ws)
            try:
                await old_ws.close()
            except Exception:
                pass

    # Register connection
    room.add_connection(websocket, ws_role)

    # If player2 just joined, set their name
    if ws_role == Role.PLAYER_2:
        room.player_names[Role.PLAYER_2] = msg_player_name or "Player 2"

    # Send welcome
    await send_to_ws(websocket, {
        "type": "welcome",
        "game_id": room.game_id,
        "role": ws_role.value,
    })

    # Notify others
    await broadcast_to_room(room, {
        "type": "player_joined",
        "role": ws_role.value,
        "player1_connected": room.player1_connected,
        "player2_connected": room.player2_connected,
        "spectator_count": room.spectator_count,
    })

    # Create game instance once both players are connected
    if room.game is None and room.player1_connected and room.player2_connected:
        p1_name = room.player_names.get(Role.PLAYER_1, "Player 1")
        p2_name = room.player_names.get(Role.PLAYER_2, "Player 2")
        room.game = Game(p1_name, p2_name)
        # Send initial state to both players so they can begin ship placement
        for p_idx, p_role in enumerate([Role.PLAYER_1, Role.PLAYER_2]):
            state = build_player_state(room, p_idx)
            await send_to_role(room, p_role, {"type": "state_update", "state": state})
        # Notify spectators
        await send_to_role(room, Role.SPECTATOR, {
            "type": "spectator_state",
            "state": build_spectator_state(room),
        })

    # Send spectator state to newly connected spectator
    if ws_role == Role.SPECTATOR:
        await send_to_ws(websocket, {
            "type": "spectator_state",
            "state": build_spectator_state(room),
        })

    # Listen for messages
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await send_to_ws(websocket, {"type": "error", "message": "Invalid JSON"})
                continue

            msg_type = msg.get("type", "")

            # ── PLAYER ACTIONS ────────────────────────────────────

            if ws_role in (Role.PLAYER_1, Role.PLAYER_2):
                player_index = 0 if ws_role == Role.PLAYER_1 else 1

                if msg_type == "place_ship":
                    await handle_mp_place_ship(room, websocket, player_index, msg)

                elif msg_type == "ready":
                    await handle_mp_ready(room, ws_role, player_index)

                elif msg_type == "attack":
                    await handle_mp_attack(room, websocket, ws_role, player_index, msg)

                else:
                    await send_to_ws(websocket, {
                        "type": "error",
                        "message": f"Unknown message type: {msg_type}",
                    })

            # ── SPECTATOR ACTIONS (read-only) ────────────────────

            elif ws_role == Role.SPECTATOR:
                if msg_type == "get_state":
                    if room.game is not None:
                        await send_to_ws(websocket, {
                            "type": "spectator_state",
                            "state": build_spectator_state(room),
                        })
                else:
                    await send_to_ws(websocket, {
                        "type": "error",
                        "message": "Spectators cannot perform actions.",
                    })

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        departed_role = room.remove_connection(websocket)
        if departed_role:
            await broadcast_to_room(room, {
                "type": "player_disconnected",
                "role": departed_role.value,
                "player1_connected": room.player1_connected,
                "player2_connected": room.player2_connected,
                "spectator_count": room.spectator_count,
            })
            # Terminate game if Player 1 or Player 2 leaves or disconnects
            if departed_role in (Role.PLAYER_1, Role.PLAYER_2):
                departed_name = room.player_names.get(departed_role, "Opponent")
                await broadcast_to_room(room, {
                    "type": "game_terminated",
                    "reason": f"{departed_name} left or disconnected from the game.",
                    "disconnected_role": departed_role.value,
                })
                mp_manager.remove_room(room.game_id)


# ======================================================================
# MULTIPLAYER ACTION HANDLERS
# ======================================================================

async def handle_mp_place_ship(room: Room, ws: WebSocket, player_index: int, msg: Dict[str, Any]):
    """Handle a ship placement from a multiplayer player."""
    if room.game is None:
        await send_to_ws(ws, {"type": "error", "message": "Game not started yet."})
        return

    if room.game.game_over:
        await send_to_ws(ws, {"type": "error", "message": "Game is over."})
        return

    ship_name = msg.get("ship_name")
    coordinate = msg.get("coordinate", "")
    orientation = msg.get("orientation", "horizontal")

    try:
        row, col = parse_coordinate(coordinate)
        orientation = validate_orientation(orientation)
    except ValueError as exc:
        await send_to_ws(ws, {"type": "error", "message": str(exc)})
        return

    success = room.game.place_ship(player_index, ship_name, row, col, orientation)
    if not success:
        await send_to_ws(ws, {
            "type": "error",
            "message": "Invalid ship placement: overlap, out of bounds, or duplicate.",
        })
        return

    # Check if this player finished placing
    player = room.game.players[player_index]
    all_placed = player.has_finished_placing()
    if all_placed:
        room.set_player_ready(player_index)

    # Send updated state to the placing player
    state = build_player_state(room, player_index)
    await send_to_ws(ws, {"type": "ship_placed", "state": state, "all_placed": all_placed})

    # Notify spectators
    await send_to_role(room, Role.SPECTATOR, {
        "type": "spectator_state",
        "state": build_spectator_state(room),
    })

    # Broadcast player ready state
    await broadcast_to_room(room, {
        "type": "player_ready",
        "player_index": player_index,
        "player1_ready": room.player1_ready,
        "player2_ready": room.player2_ready,
    })

    # If both ready, start the game
    if room.both_ready and room.game is not None:
        room.game_start_time = room.game_start_time or time.time()
        await broadcast_to_room(room, {
            "type": "game_started",
            "current_player": room.game.current_player,
            "current_player_name": room.game.get_current_player().name,
        })
        for p_idx, p_role in enumerate([Role.PLAYER_1, Role.PLAYER_2]):
            st = build_player_state(room, p_idx)
            await send_to_role(room, p_role, {"type": "state_update", "state": st})


async def handle_mp_ready(room: Room, ws_role: Role, player_index: int):
    """Handle a player becoming ready (all ships placed)."""
    if room.game is None:
        return

    room.set_player_ready(player_index)

    # Notify everyone
    await broadcast_to_room(room, {
        "type": "player_ready",
        "player_index": player_index,
        "player1_ready": room.player1_ready,
        "player2_ready": room.player2_ready,
    })

    if room.both_ready:
        room.game_start_time = time.time()
        await broadcast_to_room(room, {
            "type": "game_started",
            "current_player": room.game.current_player,
            "current_player_name": room.game.get_current_player().name,
        })
        for p_idx, p_role in enumerate([Role.PLAYER_1, Role.PLAYER_2]):
            st = build_player_state(room, p_idx)
            await send_to_role(room, p_role, {"type": "state_update", "state": st})


async def handle_mp_attack(room: Room, sender_ws: WebSocket, ws_role: Role, player_index: int, msg: Dict[str, Any]):
    """Handle an attack from a multiplayer player."""
    game = room.game

    if game is None:
        await send_to_ws(sender_ws, {"type": "error", "message": "Game not started."})
        return

    if game.game_over:
        await send_to_ws(sender_ws, {"type": "error", "message": "Game is over."})
        return

    if not room.both_ready:
        await send_to_ws(sender_ws, {"type": "error", "message": "Both players must place ships first."})
        return

    if game.current_player != player_index:
        await send_to_ws(sender_ws, {"type": "error", "message": "Not your turn."})
        return

    row = msg.get("row")
    col = msg.get("col")

    if row is None or col is None:
        await send_to_ws(sender_ws, {"type": "error", "message": "Missing row/col."})
        return

    if not (0 <= row < 10 and 0 <= col < 10):
        await send_to_ws(sender_ws, {"type": "error", "message": "Invalid coordinates."})
        return

    coord_str = f"{chr(ord('A') + row)}{col + 1}"
    result = game.attack(row, col)

    if not result.get("valid"):
        await send_to_ws(sender_ws, {"type": "error", "message": result.get("message", "Invalid attack.")})
        return

    # Record attack for stats
    room.attack_history.append({
        "attacker": player_index,
        "row": row,
        "col": col,
        "coordinate": coord_str,
        "result": result["result"],
    })

    # Log event
    attacker_name = game.players[player_index].name
    event_data = {
        "attacker": attacker_name,
        "attacker_index": player_index,
        "row": row,
        "col": col,
        "coordinate": coord_str,
        "result": result["result"],
    }
    if result["result"] == "sunk":
        event_data["ship_name"] = result.get("ship")
    room.log_event("attack", event_data)

    # Build result message
    attack_result = {
        "type": "attack_result",
        "attacker": player_index,
        "attacker_name": attacker_name,
        "row": row,
        "col": col,
        "coordinate": coord_str,
        "result": result["result"],
        "message": result.get("message", ""),
        "next_player": game.current_player,
        "next_player_name": game.get_current_player().name,
        "game_over": game.game_over,
        "winner": game.winner,
    }
    if result["result"] == "sunk":
        attack_result["ship_name"] = result.get("ship")
        attack_result["ship_size"] = SHIP_SIZES.get(result.get("ship"))

    # Send to all clients
    await broadcast_to_room(room, attack_result)

    # Send updated states to each player (role-aware)
    for p_idx in range(2):
        p_role = Role.PLAYER_1 if p_idx == 0 else Role.PLAYER_2
        state = build_player_state(room, p_idx)
        await send_to_role(room, p_role, {"type": "state_update", "state": state})

    # Send full state to spectators
    await send_to_role(room, Role.SPECTATOR, {
        "type": "spectator_state",
        "state": build_spectator_state(room),
    })
