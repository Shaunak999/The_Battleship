"""
Multiplayer game room.

Tracks players, spectators, game state, and WebSocket connections
for a single LAN multiplayer game.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class Role(str, Enum):
    PLAYER_1 = "player1"
    PLAYER_2 = "player2"
    SPECTATOR = "spectator"


class Room:
    """One multiplayer game room."""

    def __init__(self, game_id: str, player1_name: str = "Player 1"):
        self.game_id = game_id
        self.created_at = time.time()

        # Game engine (set via GameManager after creation)
        self.game = None
        self.mode = "human_vs_human"

        # Connected clients: role -> set of WebSocket connections
        self.connections: Dict[Role, Set] = {
            Role.PLAYER_1: set(),
            Role.PLAYER_2: set(),
            Role.SPECTATOR: set(),
        }

        # Role assignment: WebSocket -> Role
        self._ws_roles: Dict[Any, Role] = {}

        # Player identity: role -> player name
        self.player_names: Dict[Role, str] = {
            Role.PLAYER_1: player1_name,
            Role.PLAYER_2: "Player 2",
        }

        # Placement tracking
        self.player1_ready = False
        self.player2_ready = False

        # Game events log (for spectators)
        self.event_log: List[Dict[str, Any]] = []
        self.event_counter = 0

        # Attack history for statistics
        self.attack_history: List[Dict[str, Any]] = []

        # Game start time
        self.game_start_time: Optional[float] = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def add_connection(self, ws: Any, role: Role) -> None:
        """Register a WebSocket connection with its role."""
        self.connections[role].add(ws)
        self._ws_roles[ws] = role

    def remove_connection(self, ws: Any) -> Optional[Role]:
        """Remove a WebSocket connection. Returns the role it had."""
        role = self._ws_roles.pop(ws, None)
        if role:
            self.connections[role].discard(ws)
        return role

    def get_role(self, ws: Any) -> Optional[Role]:
        """Get the role for a WebSocket connection."""
        return self._ws_roles.get(ws)

    # ------------------------------------------------------------------
    # Player queries
    # ------------------------------------------------------------------

    @property
    def player1_connected(self) -> bool:
        return len(self.connections[Role.PLAYER_1]) > 0

    @property
    def player2_connected(self) -> bool:
        return len(self.connections[Role.PLAYER_2]) > 0

    @property
    def is_full(self) -> bool:
        """True when both player slots are taken."""
        return self.player1_connected and self.player2_connected

    @property
    def spectator_count(self) -> int:
        return len(self.connections[Role.SPECTATOR])

    # ------------------------------------------------------------------
    # Ready tracking
    # ------------------------------------------------------------------

    def set_player_ready(self, player_index: int) -> None:
        if player_index == 0:
            self.player1_ready = True
        elif player_index == 1:
            self.player2_ready = True

    @property
    def both_ready(self) -> bool:
        return self.player1_ready and self.player2_ready

    # ------------------------------------------------------------------
    # Event log
    # ------------------------------------------------------------------

    def log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Append an event to the log for spectator history."""
        self.event_counter += 1
        self.event_log.append({
            "id": self.event_counter,
            "type": event_type,
            "time": time.time(),
            **data,
        })

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Compute per-player statistics from the game engine."""
        if not self.game:
            return {}

        stats = {}
        for idx, player in enumerate(self.game.players):
            board = player.board
            total_attacks = len(board.attacks)
            hits_on_board = sum(
                1 for pos in board.attacks
                if any(pos in ship.positions for ship in board.ships)
            )
            misses_on_board = total_attacks - hits_on_board

            shots_fired = len(self.attack_history) if idx == 0 else len(self.attack_history)
            player_shots = [
                a for a in self.attack_history if a.get("attacker") == idx
            ]
            player_hits = sum(1 for a in player_shots if a.get("result") in ("hit", "sunk"))
            player_misses = sum(1 for a in player_shots if a.get("result") == "miss")
            player_total = len(player_shots)
            accuracy = round(player_hits / player_total * 100) if player_total > 0 else 0

            stats[idx] = {
                "name": player.name,
                "shots": player_total,
                "hits": player_hits,
                "misses": player_misses,
                "accuracy": accuracy,
                "ships_remaining": len(board.remaining_ships()),
                "ships_sunk": len([s for s in board.ships if s.is_sunk()]),
            }

        game_duration = time.time() - (self.game_start_time or self.created_at)
        return {
            "players": stats,
            "current_turn": self.game.current_player,
            "game_status": "over" if self.game.game_over else "playing",
            "total_shots": len(self.attack_history),
            "game_duration": round(game_duration, 1),
        }

    def __repr__(self) -> str:
        return (
            f"Room(game_id={self.game_id!r}, "
            f"p1={self.player1_connected}, p2={self.player2_connected}, "
            f"spectators={self.spectator_count})"
        )
