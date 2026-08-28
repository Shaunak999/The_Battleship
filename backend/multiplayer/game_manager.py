"""
GameManager — in-memory registry of all multiplayer game rooms.

No database.  Rooms are created on demand and garbage-collected
when stale.
"""

from __future__ import annotations

import random
import string
from typing import Any, Dict, Optional

from .room import Room


def _generate_game_id() -> str:
    """Generate a short, human-readable game ID like 'ABC123'."""
    letters = "".join(random.choices(string.ascii_uppercase, k=3))
    digits = "".join(random.choices(string.digits, k=3))
    return f"{letters}{digits}"


class GameManager:
    """Singleton-ish registry of all active multiplayer rooms."""

    def __init__(self) -> None:
        self.rooms: Dict[str, Room] = {}

    # ------------------------------------------------------------------
    # Room lifecycle
    # ------------------------------------------------------------------

    def create_room(self, player1_name: str = "Player 1") -> Room:
        """Create a new room with a unique game ID."""
        for _ in range(20):
            game_id = _generate_game_id()
            if game_id not in self.rooms:
                room = Room(game_id, player1_name)
                self.rooms[game_id] = room
                return room
        raise RuntimeError("Could not generate a unique game ID")

    def get_room(self, game_id: str) -> Optional[Room]:
        """Look up a room by game ID (case-insensitive)."""
        return self.rooms.get(game_id.upper())

    def remove_room(self, game_id: str) -> None:
        """Delete a room from the registry."""
        self.rooms.pop(game_id.upper(), None)

    # ------------------------------------------------------------------
    # Join helpers
    # ------------------------------------------------------------------

    def join_as_player(self, game_id: str) -> Optional[Room]:
        """Return the room if a player slot is available, else None."""
        room = self.get_room(game_id)
        if room is None:
            return None
        if room.player1_connected and room.player2_connected:
            return None  # both slots full
        return room

    def join_as_spectator(self, game_id: str) -> Optional[Room]:
        """Return the room if it exists (spectators always allowed)."""
        return self.get_room(game_id)
