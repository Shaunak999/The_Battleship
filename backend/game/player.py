from .board import Board
from .ship import Ship


class Player:
    """
    A game participant that owns a board and can interact with an opponent.

    Responsibilities:
        - hold a Board instance for the player's fleet
        - place ships on that board
        - fire on an opponent's board through board.attack()
        - report whether the player has been defeated
    """

    def __init__(self, name="Player", board=None, ai=None):
        self.name = name
        self.board = board if board is not None else Board()
        self.ai = ai
        self.last_attack_result = None

    def place_ship(self, ship, row, col, orientation):
        """
        Delegate placement to the player's board.

        Accepts either a Ship instance or a standard ship name string.

        Example:
            player.place_ship(Ship("Destroyer", 2), 0, 0, "horizontal")
            player.place_ship("Destroyer", 0, 0, "horizontal")
        """
        if isinstance(ship, str):
            ship_name = ship
            if ship_name not in Board.SHIPS:
                raise ValueError(f"Unknown ship: {ship_name}")

            for existing_ship in self.board.ships:
                if existing_ship.name == ship_name:
                    return False

            ship_obj = Ship(ship_name, Board.SHIPS[ship_name])
            return self.board.place_ship(ship_obj, row, col, orientation)

        if not isinstance(ship, Ship):
            raise TypeError("ship must be a Ship instance or a ship-name string")

        return self.board.place_ship(ship, row, col, orientation)

    def has_finished_placing(self):
        """
        Check whether every standard Battleship ship has been placed once.
        """
        return len(self.board.ships) == len(Board.SHIPS)

    def place_ships(self, ships):
        """
        Convenience method for bulk fleet setup.
        ships is expected to be a sequence of (ship, row, col, orientation)
        tuples.
        """
        results = []

        for ship, row, col, orientation in ships:
            results.append(self.place_ship(ship, row, col, orientation))

        return results

    def fire_at(self, opponent, row, col):
        """
        Fire at a coordinate on an opponent's board.

        Parameters:
            opponent: another Player instance
            row: zero-index row coordinate
            col: zero-index column coordinate

        Returns:
            The dict returned from Board.attack() describing the result.
        """
        if not isinstance(opponent, Player):
            raise TypeError("opponent must be a Player instance")

        if opponent is self:
            raise ValueError("A player cannot attack itself")

        result = opponent.board.attack(row, col)
        self.last_attack_result = result
        return result

    def is_defeated(self):
        """
        Returns True when every ship on this player's board has been sunk.
        """
        return self.board.all_ships_sunk()

    def remaining_ships(self):
        """
        Return the names of ships still afloat.
        """
        return self.board.remaining_ships()

    def get_board(self, hide_ships=False):
        """
        Return a board view for serialization or display.
        """
        return self.board.get_board(hide_ships=hide_ships)

    def display(self, hide_ships=False):
        """
        Render the player's board in the terminal.
        """
        self.board.display(hide_ships=hide_ships)

    def __repr__(self):
        return (
            f"Player(name={self.name!r}, "
            f"defeated={self.is_defeated()}, "
            f"ships={len(self.board.ships)})"
        )
