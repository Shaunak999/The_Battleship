class Board:
    SIZE = 10

    SHIPS = {
        "Carrier": 5,
        "Battleship": 4,
        "Cruiser": 3,
        "Submarine": 3,
        "Destroyer": 2,
    }

    def __init__(self):
        self.ships = []
        self.attacks = set()

    # --------------------------------------------------
    # Coordinate validation
    # --------------------------------------------------

    def is_valid_coordinate(self, row, col):
        return (
            0 <= row < self.SIZE
            and 0 <= col < self.SIZE
        )

    # --------------------------------------------------
    # Generate positions for a ship
    # --------------------------------------------------

    def get_positions(self, row, col, size, orientation):
        """
        Generate all coordinates occupied by a ship.

        orientation:
            'horizontal'
            'vertical'
        """

        positions = []

        for i in range(size):

            if orientation == "horizontal":
                new_row = row
                new_col = col + i

            elif orientation == "vertical":
                new_row = row + i
                new_col = col

            else:
                raise ValueError(
                    "Orientation must be 'horizontal' or 'vertical'."
                )

            if not self.is_valid_coordinate(new_row, new_col):
                return None

            positions.append((new_row, new_col))

        return positions

    # --------------------------------------------------
    # Check whether ship placement is valid
    # --------------------------------------------------

    def is_valid_placement(self, positions):
        """
        Check whether a ship can be placed at the
        specified positions.
        """

        if positions is None:
            return False

        occupied_positions = set()

        for ship in self.ships:
            occupied_positions.update(ship.positions)

        for position in positions:
            if position in occupied_positions:
                return False

        return True

    # --------------------------------------------------
    # Place a ship
    # --------------------------------------------------

    def place_ship(self, ship, row, col, orientation):
        """
        Attempt to place a ship on the board.
        """

        positions = self.get_positions(
            row,
            col,
            ship.size,
            orientation
        )

        if not self.is_valid_placement(positions):
            return False

        ship.place(positions)
        self.ships.append(ship)

        return True

    # --------------------------------------------------
    # Attack
    # --------------------------------------------------

    def attack(self, row, col):
        """
        Attack a cell on the board.

        Returns a dictionary describing the result.
        """

        position = (row, col)

        # Invalid coordinate
        if not self.is_valid_coordinate(row, col):
            return {
                "valid": False,
                "result": "invalid",
                "message": "Invalid coordinate."
            }

        # Already attacked
        if position in self.attacks:
            return {
                "valid": False,
                "result": "already_attacked",
                "message": "This cell has already been attacked."
            }

        # Record attack
        self.attacks.add(position)

        # Check every ship
        for ship in self.ships:

            if ship.hit(position):

                if ship.is_sunk():
                    return {
                        "valid": True,
                        "result": "sunk",
                        "ship": ship.name,
                        "message": f"You sunk the {ship.name}!"
                    }

                return {
                    "valid": True,
                    "result": "hit",
                    "ship": ship.name,
                    "message": "Hit!"
                }

        # No ship found
        return {
            "valid": True,
            "result": "miss",
            "message": "Miss!"
        }

    # --------------------------------------------------
    # Check if all ships are sunk
    # --------------------------------------------------

    def all_ships_sunk(self):
        """
        Returns True if all placed ships are sunk.
        """

        if not self.ships:
            return False

        return all(ship.is_sunk() for ship in self.ships)

    # --------------------------------------------------
    # Get remaining ships
    # --------------------------------------------------

    def remaining_ships(self):
        return [
            ship.name
            for ship in self.ships
            if not ship.is_sunk()
        ]

    # --------------------------------------------------
    # Get board state
    # --------------------------------------------------

    def get_board(self, hide_ships=False):
        """
        Return a 10x10 representation of the board.

        Symbols:

        ~ = water
        S = ship
        X = hit
        O = miss

        If hide_ships=True, unhit ships are hidden.
        """

        board = [
            ["~" for _ in range(self.SIZE)]
            for _ in range(self.SIZE)
        ]

        # Add ships
        for ship in self.ships:

            for position in ship.positions:

                row, col = position

                if position in ship.hits:
                    board[row][col] = "X"

                elif not hide_ships:
                    board[row][col] = "S"

        # Add misses
        for position in self.attacks:

            row, col = position

            # Don't overwrite hits
            if board[row][col] == "~":
                board[row][col] = "O"

        return board

    # --------------------------------------------------
    # Display board in terminal
    # --------------------------------------------------

    def display(self, hide_ships=False):

        board = self.get_board(hide_ships)

        print("\n    " + " ".join(
            chr(ord("A") + i)
            for i in range(self.SIZE)
        ))

        for row in range(self.SIZE):

            print(
                f"{row + 1:2}  "
                + " ".join(board[row])
            )