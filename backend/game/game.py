from .player import Player


class Game:
    def __init__(self, player1_name="Player 1", player2_name="Player 2"):

        self.players = [
            Player(player1_name),
            Player(player2_name)
        ]

        self.current_player = 0
        self.game_over = False
        self.winner = None

    # --------------------------------------------------
    # Current player
    # --------------------------------------------------

    def get_current_player(self):
        return self.players[self.current_player]

    # --------------------------------------------------
    # Opponent
    # --------------------------------------------------

    def get_opponent(self):
        return self.players[1 - self.current_player]

    # --------------------------------------------------
    # Place ship
    # --------------------------------------------------

    def place_ship(
        self,
        player_index,
        ship_name,
        row,
        col,
        orientation
    ):

        if player_index not in [0, 1]:
            raise ValueError("Invalid player index.")

        player = self.players[player_index]

        return player.place_ship(
            ship_name,
            row,
            col,
            orientation
        )

    # --------------------------------------------------
    # Check whether both players are ready
    # --------------------------------------------------

    def both_players_ready(self):
        return all(
            len(player.board.ships) == len(player.board.SHIPS)
            for player in self.players
        )

    # --------------------------------------------------
    # Attack
    # --------------------------------------------------

    def attack(self, row, col):

        if self.game_over:
            return {
                "valid": False,
                "result": "game_over",
                "message": "The game is already over."
            }

        attacker = self.get_current_player()
        defender = self.get_opponent()

        # Don't allow attacks until ships are placed
        if not self.both_players_ready():
            return {
                "valid": False,
                "result": "not_ready",
                "message": "Both players must place all ships first."
            }

        result = defender.board.attack(row, col)

        # Invalid attack — don't change turn
        if not result["valid"]:
            return result

        # Check victory
        if defender.board.all_ships_sunk():

            self.game_over = True
            self.winner = attacker.name

            result["game_over"] = True
            result["winner"] = attacker.name

            return result

        # Change turn
        self.current_player = 1 - self.current_player

        result["game_over"] = False
        result["next_player"] = self.get_current_player().name

        return result

    # --------------------------------------------------
    # Get public game state
    # --------------------------------------------------

    def get_state(self):

        player1 = self.players[0]
        player2 = self.players[1]

        return {
            "current_player": self.get_current_player().name,
            "game_over": self.game_over,
            "winner": self.winner,

            "players": [
                {
                    "name": player1.name,
                    "board": player1.board.get_board(
                        hide_ships=False
                    ),
                    "remaining_ships":
                        player1.board.remaining_ships()
                },

                {
                    "name": player2.name,
                    "board": player2.board.get_board(
                        hide_ships=True
                    ),
                    "remaining_ships":
                        player2.board.remaining_ships()
                }
            ]
        }