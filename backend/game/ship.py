class Ship:
    def __init__(self, name, size):
        self.name = name
        self.size = size
        self.positions = []
        self.hits = set()

    def place(self, positions):
        """
        Store the coordinates occupied by this ship.
        """
        if len(positions) != self.size:
            raise ValueError(
                f"{self.name} requires {self.size} positions."
            )

        self.positions = positions

    def hit(self, position):
        """
        Register a hit on this ship.
        Returns True if the position belongs to the ship.
        """
        if position in self.positions:
            self.hits.add(position)
            return True

        return False

    def is_sunk(self):
        """
        Returns True if all positions of the ship have been hit.
        """
        return len(self.hits) == self.size

    def __repr__(self):
        return f"Ship(name='{self.name}', size={self.size})"