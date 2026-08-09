import { useState } from "react";
import Board from "./Board";

const ALL_SHIPS = [
  { name: "Carrier", size: 5 },
  { name: "Battleship", size: 4 },
  { name: "Cruiser", size: 3 },
  { name: "Submarine", size: 3 },
  { name: "Destroyer", size: 2 },
];

/**
 * board: the player's own 10x10 board array (to render placed ships as they go).
 * placedShipNames: ships already placed, derived from state.your_player.remaining_ships
 *   during setup — before the game starts no ship can be sunk, so this list is
 *   exactly "ships placed so far."
 * onPlace(shipName, row, col, orientation) -> Promise, expected to call the
 *   backend and refresh state; throwing surfaces the error message here.
 */
export default function ShipPlacement({ board, placedShipNames, onPlace }) {
  const [orientation, setOrientation] = useState("horizontal");
  const [error, setError] = useState(null);
  const [placing, setPlacing] = useState(false);

  const shipsToPlace = ALL_SHIPS.filter((s) => !placedShipNames.includes(s.name));
  const currentShip = shipsToPlace[0];

  const handleCellClick = async (row, col) => {
    if (!currentShip || placing) return;
    setError(null);
    setPlacing(true);
    try {
      await onPlace(currentShip.name, row, col, orientation);
    } catch (err) {
      setError(err.message);
    } finally {
      setPlacing(false);
    }
  };

  if (!currentShip) {
    return <p>All ships placed. Waiting to start...</p>;
  }

  return (
    <div className="card" style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <div>
        <strong>Place your {currentShip.name}</strong> ({currentShip.size} cells)
      </div>

      <div className="ship-picker">
        <button
          className={`btn ${orientation === "horizontal" ? "" : "secondary"}`}
          onClick={() => setOrientation("horizontal")}
        >
          Horizontal
        </button>
        <button
          className={`btn ${orientation === "vertical" ? "" : "secondary"}`}
          onClick={() => setOrientation("vertical")}
        >
          Vertical
        </button>
      </div>

      <Board
        title="Your Board"
        board={board}
        variant="user"
        clickable={!placing}
        onCellClick={handleCellClick}
      />

      {error && <div className="error-text">{error}</div>}
    </div>
  );
}