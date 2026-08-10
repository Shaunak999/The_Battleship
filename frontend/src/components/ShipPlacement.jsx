import { useState } from "react";
import Board from "./Board";

const ALL_SHIPS = [
  { name: "Carrier", size: 5 },
  { name: "Battleship", size: 4 },
  { name: "Cruiser", size: 3 },
  { name: "Submarine", size: 3 },
  { name: "Destroyer", size: 2 },
];

const BOARD_SIZE = 10;

/** Cells a ship of `size` occupies starting at (row, col), extending
 *  right (horizontal) or down (vertical). Matches Board.get_positions()
 *  on the backend exactly — same anchor convention. */
function getShipCells(row, col, size, orientation) {
  const cells = [];
  for (let i = 0; i < size; i++) {
    if (orientation === "horizontal") cells.push([row, col + i]);
    else cells.push([row + i, col]);
  }
  return cells;
}

function inBounds(row, col) {
  return row >= 0 && row < BOARD_SIZE && col >= 0 && col < BOARD_SIZE;
}

/** Set of "row-col" keys for every already-placed own-ship cell,
 *  read straight off the board symbols ("S" = your unhit ship). */
function buildOccupiedSet(board) {
  const occupied = new Set();
  board.forEach((rowCells, r) => {
    rowCells.forEach((symbol, c) => {
      if (symbol === "S") occupied.add(`${r}-${c}`);
    });
  });
  return occupied;
}

function isValidFootprint(cells, occupied) {
  return cells.every(([r, c]) => inBounds(r, c) && !occupied.has(`${r}-${c}`));
}

/** Client-side mirror of the backend's random-placement approach —
 *  used only to compute candidates for the "Randomize" button. The
 *  backend re-validates everything server-side regardless, so a bug
 *  here can only produce a rejected placement, never an invalid one. */
function randomPlacementFor(size, occupied) {
  for (let attempt = 0; attempt < 500; attempt++) {
    const orientation = Math.random() < 0.5 ? "horizontal" : "vertical";
    const row =
      orientation === "horizontal"
        ? Math.floor(Math.random() * BOARD_SIZE)
        : Math.floor(Math.random() * (BOARD_SIZE - size + 1));
    const col =
      orientation === "horizontal"
        ? Math.floor(Math.random() * (BOARD_SIZE - size + 1))
        : Math.floor(Math.random() * BOARD_SIZE);

    const cells = getShipCells(row, col, size, orientation);
    if (isValidFootprint(cells, occupied)) {
      return { row, col, orientation, cells };
    }
  }
  return null; // extremely unlikely on a 10x10 board with <=5 ships
}

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
  const [hoveredCell, setHoveredCell] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  const shipsToPlace = ALL_SHIPS.filter((s) => !placedShipNames.includes(s.name));
  const currentShip = shipsToPlace[0];
  const occupied = buildOccupiedSet(board);

  const previewFootprint = (() => {
    if (!hoveredCell || !currentShip) return null;
    const cells = getShipCells(hoveredCell[0], hoveredCell[1], currentShip.size, orientation);
    const valid = isValidFootprint(cells, occupied);
    const inBoundsCells = cells.filter(([r, c]) => inBounds(r, c));
    return { valid, cellSet: new Set(inBoundsCells.map(([r, c]) => `${r}-${c}`)) };
  })();

  async function commitPlacement(row, col) {
    if (!currentShip || busy) return;
    const cells = getShipCells(row, col, currentShip.size, orientation);
    if (!isValidFootprint(cells, occupied)) {
      setError("That placement is out of bounds or overlaps another ship.");
      return;
    }
    setError(null);
    setBusy(true);
    try {
      await onPlace(currentShip.name, row, col, orientation);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
      setHoveredCell(null);
    }
  }

  async function handleRandomizeAll() {
    if (busy) return;
    setError(null);
    setBusy(true);
    try {
      const localOccupied = new Set(occupied);
      const placements = [];
      for (const ship of shipsToPlace) {
        const placement = randomPlacementFor(ship.size, localOccupied);
        if (!placement) {
          setError(`Couldn't find a random spot for ${ship.name}. Try again.`);
          setBusy(false);
          return;
        }
        placement.cells.forEach(([r, c]) => localOccupied.add(`${r}-${c}`));
        placements.push({ ship, placement });
      }
      // Commit sequentially — each call is a real network request the
      // parent must await and re-render from before the next one fires.
      for (const { ship, placement } of placements) {
        await onPlace(ship.name, placement.row, placement.col, placement.orientation);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  function handleDragStart(e) {
    if (!currentShip) return;
    e.dataTransfer.setData("text/plain", currentShip.name);
    e.dataTransfer.effectAllowed = "move";
  }

  function handleCellDragOver(row, col, e) {
    e.preventDefault(); // required to allow a drop
    setHoveredCell([row, col]);
  }

  function handleCellDrop(row, col, e) {
    e.preventDefault();
    commitPlacement(row, col);
  }

  if (!currentShip) {
    return <p>All ships placed. Waiting to start...</p>;
  }

  return (
    <div className="card" style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <div>
        <strong>Place your {currentShip.name}</strong> ({currentShip.size} cells) — drag it onto
        the board, or click a cell.
      </div>

      <div className="ship-tray">
        <div
          className="ship-tray-chip"
          draggable={!busy}
          onDragStart={handleDragStart}
          title="Drag onto your board"
        >
           {currentShip.name}
        </div>

        <button
          className="btn secondary rotate-btn"
          onClick={() => setOrientation((o) => (o === "horizontal" ? "vertical" : "horizontal"))}
          disabled={busy}
        >
          ⟳ Rotate ({orientation === "horizontal" ? "Horizontal" : "Vertical"})
        </button>

        <button className="btn secondary" onClick={handleRandomizeAll} disabled={busy}>
           Randomize All Ships
        </button>
      </div>

      <Board
        title="Your Board"
        board={board}
        variant="user"
        clickable={!busy}
        onCellClick={commitPlacement}
        previewCells={previewFootprint?.cellSet ?? null}
        previewValid={previewFootprint?.valid ?? false}
        onCellHover={(r, c) => setHoveredCell([r, c])}
        onCellLeaveBoard={() => setHoveredCell(null)}
        onCellDragOver={handleCellDragOver}
        onCellDrop={handleCellDrop}
      />

      {error && <div className="error-text">{error}</div>}
    </div>
  );
}