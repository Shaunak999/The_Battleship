import { Fragment } from "react";
import Cell from "./Cell";

const COL_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"];

/** Builds a lookup of "row-col" -> { orientation, position } for every
 * cell belonging to a sunk ship, from the backend's ordered cell list
 * for each sunk ship. Order is guaranteed contiguous start-to-end
 * (Ship.positions is built that way by Board.get_positions()). */
function buildSunkMap(sunkShips) {
  const map = new Map();
  (sunkShips || []).forEach((ship) => {
    const cells = ship.cells;
    const orientation =
      cells.length > 1 && cells[0][0] === cells[1][0] ? "horizontal" : "vertical";
    cells.forEach(([r, c], i) => {
      let position;
      if (cells.length === 1) position = "single";
      else if (i === 0) position = "start";
      else if (i === cells.length - 1) position = "end";
      else position = "mid";
      map.set(`${r}-${c}`, { orientation, position });
    });
  });
  return map;
}

/**
 * variant: "user" | "enemy" — controls the blue/red theming.
 * onCellClick(row, col) — only wired up when clickable is true.
 * sunkShips — this side's sunk_ships array from GameState, used to draw
 *   fused bar segments for cells belonging to the same sunk ship.
 * inactive — greys the whole board out and disables interaction, to show
 *   "it's not your turn to attack this board right now" without a
 *   separate pass-device screen.
 * previewCells / previewValid / onCellHover / onCellLeaveBoard /
 *   onCellDragOver / onCellDrop — placement-only ghost preview and drag support.
 */
export default function Board({
  title,
  board,
  variant = "user",
  clickable = false,
  inactive = false,
  onCellClick,
  sunkShips = null,
  previewCells = null,
  previewValid = false,
  onCellHover,
  onCellLeaveBoard,
  onCellDragOver,
  onCellDrop,
}) {
  const sunkMap = buildSunkMap(sunkShips);

  return (
    <div className={`board-wrapper ${variant}${inactive ? " inactive" : ""}`}>
      <div className="board-title">
        <span className="dot" />
        {title}
        {inactive && <span className="inactive-label">Not your turn</span>}
      </div>

      <div className="board-grid" onMouseLeave={onCellLeaveBoard}>
        <div className="board-corner" />
        {COL_LABELS.map((label) => (
          <div className="board-col-label" key={label}>
            {label}
          </div>
        ))}

        {board.map((rowCells, rowIndex) => (
          <Fragment key={`row-${rowIndex}`}>
            <div className="board-row-label">{rowIndex + 1}</div>
            {rowCells.map((symbol, colIndex) => {
              const key = `${rowIndex}-${colIndex}`;
              const isPreview = previewCells?.has(key);
              return (
                <Cell
                  key={key}
                  symbol={symbol}
                  clickable={clickable}
                  onClick={() => onCellClick?.(rowIndex, colIndex)}
                  preview={isPreview ? (previewValid ? "valid" : "invalid") : null}
                  sunk={sunkMap.get(key) ?? null}
                  onMouseEnter={onCellHover ? () => onCellHover(rowIndex, colIndex) : undefined}
                  onDragOver={
                    onCellDragOver
                      ? (e) => onCellDragOver(rowIndex, colIndex, e)
                      : undefined
                  }
                  onDrop={onCellDrop ? (e) => onCellDrop(rowIndex, colIndex, e) : undefined}
                />
              );
            })}
          </Fragment>
        ))}
      </div>
    </div>
  );
}