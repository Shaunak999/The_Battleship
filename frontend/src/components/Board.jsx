import { Fragment } from "react";
import Cell from "./Cell";

const COL_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"];

/**
 * Renders a 10x10 board from the backend's board array (10 rows of
 * 10 symbol strings each: "~" / "S" / "X" / "O").
 *
 * variant: "user" | "enemy" — controls the blue/red theming.
 * onCellClick(row, col) — only wired up when clickable is true.
 */
export default function Board({
  title,
  board,
  variant = "user",
  clickable = false,
  onCellClick,
}) {
  return (
    <div className={`board-wrapper ${variant}`}>
      <div className="board-title">
        <span className="dot" />
        {title}
      </div>

      <div className="board-grid">
        <div className="board-corner" />
        {COL_LABELS.map((label) => (
          <div className="board-col-label" key={label}>
            {label}
          </div>
        ))}

        {board.map((rowCells, rowIndex) => (
          <Fragment key={`row-${rowIndex}`}>
            <div className="board-row-label">{rowIndex + 1}</div>
            {rowCells.map((symbol, colIndex) => (
              <Cell
                key={`${rowIndex}-${colIndex}`}
                symbol={symbol}
                clickable={clickable}
                onClick={() => onCellClick?.(rowIndex, colIndex)}
              />
            ))}
          </Fragment>
        ))}
      </div>
    </div>
  );
}