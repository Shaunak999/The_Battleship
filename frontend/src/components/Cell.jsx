// Maps a backend board symbol to a visual cell.
// Symbols come straight from Board.get_board(): "~" water, "S" ship,
// "X" hit, "O" miss. "S" only ever appears on the viewer's own board —
// the backend hides the enemy's unhit ships as "~" already.
const SYMBOL_TO_STATE = {
  "~": "water",
  S: "ship",
  X: "hit",
  O: "miss",
};

/**
 * preview: null | "valid" | "invalid" — ghost-ship placement overlay,
 * used during ship placement hover/drag. Independent of the real symbol
 * underneath (a water cell can show a valid-green ghost, for instance).
 */
/**
 * preview: null | "valid" | "invalid" — ghost-ship placement overlay,
 * used during ship placement hover/drag. Independent of the real symbol
 * underneath (a water cell can show a valid-green ghost, for instance).
 * sunk: null | { orientation: "horizontal"|"vertical", position: "start"|"mid"|"end"|"single" }
 *   — when a hit cell belongs to a fully sunk ship, renders as a solid
 *   connected bar segment instead of a lone peg, so consecutive cells of
 *   the SAME ship visually fuse together while a neighbouring ship sunk
 *   right next to it still shows a clear seam (each ship only fuses
 *   internally, never across ships).
 */
export default function Cell({
  symbol,
  onClick,
  clickable = false,
  preview = null,
  sunk = null,
  onMouseEnter,
  onMouseLeave,
  onDragOver,
  onDrop,
}) {
  const state = SYMBOL_TO_STATE[symbol] ?? "water";
  const showPeg = (state === "hit" && !sunk) || state === "miss";

  const classes = ["cell", state];
  if (clickable) classes.push("clickable");
  if (preview === "valid") classes.push("preview-valid");
  if (preview === "invalid") classes.push("preview-invalid");
  if (sunk) {
    classes.push("sunk", `sunk-${sunk.orientation}`, `sunk-${sunk.position}`);
  }

  return (
    <div
      className={classes.join(" ")}
      onClick={clickable ? onClick : undefined}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
      onDragOver={onDragOver}
      onDrop={onDrop}
      role={clickable ? "button" : undefined}
      tabIndex={clickable ? 0 : undefined}
      onKeyDown={
        clickable
          ? (e) => {
              if (e.key === "Enter" || e.key === " ") onClick?.();
            }
          : undefined
      }
    >
      {showPeg && <div className="peg" />}
    </div>
  );
}