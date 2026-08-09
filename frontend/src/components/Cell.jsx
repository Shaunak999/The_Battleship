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

export default function Cell({ symbol, onClick, clickable = false }) {
  const state = SYMBOL_TO_STATE[symbol] ?? "water";
  const showPeg = state === "hit" || state === "miss";

  const classes = ["cell", state];
  if (clickable) classes.push("clickable");

  return (
    <div
      className={classes.join(" ")}
      onClick={clickable ? onClick : undefined}
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