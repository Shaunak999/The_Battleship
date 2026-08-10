function countSymbols(board) {
  let hits = 0;
  let misses = 0;
  board.forEach((row) =>
    row.forEach((symbol) => {
      if (symbol === "X") hits += 1;
      else if (symbol === "O") misses += 1;
    })
  );
  return { hits, misses, total: hits + misses };
}

/**
 * Derives per-player shot stats from a single viewer=0 GameState.
 *
 * Board.attacks tracks shots landed AGAINST that board, so:
 * - shots FIRED BY player 0 show up as X/O marks on opponent_player.board
 * - shots FIRED BY player 1 show up as X/O marks on your_player.board
 * This works for Human vs AI too — "player 1" is just whichever name
 * the backend gave the AI.
 */
function computeStats(state) {
  const p0 = countSymbols(state.opponent_player.board);
  const p1 = countSymbols(state.your_player.board);

  const withAccuracy = (s) => ({
    ...s,
    accuracy: s.total > 0 ? Math.round((s.hits / s.total) * 100) : 0,
  });

  return {
    player0: { name: state.your_player.name, ...withAccuracy(p0) },
    player1: { name: state.opponent_player.name, ...withAccuracy(p1) },
  };
}

export default function GameResults({ state, onExit }) {
  const { player0, player1 } = computeStats(state);
  const totalShots = player0.total + player1.total;

  const rows = [
    ["Total shots", player0.total, player1.total],
    ["Hits", player0.hits, player1.hits],
    ["Misses", player0.misses, player1.misses],
   
  ];

  return (
    <div className="card" style={{ textAlign: "center", maxWidth: 480 }}>
      <h2>{state.winner ? `${state.winner} wins!` : "Game over"}</h2>
      <p style={{ color: "var(--color-text-muted)" }}>{totalShots} total shots fired</p>

      <table className="results-table">
        <thead>
          <tr>
            <th></th>
            <th style={{ color: "var(--color-user)" }}>{player0.name}</th>
            <th style={{ color: "var(--color-enemy)" }}>{player1.name}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([label, a, b]) => (
            <tr key={label}>
              <td className="results-label">{label}</td>
              <td>{a}</td>
              <td>{b}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <button className="btn" style={{ marginTop: 16 }} onClick={onExit}>
        Back to Home
      </button>
    </div>
  );
}