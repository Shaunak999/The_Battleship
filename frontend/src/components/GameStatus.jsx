/**
 * isMyTurn: whether the viewer currently attacking is the human at this screen.
 * aiThinking: true while waiting on POST /ai-move.
 * lastMessage: Board.attack()'s "message" field from the most recent result, if any.
 */
export default function GameStatus({ isMyTurn, aiThinking, lastMessage }) {
  const turnClass = isMyTurn ? "turn-user" : "turn-enemy";

  let headline;
  if (aiThinking) {
    headline = "AI is thinking...";
  } else if (isMyTurn) {
    headline = "Your turn — pick a cell on the enemy board";
  } else {
    headline = "Waiting for opponent...";
  }

  return (
    <div className={`status-banner ${turnClass}`}>
      <div>
        <div>{headline}</div>
        {lastMessage && (
          <div style={{ fontWeight: 400, fontSize: "0.9rem", marginTop: 4 }}>
            {lastMessage}
          </div>
        )}
      </div>
    </div>
  );
}