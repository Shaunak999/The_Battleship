/**
 * headline: the exact text to show (computed by the caller, since what
 * counts as "your turn" differs between Human vs AI — a single fixed
 * human — and Human vs Human, where the banner must name whichever
 * player is actually up).
 */
export default function GameStatus({ headline, isMyTurn, lastMessage }) {
  const turnClass = isMyTurn ? "turn-user" : "turn-enemy";

  return (
    <div className={`status-banner ${turnClass}`}>
      <div>
        <div>{headline}</div>
        
      </div>
    </div>
  );
}