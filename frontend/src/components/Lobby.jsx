/**
 * Lobby — shown after a Human vs Human game is created.
 * Gives each player a button to open their own dedicated tab.
 * Both tabs connect to the same game via ?game=<id>&viewer=<0|1>.
 */
export default function Lobby({ gameId, onPlayHere }) {
  const base = window.location.origin + window.location.pathname;

  function tabUrl(viewerIndex) {
    return `${base}?game=${gameId}&viewer=${viewerIndex}`;
  }

  function openTab(viewerIndex) {
    window.open(tabUrl(viewerIndex), `_battleship_p${viewerIndex}`);
  }

  return (
    <div className="lobby-overlay">
      <div className="lobby-card">
        <div className="lobby-icon">⚓</div>
        <h2>Game Ready!</h2>
        <p className="lobby-subtitle">
          Each player opens their own tab so their fleet stays private.
        </p>

        <div className="lobby-players">
          <div className="lobby-player-card p1">
            <div className="lobby-player-label">Player 1</div>
            <div className="lobby-player-desc">Opens in a new tab</div>
            <button
              className="btn lobby-btn"
              onClick={() => openTab(0)}
              id="open-player1-tab"
            >
              🚢 Open Player 1 Tab
            </button>
          </div>

          <div className="lobby-divider">VS</div>

          <div className="lobby-player-card p2">
            <div className="lobby-player-label">Player 2</div>
            <div className="lobby-player-desc">Opens in a new tab</div>
            <button
              className="btn lobby-btn p2-btn"
              onClick={() => openTab(1)}
              id="open-player2-tab"
            >
              🚢 Open Player 2 Tab
            </button>
          </div>
        </div>

        <div className="lobby-hint">
          <span>💡</span>
          <span>
            Both players should open their tab before starting. Ships are
            hidden from the opponent — each player only sees their own fleet.
          </span>
        </div>

        <button
          className="btn secondary lobby-stay-btn"
          onClick={() => onPlayHere(0)}
          id="play-here-p1"
        >
          Play as Player 1 in this tab
        </button>
      </div>
    </div>
  );
}
