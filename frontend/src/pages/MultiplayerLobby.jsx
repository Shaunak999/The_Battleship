import { useState } from "react";
import { mpCreateGame, mpJoinGame, mpSpectateGame } from "../services/api";

export default function MultiplayerLobby({ onExit, onJoin }) {
  const [tab, setTab] = useState("create"); // create | join | spectate
  const [playerName, setPlayerName] = useState("");
  const [gameId, setGameId] = useState("");
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  async function handleCreate() {
    setError(null);
    setLoading(true);
    try {
      const res = await mpCreateGame(playerName || "Player 1");
      onJoin({ gameId: res.game_id, role: "player1", playerName: res.player_name });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleJoin() {
    if (!gameId.trim()) {
      setError("Enter a Game ID.");
      return;
    }
    setError(null);
    setLoading(true);
    try {
      const res = await mpJoinGame(gameId.trim(), playerName || "Player 2");
      onJoin({ gameId: res.game_id, role: "player2", playerName: res.player_name });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleSpectate() {
    if (!gameId.trim()) {
      setError("Enter a Game ID.");
      return;
    }
    setError(null);
    setLoading(true);
    try {
      await mpSpectateGame(gameId.trim());
      onJoin({ gameId: gameId.trim(), role: "spectator", playerName: "Spectator" });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <div className="card" style={{ maxWidth: 500, width: "100%" }}>
        <h2 style={{ marginBottom: 16 }}>Human vs Human</h2>

        {/* Tab buttons */}
        <div style={{ display: "flex", gap: 8, marginBottom: 20 }}>
          {[
            { key: "create", label: "Create Game" },
            { key: "join", label: "Join Game" },
            { key: "spectate", label: "Spectate" },
          ].map((t) => (
            <button
              key={t.key}
              className={`btn secondary ${tab === t.key ? "" : ""}`}
              style={{
                flex: 1,
                background: tab === t.key ? "var(--color-user-light)" : undefined,
                borderColor: tab === t.key ? "var(--color-user)" : undefined,
                color: tab === t.key ? "var(--color-user)" : undefined,
              }}
              onClick={() => { setTab(t.key); setError(null); }}
            >
              {t.label}
            </button>
          ))}
        </div>

        {/* Player name */}
        <div style={{ marginBottom: 16 }}>
          <label style={{ display: "block", fontSize: "0.85rem", fontWeight: 600, marginBottom: 4 }}>
            Your Name
          </label>
          <input
            type="text"
            value={playerName}
            onChange={(e) => setPlayerName(e.target.value)}
            placeholder={tab === "create" ? "Player 1" : tab === "join" ? "Player 2" : "Spectator"}
            style={{
              width: "100%",
              padding: "10px 12px",
              borderRadius: "var(--radius)",
              border: "1px solid var(--color-water-border)",
              fontSize: "1rem",
              outline: "none",
            }}
          />
        </div>

        {/* Game ID input (for join/spectate) */}
        {(tab === "join" || tab === "spectate") && (
          <div style={{ marginBottom: 16 }}>
            <label style={{ display: "block", fontSize: "0.85rem", fontWeight: 600, marginBottom: 4 }}>
              Game ID
            </label>
            <input
              type="text"
              value={gameId}
              onChange={(e) => setGameId(e.target.value.toUpperCase())}
              placeholder="e.g. ABC123"
              maxLength={6}
              style={{
                width: "100%",
                padding: "10px 12px",
                borderRadius: "var(--radius)",
                border: "1px solid var(--color-water-border)",
                fontSize: "1rem",
                fontFamily: "monospace",
                letterSpacing: "0.1em",
                outline: "none",
                textTransform: "uppercase",
              }}
            />
          </div>
        )}

        {error && <div className="error-text">{error}</div>}

        {/* Action buttons */}
        <div style={{ display: "flex", gap: 12, marginTop: 16 }}>
          {tab === "create" && (
            <button className="btn" style={{ flex: 1 }} onClick={handleCreate} disabled={loading}>
              {loading ? "Creating..." : "Create Game"}
            </button>
          )}
          {tab === "join" && (
            <button className="btn" style={{ flex: 1 }} onClick={handleJoin} disabled={loading}>
              {loading ? "Joining..." : "Join Game"}
            </button>
          )}
          {tab === "spectate" && (
            <button className="btn" style={{ flex: 1 }} onClick={handleSpectate} disabled={loading}>
              {loading ? "Connecting..." : "Spectate"}
            </button>
          )}
          <button className="btn secondary" onClick={onExit}>
            Back
          </button>
        </div>


      </div>
    </div>
  );
}
