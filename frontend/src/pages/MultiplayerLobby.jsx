import { useState } from "react";
import { mpCreateGame, mpJoinGame, mpSpectateGame } from "../services/api";

export default function MultiplayerLobby({ onExit, onJoin, initialTab = "create", initialGameId = "" }) {
  const [tab, setTab] = useState(initialTab); // create | join | spectate
  const [playerName, setPlayerName] = useState("");
  const [gameId, setGameId] = useState(initialGameId);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [shareInfo, setShareInfo] = useState(null); // { gameId, link }
  const [copied, setCopied] = useState(false);

  function handleCopyLink(link) {
    navigator.clipboard.writeText(link).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }

  async function handleCreate() {
    setError(null);
    setLoading(true);
    setShareInfo(null);
    try {
      const res = await mpCreateGame(playerName || "Player 1");
      // Build share link from the current page origin (correct LAN IP automatically)
      const link = `${window.location.origin}/?join=${res.game_id}`;
      setShareInfo({ gameId: res.game_id, link });
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
              boxSizing: "border-box",
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
                boxSizing: "border-box",
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

        {/* Share link — shown after creating a game */}
        {shareInfo && (
          <div style={{
            marginTop: 20,
            padding: "14px 16px",
            borderRadius: "var(--radius)",
            background: "var(--color-user-light)",
            border: "1px solid var(--color-user)",
          }}>
            <div style={{ fontSize: "0.8rem", fontWeight: 700, marginBottom: 6, color: "var(--color-user)" }}>
              🎮 Game created! Share this link with your friend:
            </div>
            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <code style={{
                flex: 1,
                fontSize: "0.78rem",
                wordBreak: "break-all",
                background: "rgba(0,0,0,0.08)",
                padding: "6px 8px",
                borderRadius: 4,
              }}>
                {shareInfo.link}
              </code>
              <button
                className="btn secondary"
                style={{ whiteSpace: "nowrap", padding: "6px 12px", fontSize: "0.82rem" }}
                onClick={() => handleCopyLink(shareInfo.link)}
              >
                {copied ? "✓ Copied!" : "Copy"}
              </button>
            </div>
            <div style={{ fontSize: "0.78rem", marginTop: 8, opacity: 0.7 }}>
              Or share just the code: <strong style={{ letterSpacing: "0.1em" }}>{shareInfo.gameId}</strong>
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
