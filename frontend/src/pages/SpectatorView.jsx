import { useEffect, useRef, useState, useCallback } from "react";
import Board from "../components/Board";
import ShipStatus from "../components/ShipStatus";
import { getWsUrl } from "../services/api";

const ALL_SHIPS = [
  { name: "Carrier", size: 5 },
  { name: "Battleship", size: 4 },
  { name: "Cruiser", size: 3 },
  { name: "Submarine", size: 3 },
  { name: "Destroyer", size: 2 },
];

function formatTime(ts) {
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function EventLog({ events }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [events.length]);

  return (
    <div style={{
      maxHeight: 240,
      overflowY: "auto",
      padding: 12,
      background: "var(--color-water)",
      borderRadius: 8,
      fontSize: "0.82rem",
      lineHeight: 1.6,
    }}>
      <div style={{ fontWeight: 700, marginBottom: 8 }}>Live Event Log</div>
      {events.length === 0 && (
        <div style={{ color: "var(--color-text-muted)" }}>No events yet...</div>
      )}
      {events.map((ev, i) => {
        const resultColor =
          ev.result === "hit" ? "var(--color-hit)" :
          ev.result === "sunk" ? "#7c3aed" :
          "var(--color-text-muted)";
        return (
          <div key={ev.id || i} style={{ borderBottom: "1px solid var(--color-water-border)", paddingBottom: 4, marginBottom: 4 }}>
            <span style={{ color: "var(--color-text-muted)" }}>{formatTime(ev.time)}</span>
            {" "}
            <span style={{ fontWeight: 600 }}>{ev.attacker}</span>
            {" -> "}
            <span style={{ fontFamily: "monospace" }}>{ev.coordinate}</span>
            {" -> "}
            <span style={{ color: resultColor, fontWeight: 600 }}>
              {ev.result === "sunk" ? `SUNK ${ev.ship_name}` : ev.result.toUpperCase()}
            </span>
          </div>
        );
      })}
      <div ref={bottomRef} />
    </div>
  );
}

function StatsPanel({ stats }) {
  if (!stats || !stats.players) return null;

  const p1 = stats.players[0];
  const p2 = stats.players[1];
  if (!p1 || !p2) return null;

  const rows = [
    ["Shots", p1.shots, p2.shots],
    ["Hits", p1.hits, p2.hits],
    ["Misses", p1.misses, p2.misses],
    ["Accuracy", `${p1.accuracy}%`, `${p2.accuracy}%`],
    ["Ships Left", p1.ships_remaining, p2.ships_remaining],
  ];

  return (
    <div style={{
      padding: 12,
      background: "var(--color-water)",
      borderRadius: 8,
      fontSize: "0.82rem",
    }}>
      <div style={{ fontWeight: 700, marginBottom: 8 }}>Statistics</div>
      <table style={{ width: "100%", borderCollapse: "collapse" }}>
        <thead>
          <tr>
            <th style={{ textAlign: "left", padding: "4px 8px" }}></th>
            <th style={{ textAlign: "center", padding: "4px 8px", color: "var(--color-user)" }}>{p1.name}</th>
            <th style={{ textAlign: "center", padding: "4px 8px", color: "var(--color-enemy)" }}>{p2.name}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([label, v1, v2]) => (
            <tr key={label}>
              <td style={{ padding: "4px 8px", color: "var(--color-text-muted)", fontWeight: 600 }}>{label}</td>
              <td style={{ textAlign: "center", padding: "4px 8px" }}>{v1}</td>
              <td style={{ textAlign: "center", padding: "4px 8px" }}>{v2}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div style={{ marginTop: 8, color: "var(--color-text-muted)", fontSize: "0.78rem" }}>
        Duration: {Math.floor(stats.game_duration / 60)}m {Math.floor(stats.game_duration % 60)}s
      </div>
    </div>
  );
}

function FleetStatus({ ships, title }) {
  return (
    <div style={{ padding: 12, background: "var(--color-water)", borderRadius: 8 }}>
      <div style={{ fontWeight: 700, marginBottom: 8, fontSize: "0.85rem" }}>{title}</div>
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        {ALL_SHIPS.map((def) => {
          const ship = ships.find((s) => s.name === def.name);
          const sunk = ship ? ship.sunk : false;
          return (
            <div
              key={def.name}
              style={{
                display: "flex",
                justifyContent: "space-between",
                padding: "4px 8px",
                borderRadius: 4,
                fontSize: "0.82rem",
                textDecoration: sunk ? "line-through" : "none",
                color: sunk ? "var(--color-text-muted)" : "var(--color-text)",
                background: sunk ? "transparent" : "var(--color-surface)",
              }}
            >
              <span>{def.name}</span>
              <span style={{ fontFamily: "monospace" }}>
                {sunk ? "SUNK" : `${def.size} cells`}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default function SpectatorView({ gameId, onExit }) {
  const [spectatorState, setSpectatorState] = useState(null);
  const [error, setError] = useState(null);
  const [lastMessage, setLastMessage] = useState(null);
  const [winner, setWinner] = useState(null);
  const wsRef = useRef(null);

  const connectWs = useCallback(() => {
    const url = getWsUrl(gameId, "spectator");
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setLastMessage("Connected as spectator.");
    };

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      handleMessage(msg);
    };

    ws.onclose = (event) => {
      if (wsRef.current !== ws) return;
      if (event.wasClean || event.code === 1000) return;
      if (!winner) {
        setError("Connection lost.");
      }
    };

    ws.onerror = () => {
      if (wsRef.current !== ws) return;
      setError("WebSocket connection failed.");
    };
  }, [gameId, winner]);

  useEffect(() => {
    connectWs();
    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, [connectWs]);

  function handleMessage(msg) {
    switch (msg.type) {
      case "welcome":
        break;

      case "spectator_state":
        setSpectatorState(msg.state);
        if (msg.state.status === "over" && msg.state.winner) {
          setWinner(msg.state.winner);
        }
        break;

      case "player_joined":
        setLastMessage(`${msg.role} joined`);
        break;

      case "player_disconnected":
        setLastMessage(`${msg.role} disconnected`);
        break;

      case "player_ready":
        setLastMessage("Player ready");
        break;

      case "game_started":
        setLastMessage("Game started!");
        break;

      case "attack_result": {
        const coord = msg.coordinate;
        const resultMsg = msg.result === "sunk"
          ? `${msg.attacker_name} -> ${coord} -> SUNK ${msg.ship_name}!`
          : `${msg.attacker_name} -> ${coord} -> ${msg.result.toUpperCase()}`;
        setLastMessage(resultMsg);
        break;
      }

      case "error":
        setError(msg.message);
        setTimeout(() => setError(null), 3000);
        break;

      default:
        break;
    }
  }

  // ── RENDER ──────────────────────────────────────────────────────

  if (error && !spectatorState) {
    return (
      <div className="card" style={{ textAlign: "center", maxWidth: 420 }}>
        <p className="error-text">{error}</p>
        <button className="btn" onClick={onExit}>Back</button>
      </div>
    );
  }

  if (!spectatorState || !spectatorState.player1) {
    return (
      <div className="card" style={{ textAlign: "center", maxWidth: 420 }}>
        <h2>Spectating: {gameId}</h2>
        <p>Waiting for game data...</p>
        <div className="error-text">{error}</div>
      </div>
    );
  }

  const state = spectatorState;
  const isGameOver = state.status === "over";
  const currentTurn = state.current_player_name;

  return (
    <div className="app-shell">
      {/* Header */}
      <div style={{
        width: "100%",
        maxWidth: 1000,
        padding: "12px 20px",
        background: "linear-gradient(135deg, #1e293b, #334155)",
        borderRadius: "var(--radius)",
        color: "white",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
      }}>
        <div>
          <div style={{ fontSize: "1.1rem", fontWeight: 700 }}>SPECTATOR MODE</div>
          <div style={{ fontSize: "0.8rem", opacity: 0.7 }}>Game: {gameId}</div>
        </div>
        <div style={{ textAlign: "center" }}>
          {isGameOver ? (
            <div style={{ fontSize: "1.1rem", fontWeight: 700, color: "#4ade80" }}>
              {winner} WINS!
            </div>
          ) : (
            <div style={{ fontSize: "0.9rem" }}>
              Current Turn: <strong>{currentTurn}</strong>
            </div>
          )}
        </div>
        <button
          className="btn secondary"
          style={{ fontSize: "0.8rem", padding: "6px 14px" }}
          onClick={onExit}
        >
          Leave
        </button>
      </div>

      {error && <div className="error-text">{error}</div>}

      {/* Boards */}
      <div className="boards-row">
        <div>
          <Board
            title={`${state.player1.name} (P1)`}
            board={state.player1.board}
            variant="user"
            sunkShips={state.player1.sunk_ships}
          />
          <div style={{ marginTop: 12 }}>
            <FleetStatus ships={state.player1.ships} title={`${state.player1.name}'s Fleet`} />
          </div>
        </div>
        <div>
          <Board
            title={`${state.player2.name} (P2)`}
            board={state.player2.board}
            variant="enemy"
            sunkShips={state.player2.sunk_ships}
          />
          <div style={{ marginTop: 12 }}>
            <FleetStatus ships={state.player2.ships} title={`${state.player2.name}'s Fleet`} />
          </div>
        </div>
      </div>

      {/* Event log + Statistics */}
      <div style={{ display: "flex", gap: 16, width: "100%", maxWidth: 1000, flexWrap: "wrap" }}>
        <div style={{ flex: 1, minWidth: 300 }}>
          <EventLog events={state.event_log || []} />
        </div>
        <div style={{ flex: 1, minWidth: 280 }}>
          <StatsPanel stats={state.statistics} />
        </div>
      </div>
    </div>
  );
}
