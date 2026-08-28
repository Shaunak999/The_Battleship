import { useEffect, useRef, useState, useCallback } from "react";
import Board from "../components/Board";
import ShipStatus from "../components/ShipStatus";
import GameStatus from "../components/GameStatus";
import ShipPlacement from "../components/ShipPlacement";
import { getWsUrl, toCoordinate } from "../services/api";

const PHASES = {
  CONNECTING: "connecting",
  WAITING_FOR_OPPONENT: "waiting_for_opponent",
  PLACEMENT: "placement",
  WAITING_FOR_READY: "waiting_for_ready",
  BATTLE: "battle",
  RESULTS: "results",
};

const ALL_SHIPS = [
  { name: "Carrier", size: 5 },
  { name: "Battleship", size: 4 },
  { name: "Cruiser", size: 3 },
  { name: "Submarine", size: 3 },
  { name: "Destroyer", size: 2 },
];

export default function MultiplayerGame({ gameId, role, playerName, onExit }) {
  const [phase, setPhase] = useState(PHASES.CONNECTING);
  const [gameState, setGameState] = useState(null);
  const [error, setError] = useState(null);
  const [lastMessage, setLastMessage] = useState(null);
  const [opponentConnected, setOpponentConnected] = useState(false);
  const [player1Ready, setPlayer1Ready] = useState(false);
  const [player2Ready, setPlayer2Ready] = useState(false);
  const [winner, setWinner] = useState(null);
  const [myShipsPlaced, setMyShipsPlaced] = useState(false);

  const wsRef = useRef(null);
  const playerIndex = role === "player1" ? 0 : 1;

  const connectWs = useCallback(() => {
    // Include player_name as query param so backend can name the player
    const url = getWsUrl(gameId, role) + `?player_name=${encodeURIComponent(playerName)}`;
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setPhase(PHASES.WAITING_FOR_OPPONENT);
    };

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      handleMessage(msg);
    };

    ws.onclose = (event) => {
      // Ignore close events from stale/unmounted socket instances
      if (wsRef.current !== ws) return;
      if (event.wasClean || event.code === 1000) return;
      setError("Connection lost. Please refresh.");
    };

    ws.onerror = () => {
      if (wsRef.current !== ws) return;
      setError("WebSocket connection failed.");
    };
  }, [gameId, role, playerName]);

  useEffect(() => {
    connectWs();
    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, [connectWs]);

  function send(msg) {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(msg));
    }
  }

  function handleMessage(msg) {
    switch (msg.type) {
      case "welcome":
        break;

      case "player_joined":
        setOpponentConnected(msg.player1_connected && msg.player2_connected);
        if (msg.player1_connected && msg.player2_connected) {
          setLastMessage("Both players connected!");
        }
        break;

      case "player_disconnected":
        if (msg.role === "player1" || msg.role === "player2") {
          setLastMessage("Opponent disconnected.");
          setOpponentConnected(false);
        }
        break;

      case "state_update":
        setGameState(msg.state);
        // When we first receive game state, transition to placement phase
        if (msg.state && msg.state.status !== "over" && !myShipsPlaced) {
          setPhase(PHASES.PLACEMENT);
        }
        break;

      case "ship_placed":
        setGameState(msg.state);
        if (msg.all_placed) {
          setMyShipsPlaced(true);
          // Send explicit ready
          send({ type: "ready" });
        }
        break;

      case "player_ready":
        setPlayer1Ready(msg.player1_ready);
        setPlayer2Ready(msg.player2_ready);
        break;

      case "game_started":
        setPhase(PHASES.BATTLE);
        setLastMessage(`Game started! ${msg.current_player_name}'s turn`);
        break;

      case "attack_result": {
        const coord = msg.coordinate;
        const resultMsg = msg.result === "sunk"
          ? `${msg.attacker_name} → ${coord} → SUNK ${msg.ship_name}!`
          : `${msg.attacker_name} → ${coord} → ${msg.result.toUpperCase()}`;
        setLastMessage(resultMsg);

        if (msg.game_over) {
          setWinner(msg.winner);
          setPhase(PHASES.RESULTS);
        }
        break;
      }

      case "spectator_state":
        // Players shouldn't receive this, but ignore gracefully
        break;

      case "error":
        setError(msg.message);
        setTimeout(() => setError(null), 4000);
        break;

      default:
        break;
    }
  }

  async function handlePlace(shipName, row, col, orientation) {
    send({
      type: "place_ship",
      ship_name: shipName,
      coordinate: toCoordinate(row, col),
      orientation,
    });
  }

  function handleAttack(row, col) {
    send({ type: "attack", row, col });
  }

  // ── RENDER ────────────────────────────────────────────────────────

  if (phase === PHASES.CONNECTING) {
    return (
      <div className="app-shell">
        <div className="card" style={{ textAlign: "center", maxWidth: 420 }}>
          <p>Connecting to game...</p>
          {error && <div className="error-text">{error}</div>}
        </div>
      </div>
    );
  }

  if (phase === PHASES.WAITING_FOR_OPPONENT && !gameState) {
    return (
      <div className="app-shell">
        <div className="card" style={{ textAlign: "center", maxWidth: 440 }}>
          <p style={{ color: "var(--color-text-muted)", marginTop: 8 }}>
            You are <strong>{playerName}</strong> ({role === "player1" ? "Player 1" : "Player 2"})
          </p>
          <p style={{ marginTop: 12 }}>
            {opponentConnected
              ? "Opponent connected! Setting up..."
              : "Waiting for opponent to join..."}
          </p>
          <div style={{
            marginTop: 16,
            padding: 12,
            background: "var(--color-water)",
            borderRadius: 8,
            fontFamily: "monospace",
            fontSize: "1.1rem",
            letterSpacing: "0.15em",
            fontWeight: 700,
          }}>
            Game ID: {gameId}
          </div>
          <p style={{ marginTop: 12, fontSize: "0.82rem", color: "var(--color-text-muted)" }}>
            Share this Game ID with the other player
          </p>
          {error && <div className="error-text">{error}</div>}
        </div>
      </div>
    );
  }

  if (phase === PHASES.PLACEMENT && gameState && !myShipsPlaced) {
    const placedShipNames = gameState.your_player
      ? gameState.your_player.remaining_ships || []
      : [];

    return (
      <div className="app-shell">
        <h2>{playerName}, place your ships</h2>
        <ShipPlacement
          board={gameState?.your_player?.board || Array.from({ length: 10 }, () => Array(10).fill("~"))}
          placedShipNames={placedShipNames}
          onPlace={handlePlace}
        />
        <div style={{ marginTop: 12, fontSize: "0.85rem", color: "var(--color-text-muted)" }}>
          Player 1: {player1Ready ? "✓ Ready" : "Placing..."} | Player 2: {player2Ready ? "✓ Ready" : "Placing..."}
        </div>
        {error && <div className="error-text">{error}</div>}
      </div>
    );
  }

  if (myShipsPlaced && phase !== PHASES.BATTLE && phase !== PHASES.RESULTS) {
    // Waiting for opponent to finish placing ships
    return (
      <div className="app-shell">
        <div className="card" style={{ textAlign: "center", maxWidth: 440 }}>
          <h2>Ships Placed!</h2>
          <p style={{ color: "var(--color-text-muted)" }}>
            Waiting for opponent to finish placing ships...
          </p>
          <div style={{ marginTop: 12, fontSize: "0.85rem", color: "var(--color-text-muted)" }}>
            Player 1: {player1Ready ? "✓ Ready" : "Placing..."} | Player 2: {player2Ready ? "✓ Ready" : "Placing..."}
          </div>
          {error && <div className="error-text">{error}</div>}
        </div>
      </div>
    );
  }

  if (phase === PHASES.RESULTS) {
    return (
      <div className="app-shell">
        <div className="card" style={{ textAlign: "center", maxWidth: 480 }}>
          <h2>{winner ? `${winner} wins!` : "Game over"}</h2>
        </div>

        {gameState && (
          <div className="boards-row">
            <div>
              <Board
                title={gameState.your_player.name}
                board={gameState.your_player.board}
                variant="user"
                sunkShips={gameState.your_player.sunk_ships}
              />
              <div style={{ marginTop: 12 }}>
                <ShipStatus
                  title={`${gameState.your_player.name}'s Fleet`}
                  remainingShips={gameState.your_player.remaining_ships}
                />
              </div>
            </div>
            <div>
              <Board
                title={gameState.opponent_player.name}
                board={gameState.opponent_player.board}
                variant="enemy"
                sunkShips={gameState.opponent_player.sunk_ships}
              />
              <div style={{ marginTop: 12 }}>
                <ShipStatus
                  title={`${gameState.opponent_player.name}'s Fleet`}
                  remainingShips={gameState.opponent_player.remaining_ships}
                />
              </div>
            </div>
          </div>
        )}

        <button className="btn" style={{ marginTop: 16 }} onClick={onExit}>
          Back to Home
        </button>
      </div>
    );
  }

  // ── BATTLE PHASE ──────────────────────────────────────────────────
  if (!gameState) return <p>Loading game state...</p>;

  const isMyTurn = gameState.current_player === playerIndex;
  const myName = gameState.your_player.name;
  const oppName = gameState.opponent_player.name;

  const statusHeadline = isMyTurn
    ? `Your turn — attack ${oppName}'s board`
    : `Waiting for ${gameState.current_player_name}...`;

  return (
    <div className="app-shell">
      <GameStatus
        headline={statusHeadline}
        isMyTurn={isMyTurn}
        lastMessage={lastMessage}
      />

      {error && <div className="error-text">{error}</div>}

      <div className="boards-row">
        {/* Left: Your board — show your own ships */}
        <div>
          <Board
            title={`${myName} (You)`}
            board={gameState.your_player.board}
            variant="user"
            sunkShips={gameState.your_player.sunk_ships}
          />
          <div style={{ marginTop: 12 }}>
            <ShipStatus
              title={`${myName}'s Fleet`}
              remainingShips={gameState.your_player.remaining_ships}
            />
          </div>
        </div>

        {/* Right: Opponent board (attack target) */}
        <div>
          <Board
            title={oppName}
            board={gameState.opponent_player.board}
            variant="enemy"
            clickable={isMyTurn}
            inactive={!isMyTurn}
            onCellClick={isMyTurn ? handleAttack : undefined}
            sunkShips={gameState.opponent_player.sunk_ships}
          />
          <div style={{ marginTop: 12 }}>
            <ShipStatus
              title={`${oppName}'s Fleet`}
              remainingShips={gameState.opponent_player.remaining_ships}
            />
          </div>
        </div>
      </div>

      <button className="btn secondary" style={{ marginTop: 12 }} onClick={onExit}>
        Leave Game
      </button>
    </div>
  );
}
