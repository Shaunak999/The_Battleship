import { useEffect, useRef, useState } from "react";
import Board from "../components/Board";
import ShipStatus from "../components/ShipStatus";
import { createGame, getGame, aiStep, getStrategies } from "../services/api";

const DEFAULT_DELAY = 400; // ms between moves

export default function WatchAiBattle({ onExit }) {
  const [strategies, setStrategies] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Config
  const [player1Strategy, setPlayer1Strategy] = useState(null);
  const [player2Strategy, setPlayer2Strategy] = useState(null);
  const [speed, setSpeed] = useState(DEFAULT_DELAY);

  // Game state
  const [phase, setPhase] = useState("setup"); // setup | watching | done
  const [gameState, setGameState] = useState(null);
  const [lastMessage, setLastMessage] = useState(null);
  const [moveCount, setMoveCount] = useState(0);
  const [paused, setPaused] = useState(false);
  const [winner, setWinner] = useState(null);

  const cancelledRef = useRef(false);
  const timerRef = useRef(null);
  const pausedRef = useRef(false);

  useEffect(() => {
    getStrategies()
      .then((data) => setStrategies(data))
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      cancelledRef.current = true;
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  async function startWatching() {
    if (!player1Strategy || !player2Strategy) {
      setError("Please select strategies.");
      return;
    }
    setError(null);
    setPhase("watching");
    setMoveCount(0);
    setWinner(null);
    setLastMessage(null);
    cancelledRef.current = false;

    try {
      const { game_id } = await createGame({
        mode: "ai_vs_ai",
        aiStrategy: player1Strategy,
        aiStrategy2: player2Strategy,
      });

      // Fetch initial state (viewer 0 to see player 1's board as "user")
      const state = await getGame(game_id, 0);
      setGameState(state);

      // Start the auto-play loop
      playLoop(game_id);
    } catch (err) {
      setError(err.message);
      setPhase("setup");
    }
  }

  function playLoop(gameId) {
    if (cancelledRef.current) return;

    timerRef.current = setTimeout(async () => {
      if (cancelledRef.current) return;

      // If paused, just re-schedule without advancing
      if (pausedRef.current) {
        playLoop(gameId);
        return;
      }

      try {
        const result = await aiStep(gameId);
        const coord = result.move.coordinate;
        const res = result.result;
        const msg = res.message || `${res.result}`;
        setLastMessage(`${coord}: ${msg}`);
        setMoveCount((c) => c + 1);

        // Fetch both views for a full picture
        const [asP1, asP2] = await Promise.all([
          getGame(gameId, 0),
          getGame(gameId, 1),
        ]);
        setGameState({ p1: asP1, p2: asP2 });

        if (res.game_over) {
          setWinner(res.winner);
          setPhase("done");
          return;
        }

        playLoop(gameId);
      } catch (err) {
        setError(err.message);
      }
    }, speed);
  }

  function togglePause() {
    setPaused((p) => {
      pausedRef.current = !p;
      return !p;
    });
  }

  function handleBack() {
    cancelledRef.current = true;
    if (timerRef.current) clearTimeout(timerRef.current);
    onExit();
  }

  // --- SETUP PHASE ---
  if (phase === "setup") {
    if (loading) return <p>Loading strategies...</p>;

    return (
      <div className="app-shell">
        <div className="card" style={{ maxWidth: 520, width: "100%" }}>
          <h2> Computer Battle</h2>
           

          {error && <div className="error-text">{error}</div>}

          <div style={{ display: "flex", gap: 24, marginTop: 16 }}>
            {/* Player 1 */}
            <div style={{ flex: 1 }}>
              <h3 style={{ fontSize: "0.95rem", marginBottom: 8 }}>
                <span style={{ color: "var(--color-user)" }}>■</span> Player 1
              </h3>
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {strategies.map((s) => (
                  <button
                    key={s.key}
                    className={`btn secondary ${player1Strategy === s.key ? "active" : ""}`}
                    style={{
                      background:
                        player1Strategy === s.key
                          ? "var(--color-user-light)"
                          : undefined,
                      borderColor:
                        player1Strategy === s.key
                          ? "var(--color-user)"
                          : undefined,
                      color:
                        player1Strategy === s.key
                          ? "var(--color-user)"
                          : undefined,
                      textAlign: "left",
                      fontSize: "0.85rem",
                    }}
                    onClick={() => setPlayer1Strategy(s.key)}
                  >
                    {s.name}
                  </button>
                ))}
              </div>
            </div>

            {/* Player 2 */}
            <div style={{ flex: 1 }}>
              <h3 style={{ fontSize: "0.95rem", marginBottom: 8 }}>
                <span style={{ color: "var(--color-enemy)" }}>■</span> Player 2
              </h3>
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {strategies.map((s) => (
                  <button
                    key={s.key}
                    className={`btn secondary ${player2Strategy === s.key ? "active" : ""}`}
                    style={{
                      background:
                        player2Strategy === s.key
                          ? "var(--color-enemy-light)"
                          : undefined,
                      borderColor:
                        player2Strategy === s.key
                          ? "var(--color-enemy)"
                          : undefined,
                      color:
                        player2Strategy === s.key
                          ? "var(--color-enemy)"
                          : undefined,
                      textAlign: "left",
                      fontSize: "0.85rem",
                    }}
                    onClick={() => setPlayer2Strategy(s.key)}
                  >
                    {s.name}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Speed control */}
          <div style={{ marginTop: 20 }}>
            <label style={{ fontSize: "0.85rem", fontWeight: 600 }}>
              Move delay: {speed}ms
            </label>
            <input
              type="range"
              min={100}
              max={1500}
              step={50}
              value={speed}
              onChange={(e) => setSpeed(Number(e.target.value))}
              style={{ width: "100%", marginTop: 4 }}
            />
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                fontSize: "0.75rem",
                color: "var(--color-text-muted)",
              }}
            >
              
            </div>
          </div>

          <div style={{ display: "flex", gap: 12, marginTop: 20 }}>
            <button className="btn" onClick={startWatching} style={{ flex: 1 }}>
               Start Battle
            </button>
            <button className="btn secondary" onClick={handleBack}>
               Back
            </button>
          </div>
        </div>
      </div>
    );
  }

  // --- WATCHING / DONE PHASE ---
  if (!gameState) return <p>Loading game...</p>;

  const p1 = gameState.p1 || gameState;
  const p2 = gameState.p2 || gameState;

  return (
    <div className="app-shell">
      {/* Status bar */}
      <div
        className="status-banner"
        style={{
          borderLeftColor:
            phase === "done"
              ? "var(--color-success)"
              : "var(--color-user)",
          justifyContent: "space-between",
          width: "100%",
          maxWidth: 1000,
        }}
      >
        <div>
          {phase === "done" ? (
            <span>
              <strong>{winner}</strong> wins!
            </span>
          ) : (
            <span>
              {lastMessage || "Starting..."}{" "}
              <span style={{ color: "var(--color-text-muted)", fontSize: "0.85rem" }}>
                (Move #{moveCount})
              </span>
            </span>
          )}
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          {phase === "watching" && (
            <button
              className="btn secondary"
              onClick={togglePause}
              style={{ padding: "6px 14px", fontSize: "0.85rem" }}
            >
              {paused ? " Resume" : " Pause"}
            </button>
          )}
          <button
            className="btn secondary"
            onClick={handleBack}
            style={{ padding: "6px 14px", fontSize: "0.85rem" }}
          >
             Back
          </button>
        </div>
      </div>

      {error && <div className="error-text">{error}</div>}

      <div className="boards-row">
        {/* Player 1 board — show their own fleet + opponent attacks */}
        <div>
          <Board
            title={p1.your_player.name}
            board={p1.your_player.board}
            variant="user"
            sunkShips={p1.your_player.sunk_ships}
          />
          <div style={{ marginTop: 12 }}>
            <ShipStatus
              title={`${p1.your_player.name}'s Fleet`}
              remainingShips={p1.your_player.remaining_ships}
            />
          </div>
        </div>

        {/* Player 2 board — show their own fleet + opponent attacks */}
        <div>
          <Board
            title={p2.your_player.name}
            board={p2.your_player.board}
            variant="user"
            sunkShips={p2.your_player.sunk_ships}
          />
          <div style={{ marginTop: 12 }}>
            <ShipStatus
              title={`${p2.your_player.name}'s Fleet`}
              remainingShips={p2.your_player.remaining_ships}
            />
          </div>
        </div>
      </div>

      {phase === "done" && (
        <div style={{ display: "flex", gap: 12, marginTop: 8 }}>
          <button className="btn" onClick={startWatching}>
             Watch Again
          </button>
          <button className="btn secondary" onClick={handleBack}>
             Back to Menu
          </button>
        </div>
      )}
    </div>
  );
}
