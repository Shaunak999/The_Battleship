import { useEffect, useState } from "react";
import Board from "../components/Board";
import ShipStatus from "../components/ShipStatus";
import GameStatus from "../components/GameStatus";
import StrategySelector from "../components/StrategySelector";
import ShipPlacement from "../components/ShipPlacement";
import GameResults from "../components/GameResults";
import { createGame, placeShip, attack, aiMove, getGame } from "../services/api";

const PHASES = {
  CHOOSE_STRATEGY: "choose_strategy",
  PLACEMENT: "placement",
  PASS_DEVICE: "pass_device",
  BATTLE: "battle",
  REVEAL: "reveal",
  RESULTS: "results",
};

export default function Game({ mode, onExit }) {
  const [phase, setPhase] = useState(
    mode === "human_vs_ai" ? PHASES.CHOOSE_STRATEGY : PHASES.PLACEMENT
  );
  const [gameId, setGameId] = useState(null);
  const [viewerIndex, setViewerIndex] = useState(0);
  const [gameState, setGameState] = useState(null);
  const [aiThinking, setAiThinking] = useState(false);
  const [lastMessage, setLastMessage] = useState(null);
  const [error, setError] = useState(null);
  const [pendingViewerAfterPass, setPendingViewerAfterPass] = useState(0);
  const [finalState, setFinalState] = useState(null);
  const [revealData, setRevealData] = useState(null);

  // HvH: per-player fleet visibility. Index 0 = Player 1, index 1 = Player 2.
  // Each player's hide state persists independently across turns.
  const [fleetHidden, setFleetHidden] = useState([false, false]);

  // Human vs Human: create the game immediately on mount.
  useEffect(() => {
    if (mode === "human_vs_human") {
      startGame(null, 0);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function refreshView(nextViewerIndex) {
    const idx = nextViewerIndex !== undefined ? nextViewerIndex : viewerIndex;
    const state = await getGame(gameId, idx);
    setGameState(state);
    setViewerIndex(idx);
    return state;
  }

  async function startGame(aiStrategy, initialViewer = 0) {
    setError(null);
    try {
      const { game_id } = await createGame({ mode, aiStrategy });
      setGameId(game_id);
      const state = await getGame(game_id, initialViewer);
      setGameState(state);
      setViewerIndex(initialViewer);
      setPhase(PHASES.PLACEMENT);
    } catch (err) {
      setError(err.message);
    }
  }

  async function handlePlace(shipName, row, col, orientation) {
    await placeShip(gameId, { playerIndex: viewerIndex, shipName, row, col, orientation });
    const state = await refreshView(viewerIndex);

    const fullyPlaced = state.your_player.remaining_ships.length === 5;
    if (!fullyPlaced) return;

    if (mode === "human_vs_ai") {
      setPhase(PHASES.BATTLE);
      return;
    }

    // HvH: hand off to the other player for placement.
    setPendingViewerAfterPass(1 - viewerIndex);
    setPhase(PHASES.PASS_DEVICE);
  }

  async function handleContinuePassDevice() {
    const state = await refreshView(pendingViewerAfterPass);
    setPhase(state.players_ready ? PHASES.BATTLE : PHASES.PLACEMENT);
  }

  async function loadReveal() {
    const [asPlayer0, asPlayer1] = await Promise.all([
      getGame(gameId, 0),
      getGame(gameId, 1),
    ]);
    setRevealData({
      left: asPlayer0.your_player,
      right: asPlayer1.your_player,
      winner: asPlayer0.winner,
    });
    setFinalState(asPlayer0);
    setPhase(PHASES.REVEAL);
  }

  async function handleAttack(row, col) {
    if (aiThinking) return;
    setError(null);

    // Determine which player is acting this turn from the game state.
    const isMyTurn = gameState.current_player === gameState.your_player.name;
    const actingPlayerIndex = isMyTurn ? viewerIndex : 1 - viewerIndex;

    try {
      const { result } = await attack(gameId, { playerIndex: actingPlayerIndex, row, col });
      setLastMessage(result.message);

      if (result.game_over) {
        await loadReveal();
        return;
      }

      await refreshView();

      if (mode === "human_vs_ai") {
        await triggerAiMove();
      } else {
        // HvH: flip the viewer to the next player so they see their own fleet.
        await refreshView(1 - viewerIndex);
      }
    } catch (err) {
      setError(err.message);
    }
  }

  async function triggerAiMove() {
    setAiThinking(true);
    try {
      await new Promise((r) => setTimeout(r, 500));
      const { result } = await aiMove(gameId);
      setLastMessage(`AI attacked. ${result.message}`);
      if (result.game_over) {
        await loadReveal();
      } else {
        await refreshView();
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setAiThinking(false);
    }
  }

  // Mask own ship cells when fleet is hidden (replace "S" with "~" visually).
  function maskBoard(board) {
    return board.map((row) =>
      row.map((cell) => (cell === "S" ? "~" : cell))
    );
  }

  // --- Phase renders ---

  if (phase === PHASES.CHOOSE_STRATEGY) {
    return (
      <div className="card">
        <StrategySelector onSelect={(key) => startGame(key)} />
        {error && <div className="error-text">{error}</div>}
      </div>
    );
  }

  if (phase === PHASES.RESULTS) {
    if (!finalState) return <p>Loading results...</p>;
    return <GameResults state={finalState} onExit={onExit} />;
  }

  if (phase === PHASES.REVEAL) {
    if (!revealData) return <p>Loading final boards...</p>;
    return (
      <div className="app-shell">
        <div className="card" style={{ textAlign: "center" }}>
          <h2>{revealData.winner ? `${revealData.winner} wins!` : "Game over"}</h2>
        </div>

        <div className="boards-row">
          <div>
            <Board
              title={revealData.left.name}
              board={revealData.left.board}
              variant="user"
              sunkShips={revealData.left.sunk_ships}
            />
            <div style={{ marginTop: 12 }}>
              <ShipStatus
                title={`${revealData.left.name}'s Fleet`}
                remainingShips={revealData.left.remaining_ships}
              />
            </div>
          </div>
          <div>
            <Board
              title={revealData.right.name}
              board={revealData.right.board}
              variant="enemy"
              sunkShips={revealData.right.sunk_ships}
            />
            <div style={{ marginTop: 12 }}>
              <ShipStatus
                title={`${revealData.right.name}'s Fleet`}
                remainingShips={revealData.right.remaining_ships}
              />
            </div>
          </div>
        </div>

        <button className="btn" onClick={() => setPhase(PHASES.RESULTS)}>
          Continue
        </button>
      </div>
    );
  }

  if (!gameState) {
    return (
      <div className="card" style={{ textAlign: "center" }}>
        {error ? (
          <>
            <p className="error-text">Failed to start game: {error}</p>
            <button className="btn" onClick={() => startGame(null, 0)}>
              Retry
            </button>
          </>
        ) : (
          <p>Setting up game...</p>
        )}
      </div>
    );
  }

  if (phase === PHASES.PASS_DEVICE) {
    const nextName =
      pendingViewerAfterPass === viewerIndex
        ? gameState.your_player.name
        : gameState.opponent_player.name;
    return (
      <div className="pass-device-overlay">
        <div className="pass-device-card">
          <h2>Pass the device to <strong>{nextName}</strong></h2>
          <button className="btn" onClick={handleContinuePassDevice} id="pass-device-continue">
            Continue
          </button>
        </div>
      </div>
    );
  }

  if (phase === PHASES.PLACEMENT) {
    return (
      <div className="app-shell">
        <h2>{gameState.your_player.name}, place your ships</h2>
        <ShipPlacement
          board={gameState.your_player.board}
          placedShipNames={gameState.your_player.remaining_ships}
          onPlace={handlePlace}
        />
        {error && <div className="error-text">{error}</div>}
      </div>
    );
  }

  // --- BATTLE ---
  const isMyTurn = gameState.current_player === gameState.your_player.name;
  const canAttack = phase === PHASES.BATTLE && !aiThinking;

  // Both players always attack the RIGHT board; left is always their own fleet.
  const leftClickable = false;
  const rightClickable = canAttack && isMyTurn;
  const rightInactive = !rightClickable && phase === PHASES.BATTLE;

  const statusHeadline = aiThinking
    ? "AI is thinking..."
    : isMyTurn
      ? mode === "human_vs_human"
        ? `${gameState.your_player.name}'s turn — attack the right board`
        : "Your turn — pick a cell on the enemy board"
      : mode === "human_vs_human"
        ? `Waiting for ${gameState.current_player}…`
        : "Waiting for opponent...";

  const leftData = gameState.your_player;
  const rightData = gameState.opponent_player;
  // Current viewer's hide state (persists per player independently).
  const currentHidden = fleetHidden[viewerIndex];
  function toggleFleet() {
    setFleetHidden((prev) => {
      const next = [...prev];
      next[viewerIndex] = !next[viewerIndex];
      return next;
    });
  }
  const displayLeftBoard = currentHidden ? maskBoard(leftData.board) : leftData.board;

  return (
    <div className="app-shell">
      <GameStatus
        headline={statusHeadline}
        isMyTurn={isMyTurn && !aiThinking}
        lastMessage={lastMessage}
      />

      {error && <div className="error-text">{error}</div>}

      <div className="boards-row">
        {/* LEFT — current viewer's own fleet */}
        <div>
          <div className="board-header-row">
            <span className="board-header-title">{leftData.name} (You)</span>
            {mode === "human_vs_human" && (
              <button
                className="btn secondary hide-fleet-btn"
                onClick={toggleFleet}
                disabled={!isMyTurn}
                title={
                  !isMyTurn
                    ? "Only available on your turn"
                    : currentHidden
                      ? "Show your fleet"
                      : "Hide your fleet from opponent"
                }
                id="toggle-hide-fleet"
              >
                {currentHidden ? "👁 Show Fleet" : "🙈 Hide Fleet"}
              </button>
            )}
          </div>
          <Board
            title={`${leftData.name} (You)`}
            board={displayLeftBoard}
            variant="user"
            clickable={false}
            inactive={false}
            sunkShips={leftData.sunk_ships}
          />
          <div style={{ marginTop: 12 }}>
            <ShipStatus
              title={`${leftData.name}'s Fleet`}
              remainingShips={leftData.remaining_ships}
            />
          </div>
        </div>

        {/* RIGHT — opponent's board (attack target) */}
        <div>
          <Board
            title={rightData.name}
            board={rightData.board}
            variant="enemy"
            clickable={rightClickable}
            inactive={rightInactive}
            onCellClick={handleAttack}
            sunkShips={rightData.sunk_ships}
          />
          <div style={{ marginTop: 12 }}>
            <ShipStatus
              title={`${rightData.name}'s Fleet`}
              remainingShips={rightData.remaining_ships}
            />
          </div>
        </div>
      </div>
    </div>
  );
}