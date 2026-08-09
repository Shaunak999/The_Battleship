import { useEffect, useState } from "react";
import Board from "../components/Board";
import ShipStatus from "../components/ShipStatus";
import GameStatus from "../components/GameStatus";
import StrategySelector from "../components/StrategySelector";
import ShipPlacement from "../components/ShipPlacement";
import { createGame, placeShip, attack, aiMove, getGame } from "../services/api";

/**
 * Perspective note: every render always re-fetches state with an explicit
 * viewer_index via getGame(), rather than trusting the "state" object
 * embedded in place/attack/ai-move responses. Those responses default
 * their viewer to whoever's turn it currently is post-action, which is
 * NOT necessarily the screen we want to render (e.g. right after a human
 * attacks in Human vs AI mode, the embedded default would briefly be the
 * AI's own perspective — which would incorrectly hide the human's own
 * ships if shown on the human's screen). One extra request per action
 * buys correctness on which ships are visible to whom.
 */

const PHASES = {
  CHOOSE_STRATEGY: "choose_strategy",
  PLACEMENT: "placement",
  PASS_DEVICE: "pass_device",
  BATTLE: "battle",
  GAME_OVER: "game_over",
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

  // Human vs Human only: create the game immediately, no strategy step.
  useEffect(() => {
    if (mode === "human_vs_human") {
      startGame(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function refreshView(nextViewerIndex) {
    const state = await getGame(gameId, nextViewerIndex);
    setGameState(state);
    setViewerIndex(nextViewerIndex);
    return state;
  }

  async function startGame(aiStrategy) {
    setError(null);
    try {
      const { game_id } = await createGame({
        mode,
        aiStrategy,
      });
      setGameId(game_id);
      const state = await getGame(game_id, 0);
      setGameState(state);
      setViewerIndex(0);
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
      // Backend auto-places the AI's fleet once the human finishes.
      setPhase(PHASES.BATTLE);
      return;
    }

    // Human vs Human: whoever just finished placing hands off to the other player.
    // If that's player 1 finishing, players_ready flips true and the pass-device
    // screen leads into battle instead of back into placement.
    setPendingViewerAfterPass(1 - viewerIndex);
    setPhase(PHASES.PASS_DEVICE);
  }

  async function handleContinuePassDevice() {
    const state = await refreshView(pendingViewerAfterPass);
    setPhase(state.players_ready ? PHASES.BATTLE : PHASES.PLACEMENT);
  }

  async function handleAttack(row, col) {
    if (aiThinking) return;
    setError(null);
    try {
      const { result } = await attack(gameId, { playerIndex: viewerIndex, row, col });
      setLastMessage(result.message);

      if (result.game_over) {
        await refreshView(viewerIndex);
        setPhase(PHASES.GAME_OVER);
        return;
      }

      if (mode === "human_vs_ai") {
        await refreshView(0);
        await triggerAiMove();
      } else {
        // Human vs Human: always pass the device after a shot.
        setPendingViewerAfterPass(1 - viewerIndex);
        setPhase(PHASES.PASS_DEVICE);
      }
    } catch (err) {
      setError(err.message);
    }
  }

  async function triggerAiMove() {
    setAiThinking(true);
    try {
      // Small delay so "AI is thinking..." is visible rather than instant.
      await new Promise((r) => setTimeout(r, 500));
      const { result } = await aiMove(gameId);
      setLastMessage(`AI attacked. ${result.message}`);
      await refreshView(0);
      if (result.game_over) {
        setPhase(PHASES.GAME_OVER);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setAiThinking(false);
    }
  }

  if (phase === PHASES.CHOOSE_STRATEGY) {
    return (
      <div className="card">
        <StrategySelector onSelect={(key) => startGame(key)} />
        {error && <div className="error-text">{error}</div>}
      </div>
    );
  }

  if (!gameState) {
    return <p>Setting up game...</p>;
  }

  if (phase === PHASES.PASS_DEVICE) {
    const nextName =
      pendingViewerAfterPass === viewerIndex
        ? gameState.your_player.name
        : gameState.opponent_player.name;
    return (
      <div className="pass-device-overlay">
        <h2>{nextName}'s turn</h2>
        <p>Pass the device to {nextName}, then continue.</p>
        <button className="btn" onClick={handleContinuePassDevice}>
          Continue
        </button>
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

  const isMyTurn = gameState.current_player === gameState.your_player.name;

  return (
    <div className="app-shell">
      <GameStatus
        isMyTurn={isMyTurn && !aiThinking}
        aiThinking={aiThinking}
        lastMessage={lastMessage}
      />

      {error && <div className="error-text">{error}</div>}

      <div className="boards-row">
        <div>
          <Board title="Your Board" board={gameState.your_player.board} variant="user" />
          <div style={{ marginTop: 12 }}>
            <ShipStatus title="Your Fleet" remainingShips={gameState.your_player.remaining_ships} />
          </div>
        </div>

        <div>
          <Board
            title="Enemy Board"
            board={gameState.opponent_player.board}
            variant="enemy"
            clickable={phase === PHASES.BATTLE && isMyTurn && !aiThinking}
            onCellClick={handleAttack}
          />
          <div style={{ marginTop: 12 }}>
            <ShipStatus
              title="Enemy Fleet"
              remainingShips={gameState.opponent_player.remaining_ships}
            />
          </div>
        </div>
      </div>

      {phase === PHASES.GAME_OVER && (
        <div className="card" style={{ textAlign: "center" }}>
          <h2>{gameState.winner ? `${gameState.winner} wins!` : "Game over"}</h2>
          <button className="btn" onClick={onExit}>
            Back to Home
          </button>
        </div>
      )}
    </div>
  );
}