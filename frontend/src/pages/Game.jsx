import { useEffect, useState } from "react";
import Board from "../components/Board";
import ShipStatus from "../components/ShipStatus";
import GameStatus from "../components/GameStatus";
import StrategySelector from "../components/StrategySelector";
import ShipPlacement from "../components/ShipPlacement";
import GameResults from "../components/GameResults";
import { createGame, placeShip, attack, aiMove, getGame } from "../services/api";

/**
 * Perspective note: every render always re-fetches state with an explicit
 * viewer_index via getGame(), rather than trusting the "state" object
 * embedded in place/attack/ai-move responses. Those responses default
 * their viewer to whoever's turn it currently is post-action, which is
 * NOT necessarily the screen we want to render. One extra request per
 * action buys correctness on which ships are visible to whom.
 *
 * Board layout: Player 1 (index 0) is always rendered LEFT, Player 2 /
 * AI (index 1) is always rendered RIGHT — fixed by player identity, not
 * by whose turn it is. Only which board is CLICKABLE changes with turn.
 * Privacy is still enforced structurally: a given screen only ever shows
 * ship positions belonging to whichever player's device is currently up
 * (gameState.your_player), because the backend hides the other player's
 * unhit ships regardless of layout position.
 */

const PHASES = {
  CHOOSE_STRATEGY: "choose_strategy",
  PLACEMENT: "placement",
  PASS_DEVICE: "pass_device",
  BATTLE: "battle",
  REVEAL: "reveal", // game over — show both fleets fully revealed
  RESULTS: "results", // stats comparison, reached via a button from REVEAL
};

/** Presentational-only mask: hides ship cells ("S" -> "~"). Used during
 * Human vs Human battle so NEITHER player's ship positions are visible
 * on screen — not even your own — until the game ends and REVEAL shows
 * everything. Never touches real game state; the backend still knows
 * exactly where every ship is regardless. */
function maskShips(board) {
  return board.map((row) => row.map((symbol) => (symbol === "S" ? "~" : symbol)));
}

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
  const [finalState, setFinalState] = useState(null); // for stats page (viewer=0 snapshot)
  const [revealData, setRevealData] = useState(null); // both fleets fully shown

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
      const { game_id } = await createGame({ mode, aiStrategy });
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
    setPendingViewerAfterPass(1 - viewerIndex);
    setPhase(PHASES.PASS_DEVICE);
  }

  async function handleContinuePassDevice() {
    const state = await refreshView(pendingViewerAfterPass);
    setPhase(state.players_ready ? PHASES.BATTLE : PHASES.PLACEMENT);
  }

  /**
   * Fetches BOTH players' own perspectives and keeps just the fully-
   * revealed "your_player" board from each — the one view the backend
   * never hides ships on, regardless of hide_ships elsewhere. This is
   * the only way to show a winner's surviving, never-hit ships too.
   */
  async function loadReveal() {
    const [asPlayer0, asPlayer1] = await Promise.all([
      getGame(gameId, 0),
      getGame(gameId, 1),
    ]);
    setRevealData({
      left: asPlayer0.your_player, // player index 0, fully revealed
      right: asPlayer1.your_player, // player index 1, fully revealed
      winner: asPlayer0.winner,
    });
    setFinalState(asPlayer0); // stats page reuses this (hit/miss counts are never masked)
    setPhase(PHASES.REVEAL);
  }

  async function handleAttack(row, col) {
    if (aiThinking) return;
    setError(null);

    // The acting player is whoever's turn it currently is, derived from
    // gameState — not viewerIndex, which now stays fixed at 0 through the
    // whole battle phase (see the note above this component).
    const isPlayer0Turn = gameState.current_player === gameState.your_player.name;
    const actingPlayerIndex = isPlayer0Turn ? 0 : 1;

    try {
      const { result } = await attack(gameId, { playerIndex: actingPlayerIndex, row, col });
      setLastMessage(result.message);

      if (result.game_over) {
        await loadReveal();
        return;
      }

      await refreshView(0);

      if (mode === "human_vs_ai") {
        await triggerAiMove();
      }
      // Human vs Human: no pass-device here — the board that just became
      // the non-acting side simply greys out (see `inactive` below), and
      // the other board lights up for the next player's turn in place.
    } catch (err) {
      setError(err.message);
    }
  }

  async function triggerAiMove() {
    setAiThinking(true);
    try {
      await new Promise((r) => setTimeout(r, 500)); // let "AI is thinking..." be visible
      const { result } = await aiMove(gameId);
      setLastMessage(`AI attacked. ${result.message}`);
      if (result.game_over) {
        await loadReveal();
      } else {
        await refreshView(0);
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
            <Board title={revealData.left.name} board={revealData.left.board} variant="user" sunkShips={revealData.left.sunk_ships} />
            <div style={{ marginTop: 12 }}>
              <ShipStatus title={`${revealData.left.name}'s Fleet`} remainingShips={revealData.left.remaining_ships} />
            </div>
          </div>
          <div>
            <Board title={revealData.right.name} board={revealData.right.board} variant="enemy" sunkShips={revealData.right.sunk_ships} />
            <div style={{ marginTop: 12 }}>
              <ShipStatus title={`${revealData.right.name}'s Fleet`} remainingShips={revealData.right.remaining_ships} />
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

  // --- BATTLE ---
  // viewerIndex stays fixed at 0 through the whole battle phase now (it
  // only still toggles during PLACEMENT's pass-device handoff), so
  // your_player is always Player 1 (left) and opponent_player is always
  // Player 2 / AI (right). Whose turn it is comes from current_player.
  const isMyTurn = gameState.current_player === gameState.your_player.name;
  const leftData = gameState.your_player;
  const rightData = gameState.opponent_player;

  const canAttack = phase === PHASES.BATTLE && !aiThinking;
  // Human vs Human: exactly one side is clickable, following whoever's turn it is.
  // Human vs AI: only the right (AI) board is ever clickable, only on the human's turn.
  const leftClickable = mode === "human_vs_human" && canAttack && !isMyTurn;
  const rightClickable = canAttack && isMyTurn;

  const statusHeadline = aiThinking
    ? "AI is thinking..."
    : mode === "human_vs_human"
      ? `${gameState.current_player}'s turn — attack the other board`
      : isMyTurn
        ? "Your turn — pick a cell on the enemy board"
        : "Waiting for opponent...";

  // Grey out whichever side isn't currently actionable. In Human vs AI,
  // the human's own board never toggles clickable at all, so it's never
  // shown as "inactive" either — that would just be a permanent, useless grey.
  const leftInactive = mode === "human_vs_human" && phase === PHASES.BATTLE && !leftClickable;
  const rightInactive = phase === PHASES.BATTLE && !rightClickable;

  // Human vs Human: hide EVERY ship cell on both boards during battle —
  // including the viewer's own — regardless of which side "your_player"
  // currently maps to. The opponent side is already hidden by the
  // backend's hide_ships flag; masking both defensively means this
  // guarantee doesn't depend on that flag being right in every case.
  const displayLeftBoard = mode === "human_vs_human" ? maskShips(leftData.board) : leftData.board;
  const displayRightBoard =
    mode === "human_vs_human" ? maskShips(rightData.board) : rightData.board;

  return (
    <div className="app-shell">
      <GameStatus headline={statusHeadline} isMyTurn={isMyTurn && !aiThinking} lastMessage={lastMessage} />

      {error && <div className="error-text">{error}</div>}

      <div className="boards-row">
        <div>
          <Board
            title={leftData.name}
            board={displayLeftBoard}
            variant="user"
            clickable={leftClickable}
            inactive={leftInactive}
            onCellClick={handleAttack}
            sunkShips={leftData.sunk_ships}
          />
          <div style={{ marginTop: 12 }}>
            <ShipStatus title={`${leftData.name}'s Fleet`} remainingShips={leftData.remaining_ships} />
          </div>
        </div>

        <div>
          <Board
            title={rightData.name}
            board={displayRightBoard}
            variant="enemy"
            clickable={rightClickable}
            inactive={rightInactive}
            onCellClick={handleAttack}
            sunkShips={rightData.sunk_ships}
          />
          <div style={{ marginTop: 12 }}>
            <ShipStatus title={`${rightData.name}'s Fleet`} remainingShips={rightData.remaining_ships} />
          </div>
        </div>
      </div>
    </div>
  );
}