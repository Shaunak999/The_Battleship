const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const WS_BASE = import.meta.env.VITE_WS_URL || "ws://localhost:8000";

const COLS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"];

export function toCoordinate(row, col) {
  return `${COLS[row]}${col + 1}`;
}

export function getWsUrl(gameId, role) {
  return `${WS_BASE}/ws/${gameId}/${role}`;
}

async function request(path, options = {}) {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  let body = null;
  try {
    body = await res.json();
  } catch {
    // no JSON body
  }

  if (!res.ok) {
    const detail = body?.detail || `Request failed (${res.status})`;
    throw new Error(detail);
  }

  return body;
}

export function createGame({ mode, player1Name, player2Name, aiStrategy, aiStrategy2 }) {
  return request("/game/create", {
    method: "POST",
    body: JSON.stringify({
      mode,
      player1_name: player1Name ?? "Player 1",
      player2_name: player2Name ?? "Player 2",
      ai_strategy: aiStrategy ?? null,
      ai_strategy_2: aiStrategy2 ?? null,
    }),
  });
}

export function placeShip(gameId, { playerIndex, shipName, row, col, orientation }) {
  return request(`/game/${gameId}/place`, {
    method: "POST",
    body: JSON.stringify({
      player_index: playerIndex,
      ship_name: shipName,
      coordinate: toCoordinate(row, col),
      orientation,
    }),
  });
}

export function attack(gameId, { playerIndex, row, col }) {
  return request(`/game/${gameId}/attack`, {
    method: "POST",
    body: JSON.stringify({
      player_index: playerIndex,
      coordinate: toCoordinate(row, col),
    }),
  });
}

export function aiMove(gameId) {
  return request(`/game/${gameId}/ai-move`, { method: "POST" });
}

export function getGame(gameId, viewerIndex) {
  const query = viewerIndex !== undefined ? `?viewer_index=${viewerIndex}` : "";
  return request(`/game/${gameId}${query}`);
}

export function aiStep(gameId) {
  return request(`/game/${gameId}/ai-step`, { method: "POST" });
}

export function getStrategies() {
  return request("/ai/strategies");
}

// ── Multiplayer REST ──────────────────────────────────────────────────

export function mpCreateGame(playerName) {
  return request("/mp/create", {
    method: "POST",
    body: JSON.stringify({ player_name: playerName }),
  });
}

export function mpJoinGame(gameId, playerName) {
  return request("/mp/join", {
    method: "POST",
    body: JSON.stringify({ game_id: gameId, player_name: playerName }),
  });
}

export function mpSpectateGame(gameId) {
  return request("/mp/spectate", {
    method: "POST",
    body: JSON.stringify({ game_id: gameId }),
  });
}

export function mpListRooms() {
  return request("/mp/rooms");
}
