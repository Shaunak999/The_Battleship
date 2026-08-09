const BASE_URL = "http://localhost:8000";

const COLS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"];

/**
 * Convert a (row, col) grid index into the backend's coordinate string,
 * e.g. row=4, col=4 -> "E5". This is the ONLY place this conversion
 * should happen — components deal in row/col ints, api.js deals in
 * the "A1" strings the backend expects.
 */
export function toCoordinate(row, col) {
  return `${COLS[row]}${col + 1}`;
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
    // no JSON body — leave body as null
  }

  if (!res.ok) {
    const detail = body?.detail || `Request failed (${res.status})`;
    throw new Error(detail);
  }

  return body;
}

export function createGame({ mode, player1Name, player2Name, aiStrategy }) {
  return request("/game/create", {
    method: "POST",
    body: JSON.stringify({
      mode,
      player1_name: player1Name ?? "Player 1",
      player2_name: player2Name ?? "Player 2",
      ai_strategy: aiStrategy ?? null,
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

export function getStrategies() {
  return request("/ai/strategies");
}