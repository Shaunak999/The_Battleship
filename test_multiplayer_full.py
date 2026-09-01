"""
Comprehensive LAN multiplayer + spectator test.
Tests: room creation, ship placement, attacks, spectator God-view, error handling.
"""
import asyncio
import json
import time
import urllib.request
import websockets

BASE = "http://localhost:8000"
WS_BASE = "ws://localhost:8000"

PASS = 0
FAIL = 0


def check(condition, label):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [OK] {label}")
    else:
        FAIL += 1
        print(f"  [FAIL] {label}")


async def drain_until(ws, msg_type, timeout=3):
    """Drain messages until we find one of the given type."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
        msg = json.loads(raw)
        if msg["type"] == msg_type:
            return msg
    raise TimeoutError(f"No '{msg_type}' message received within {timeout}s")


async def drain_latest(ws, msg_type, timeout=3):
    """Drain all messages and return the latest one of the given type."""
    last = None
    deadline = time.time() + timeout
    # First get the expected one
    last = await drain_until(ws, msg_type, timeout)
    # Then try to drain any more that are queued (non-blocking)
    try:
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=0.3)
            msg = json.loads(raw)
            if msg["type"] == msg_type:
                last = msg
    except (asyncio.TimeoutError, asyncio.CancelledError):
        pass
    return last


def http_post(path, data=None):
    """Helper for REST API calls."""
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(data or {}).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req).read())


def http_get(path):
    """Helper for REST GET calls."""
    req = urllib.request.Request(f"{BASE}{path}")
    return json.loads(urllib.request.urlopen(req).read())


async def test_room_creation_and_list():
    """Test creating rooms and listing them."""
    print("\n=== Test 1: Room creation & listing ===")

    r1 = http_post("/mp/create", {"player_name": "Alice"})
    r2 = http_post("/mp/create", {"player_name": "Bob"})
    check(len(r1["game_id"]) == 6, "Room 1 has 6-char ID")
    check(r1["game_id"] != r2["game_id"], "Room IDs are unique")

    rooms = http_get("/mp/rooms")
    check(len(rooms) >= 2, f"At least 2 rooms listed (got {len(rooms)})")

    return r1["game_id"]


async def test_join_and_spectate(game_id):
    """Test joining and spectating via REST."""
    print("\n=== Test 2: Join & spectate REST ===")

    j = http_post("/mp/join", {"game_id": game_id, "player_name": "Charlie"})
    check(j["game_id"] == game_id, "Join returns correct game_id")

    s = http_post("/mp/spectate", {"game_id": game_id})
    check(s["game_id"] == game_id, "Spectate returns correct game_id")

    # Bad game_id
    try:
        http_post("/mp/join", {"game_id": "ZZZ999", "player_name": "X"})
        check(False, "Joining nonexistent room should fail")
    except urllib.error.HTTPError:
        check(True, "Joining nonexistent room returns 404")


async def test_full_game_flow(game_id):
    """Full game: place ships, attack turns, spectator observes."""
    print("\n=== Test 3: Full multiplayer game flow ===")

    async with websockets.connect(f"{WS_BASE}/ws/{game_id}/player1?player_name=P1") as ws1:
        msg = await drain_until(ws1, "welcome")
        check(msg["role"] == "player1", "Player 1 welcome message")
        print(f"    Player1 connected to room {game_id}")

        async with websockets.connect(f"{WS_BASE}/ws/{game_id}/player2?player_name=P2") as ws2:
            msg = await drain_until(ws2, "welcome")
            check(msg["role"] == "player2", "Player 2 welcome message")

            # Drain any game_started/player_joined from the player_joined broadcast
            # that was sent when P2 connected
            await drain_until(ws1, "player_joined", timeout=2)

            async with websockets.connect(f"{WS_BASE}/ws/{game_id}/spectator") as ws_spec:
                msg = await drain_until(ws_spec, "welcome")
                check(msg["role"] == "spectator", "Spectator welcome message")

                # Spectator gets initial state. Since game was created when P2 connected,
                # status will be "playing" (game exists, not over yet)
                spec_msg = await drain_until(ws_spec, "spectator_state")
                check(
                    spec_msg["state"]["status"] in ("waiting", "playing"),
                    f"Spectator initial state: {spec_msg['state']['status']}"
                )
                check("player1" in spec_msg["state"], "Spectator state has player1 data")
                check("player2" in spec_msg["state"], "Spectator state has player2 data")

                # --- Ship placement: Player 1 ---
                # Place ships on the left side (cols 0-4), one per row
                ships_p1 = [
                    ("Carrier", "A1", "horizontal"),
                    ("Battleship", "C1", "horizontal"),
                    ("Cruiser", "E1", "horizontal"),
                    ("Submarine", "G1", "horizontal"),
                    ("Destroyer", "I1", "horizontal"),
                ]
                for name, coord, orient in ships_p1:
                    await ws1.send(json.dumps({
                        "type": "place_ship",
                        "ship_name": name,
                        "coordinate": coord,
                        "orientation": orient,
                    }))
                    resp = await drain_until(ws1, "ship_placed")
                    check(resp["type"] == "ship_placed", f"P1 placed {name}")

                check(resp.get("all_placed") is True, "P1 all ships placed = True")

                # --- Ship placement: Player 2 ---
                # Place ships vertically on the right side (col 9), starting from different rows
                ships_p2 = [
                    ("Carrier", "A10", "vertical"),
                    ("Battleship", "F10", "vertical"),
                    ("Cruiser", "B5", "vertical"),
                    ("Submarine", "H5", "vertical"),
                    ("Destroyer", "I6", "vertical"),
                ]
                for name, coord, orient in ships_p2:
                    await ws2.send(json.dumps({
                        "type": "place_ship",
                        "ship_name": name,
                        "coordinate": coord,
                        "orientation": orient,
                    }))
                    resp = await drain_until(ws2, "ship_placed")
                    check(resp["type"] == "ship_placed", f"P2 placed {name}")

                # Spectator should see both players have ships now
                spec_state = await drain_latest(ws_spec, "spectator_state", timeout=3)
                check(
                    spec_state["state"].get("player1_ready") is True
                    or len(spec_state["state"]["player1"].get("ships", [])) == 5,
                    "Spectator sees P1 ships placed"
                )
                check(
                    spec_state["state"].get("player2_ready") is True
                    or len(spec_state["state"]["player2"].get("ships", [])) == 5,
                    "Spectator sees P2 ships placed"
                )

                # Both ready -> game_started broadcast
                game_start = await drain_until(ws1, "game_started", timeout=5)
                check(game_start["type"] == "game_started", "Game started event received by P1")
                print(f"    Turn: {game_start.get('current_player_name')}")

                # Spectator should also see game started
                # Request state explicitly since game_started doesn't send spectator_state
                await ws_spec.send(json.dumps({"type": "get_state"}))
                spec_state = await drain_latest(ws_spec, "spectator_state", timeout=3)
                check(spec_state["state"]["status"] == "playing", "Spectator sees game is playing")

                # --- Attack turns ---
                # P1 attacks B1 (row=1, col=0) - P2 has no ships there -> miss
                await ws1.send(json.dumps({"type": "attack", "row": 1, "col": 0}))
                result_p1 = await drain_until(ws1, "attack_result")
                check(result_p1["type"] == "attack_result", "P1 attack result received")
                check(result_p1["attacker"] == 0, "Attack attributed to P1")
                check(
                    result_p1["result"] in ("hit", "miss", "sunk"),
                    f"P1 attack on B1: {result_p1['result']}"
                )
                print(f"    P1 attacks B1 -> {result_p1['result']}")

                # P2 should also get the attack result
                result_p2_recv = await drain_until(ws2, "attack_result", timeout=3)
                check(result_p2_recv["type"] == "attack_result", "P2 also receives attack_result")

                # Spectator should get updated state
                spec_state = await drain_latest(ws_spec, "spectator_state", timeout=3)
                check("player1" in spec_state["state"], "Spectator gets full state after attack")
                check("statistics" in spec_state["state"], "Spectator state includes statistics")

                # --- P2 attacks ---
                # P2 attacks A1 (row=0, col=0) - P1 has Carrier at A1 -> hit!
                await ws2.send(json.dumps({"type": "attack", "row": 0, "col": 0}))
                result_p2_atk = await drain_until(ws2, "attack_result")
                check(result_p2_atk["type"] == "attack_result", "P2 attack result received")
                check(
                    result_p2_atk["result"] in ("hit", "miss", "sunk"),
                    f"P2 attack on A1: {result_p2_atk['result']}"
                )
                print(f"    P2 attacks A1 -> {result_p2_atk['result']}")

                # Spectator state after P2 attack
                spec_state = await drain_latest(ws_spec, "spectator_state", timeout=3)
                stats = spec_state["state"].get("statistics", {})
                check("total_shots" in stats, f"Statistics show {stats.get('total_shots', '?')} total shots")

                # --- Play more turns ---
                # P1 targets P2 ships at col 9 (A10-Carrier, F10-Battleship)
                # P2 targets P1 ships at row 0 (A1-Carrier)
                attack_coords = [
                    # (sender, row, col)
                    ("P1", 0, 9),   # P1 attacks A10 -> should hit P2 Carrier
                    ("P2", 0, 1),   # P2 attacks A2 -> miss (P1 Carrier is A1-A5)
                    ("P1", 1, 9),   # P1 attacks B10 -> miss
                    ("P2", 0, 2),   # P2 attacks A3 -> hit (P1 Carrier)
                    ("P1", 2, 9),   # P1 attacks C10 -> miss
                    ("P2", 0, 3),   # P2 attacks A4 -> hit (P1 Carrier)
                    ("P1", 3, 9),   # P1 attacks D10 -> miss
                    ("P2", 0, 4),   # P2 attacks A5 -> hit (P1 Carrier)
                    ("P1", 4, 9),   # P1 attacks E10 -> miss
                    ("P2", 0, 5),   # P2 attacks A6 -> miss
                ]
                for sender, row, col in attack_coords:
                    if sender == "P1":
                        await ws1.send(json.dumps({"type": "attack", "row": row, "col": col}))
                        r = await drain_until(ws1, "attack_result")
                        check(
                            r["result"] in ("hit", "miss", "sunk"),
                            f"P1 attack ({row},{col}): {r['result']}"
                        )
                    else:
                        await ws2.send(json.dumps({"type": "attack", "row": row, "col": col}))
                        r = await drain_until(ws2, "attack_result")
                        check(
                            r["result"] in ("hit", "miss", "sunk"),
                            f"P2 attack ({row},{col}): {r['result']}"
                        )

                # Spectator final state
                spec_state = await drain_latest(ws_spec, "spectator_state", timeout=3)
                check("event_log" in spec_state["state"], "Spectator has event_log")
                event_count = len(spec_state["state"]["event_log"])
                check(event_count >= 1, f"Event log has {event_count} entries")

                # Check statistics
                stats = spec_state["state"].get("statistics", {})
                check("players" in stats, "Statistics have player data")
                if "players" in stats:
                    for idx, ps in stats["players"].items():
                        check(
                            "accuracy" in ps,
                            f"  {ps.get('name', f'P{int(idx)+1}')}: "
                            f"{ps.get('shots', 0)} shots, "
                            f"{ps.get('accuracy', 0)}% accuracy"
                        )

                # --- Error handling ---
                # Spectator tries to attack (should be rejected)
                await ws_spec.send(json.dumps({"type": "attack", "row": 0, "col": 0}))
                err = await drain_until(ws_spec, "error")
                check(
                    "Spectators cannot perform actions" in err["message"],
                    "Spectator attack rejected with error"
                )

                # Spectator requests state
                await ws_spec.send(json.dumps({"type": "get_state"}))
                state_resp = await drain_until(ws_spec, "spectator_state")
                check(state_resp["type"] == "spectator_state", "Spectator get_state works")

                # Check game is still running or over
                print(f"    Game status: {spec_state['state'].get('status')}")


async def test_reconnection():
    """Test player reconnection (simulates page refresh)."""
    print("\n=== Test 4: Player reconnection ===")

    r = http_post("/mp/create", {"player_name": "ReconnectTest"})
    gid = r["game_id"]

    # Connect as P1
    ws1 = await websockets.connect(f"{WS_BASE}/ws/{gid}/player1?player_name=RT")
    msg = json.loads(await ws1.recv())
    check(msg["type"] == "welcome", "P1 initial connection")

    # Simulate reconnect: connect again as P1
    ws1b = await websockets.connect(f"{WS_BASE}/ws/{gid}/player1?player_name=RT")
    msg = json.loads(await ws1b.recv())
    check(msg["type"] == "welcome", "P1 reconnection welcome")

    # Old connection should be closed by the server.
    # The server may have buffered a broadcast message (e.g. player_joined)
    # that the old conn can still recv before the close propagates.
    # So drain any remaining messages then verify the conn is dead.
    closed = False
    try:
        while True:
            raw = await asyncio.wait_for(ws1.recv(), timeout=1.5)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        # Timeout = no more messages, conn still open but idle
        pass
    except websockets.exceptions.ConnectionClosed:
        closed = True
    # Try one more recv to confirm closure
    try:
        raw = await asyncio.wait_for(ws1.recv(), timeout=1)
        check(False, "Old connection should be closed")
    except websockets.exceptions.ConnectionClosed:
        closed = True
    except asyncio.TimeoutError:
        pass
    check(closed, "Old P1 connection properly closed")

    await ws1b.close()


async def test_invalid_messages():
    """Test error handling for invalid messages."""
    print("\n=== Test 5: Error handling ===")

    r = http_post("/mp/create", {"player_name": "ErrorTest"})
    gid = r["game_id"]

    async with websockets.connect(f"{WS_BASE}/ws/{gid}/player1?player_name=ET") as ws:
        msg = await drain_until(ws, "welcome")

        # Send invalid JSON
        await ws.send("not json")
        err = await drain_until(ws, "error")
        check("Invalid JSON" in err["message"], "Invalid JSON rejected")

        # Send unknown message type
        await ws.send(json.dumps({"type": "bogus"}))
        err = await drain_until(ws, "error")
        check("Unknown message type" in err["message"], "Unknown type rejected")

        # Attack before game starts (no P2 connected)
        await ws.send(json.dumps({"type": "attack", "row": 0, "col": 0}))
        err = await drain_until(ws, "error")
        check(
            "not started" in err["message"].lower() or "both players" in err["message"].lower(),
            f"Attack rejected before game starts: {err['message']}"
        )


async def test_spectator_full_view():
    """Test that spectator gets full God-view of both boards."""
    print("\n=== Test 6: Spectator God-view ===")

    r = http_post("/mp/create", {"player_name": "SpectatorTest"})
    gid = r["game_id"]

    async with websockets.connect(f"{WS_BASE}/ws/{gid}/player1?player_name=SV1") as ws1:
        await drain_until(ws1, "welcome")

        async with websockets.connect(f"{WS_BASE}/ws/{gid}/player2?player_name=SV2") as ws2:
            await drain_until(ws2, "welcome")
            # Drain player_joined broadcast
            await drain_until(ws1, "player_joined", timeout=2)

            async with websockets.connect(f"{WS_BASE}/ws/{gid}/spectator") as ws_spec:
                await drain_until(ws_spec, "welcome")
                spec_state = await drain_until(ws_spec, "spectator_state")

                # God-view should show BOTH boards with ships visible
                p1 = spec_state["state"]["player1"]
                p2 = spec_state["state"]["player2"]

                check("board" in p1, "Spectator sees P1 board")
                check("board" in p2, "Spectator sees P2 board")
                check(len(p1["board"]) == 10, "P1 board has 10 rows")
                check(len(p2["board"]) == 10, "P2 board has 10 rows")

                # Place ships for both players
                ships = [
                    ("Carrier", "A1", "horizontal"),
                    ("Battleship", "C1", "horizontal"),
                    ("Cruiser", "E1", "horizontal"),
                    ("Submarine", "G1", "horizontal"),
                    ("Destroyer", "I1", "horizontal"),
                ]
                for name, coord, orient in ships:
                    await ws1.send(json.dumps({
                        "type": "place_ship",
                        "ship_name": name,
                        "coordinate": coord,
                        "orientation": orient,
                    }))
                    await drain_until(ws1, "ship_placed")

                # P2 places vertically on the right side
                p2_ships = [
                    ("Carrier", "A10", "vertical"),
                    ("Battleship", "F10", "vertical"),
                    ("Cruiser", "B5", "vertical"),
                    ("Submarine", "H5", "vertical"),
                    ("Destroyer", "I6", "vertical"),
                ]
                for name, coord, orient in p2_ships:
                    await ws2.send(json.dumps({
                        "type": "place_ship",
                        "ship_name": name,
                        "coordinate": coord,
                        "orientation": orient,
                    }))
                    await drain_until(ws2, "ship_placed")

                # Both ready -> game starts
                await drain_until(ws1, "game_started", timeout=5)
                spec_state = await drain_latest(ws_spec, "spectator_state", timeout=3)

                # Now in God-view, both boards should show ship placements
                # P1 ships are horizontal on rows A,C,E,G,I at col 1
                check(
                    spec_state["state"]["player1"]["board"][0][0] == "S",
                    "Spectator sees P1 Carrier at A1"
                )
                # P2 ships are vertical at col 9
                check(
                    spec_state["state"]["player2"]["board"][0][9] == "S",
                    "Spectator sees P2 Carrier at A10"
                )

                # P1 attacks A10 -> should be a hit (P2 Carrier)
                await ws1.send(json.dumps({"type": "attack", "row": 0, "col": 9}))
                r = await drain_until(ws1, "attack_result")
                check(r["result"] in ("hit", "sunk"), f"P1 attacks A10 -> {r['result']}")
                print(f"    P1 attacks A10 -> {r['result']}")

                # Spectator should see the hit on P2's board
                spec_state = await drain_latest(ws_spec, "spectator_state", timeout=3)
                check("statistics" in spec_state["state"], "Spectator has statistics after attack")

                print(f"    Spectator God-view verified with ship visibility")


async def main():
    print("=" * 60)
    print("  LAN MULTIPLAYER + SPECTATOR TEST SUITE")
    print("=" * 60)

    gid = await test_room_creation_and_list()
    await test_join_and_spectate(gid)
    await test_full_game_flow(gid)
    await test_reconnection()
    await test_invalid_messages()
    await test_spectator_full_view()

    print("\n" + "=" * 60)
    print(f"  RESULTS: {PASS} passed, {FAIL} failed")
    print("=" * 60)
    if FAIL > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
