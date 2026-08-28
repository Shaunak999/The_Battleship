import asyncio, json, urllib.request, websockets

async def drain_until(ws, msg_type, timeout=3):
    """Drain messages until we find one of the given type."""
    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        raw = await asyncio.wait_for(ws.recv(), timeout=deadline - time.time())
        msg = json.loads(raw)
        if msg['type'] == msg_type:
            return msg
    raise TimeoutError(f"No '{msg_type}' message received within {timeout}s")

async def test():
    req = urllib.request.Request('http://localhost:8000/mp/create',
        data=json.dumps({'player_name':'P1'}).encode(), headers={'Content-Type':'application/json'})
    gid = json.loads(urllib.request.urlopen(req).read())['game_id']
    print(f'1. Room created: {gid}')

    async with websockets.connect(f'ws://localhost:8000/ws/{gid}/player1') as ws1:
        msg = json.loads(await ws1.recv())
        assert msg['type'] == 'welcome'
        print(f'2. Player1 connected')

        async with websockets.connect(f'ws://localhost:8000/ws/{gid}/player2') as ws2:
            msg = json.loads(await ws2.recv())
            assert msg['type'] == 'welcome'
            print(f'3. Player2 connected')

            async with websockets.connect(f'ws://localhost:8000/ws/{gid}/spectator') as ws_spec:
                msg = json.loads(await ws_spec.recv())
                assert msg['type'] == 'welcome'
                print(f'4. Spectator connected')

                # Spectator gets state
                msg = await drain_until(ws_spec, 'spectator_state')
                assert msg['state']['status'] in ('waiting', 'playing')
                print(f'5. Spectator God View: status={msg["state"]["status"]}')

                # P1 places all 5 ships
                ships = [
                    ('Carrier', 'A1', 'horizontal'),
                    ('Battleship', 'B1', 'horizontal'),
                    ('Cruiser', 'C1', 'horizontal'),
                    ('Submarine', 'D1', 'horizontal'),
                    ('Destroyer', 'E1', 'horizontal'),
                ]
                for name, coord, orient in ships:
                    await ws1.send(json.dumps({
                        'type': 'place_ship', 'ship_name': name,
                        'coordinate': coord, 'orientation': orient
                    }))
                    resp = await drain_until(ws1, 'ship_placed')
                    assert resp['type'] == 'ship_placed'
                print(f'6. Player1 placed all ships')

                # P2 places all 5 ships
                for name, coord, orient in ships:
                    await ws2.send(json.dumps({
                        'type': 'place_ship', 'ship_name': name,
                        'coordinate': coord, 'orientation': orient
                    }))
                    resp = await drain_until(ws2, 'ship_placed')
                    assert resp['type'] == 'ship_placed'
                print(f'7. Player2 placed all ships')

                # Game should start (both ready)
                msg = await drain_until(ws1, 'game_started')
                print(f'8. Game started! Turn: {msg["current_player_name"]}')

                # P1 attacks (row=1, col=0 = B1 = miss on P2 board)
                await ws1.send(json.dumps({'type': 'attack', 'row': 1, 'col': 0}))
                result = await drain_until(ws1, 'attack_result')
                assert result['type'] == 'attack_result'
                print(f'9. P1 attacks B1 -> {result["result"]}')

                # Spectator should have full state with both boards
                msg = await drain_until(ws_spec, 'spectator_state')
                p1_board = msg['state']['player1']['board']
                p2_board = msg['state']['player2']['board']
                assert len(p1_board) == 10
                assert len(p2_board) == 10
                print(f'10. Spectator sees BOTH boards (10x10 each)')

                print('\n=== ALL TESTS PASSED ===')

asyncio.run(test())
