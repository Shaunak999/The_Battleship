# Daily Internship Log — Battleship AI with Deep Reinforcement Learning

**Project:** Battleship game with multiple AI strategies (Random, Hunt & Target, Probability Density, Neural Network DQN) + real-time multiplayer
**Log period:** Aug 17 – Sep 4, 2026 (17 days)

---

### Day 1 — Mon, Aug 17
**Tasks:**
- Began the neural-network phase of the Battleship AI project.
- Literature study: read up on applying deep reinforcement learning to Battleship-style games — DQN fundamentals, and existing heuristic approaches (hunt/target with parity, probability-density heatmaps).
- Designed the 6-channel board observation (un-attacked mask, hits, misses, sunk ships, probability heatmap prior, target-boundary & parity mask).
- Fixed minor UI issues in the human-play game screen.

**Notes:** The key insight from the study: a plain CNN can learn shot patterns, but feeding the probability heatmap as an input channel should speed up learning a lot.

---

### Day 2 — Tue, Aug 18
**Tasks:**
- Implemented a neural network for the first time: built the Gymnasium `BattleshipEnv` (observation encoding, action space, reward function).
- Trained a first DQN model end-to-end using Stable-Baselines3.
- Saved the first trained model checkpoint.

**Notes:** First end-to-end run complete — the agent went from random shooting to finding ships, but very inefficiently. Training loop and environment API both work.

---

### Day 3 — Wed, Aug 19
**Tasks:**
- Analyzed the first model's games: it avoided repeating shots but wasted many moves in hunt mode.
- Modified the reward system — re-weighted hit / miss / sunk rewards and penalized wasted shots.
- Retrained and compared against the previous checkpoint.

**Notes:** Reward shaping matters more than network size at this stage. Games-to-win dropped noticeably after the change.

---

### Day 4 — Thu, Aug 20
**Tasks:**
- Literature study: Dueling Network Architectures for Deep RL (value stream vs advantage stream) and the Stable-Baselines3 custom feature-extractor API.
- Compared plain Q-network vs dueling architecture for grid games; decided to implement a Dueling Q-Network.

**Notes:** Dueling should help because in many board states the *value* is high regardless of which legal shot is chosen.

---

### Day 5 — Fri, Aug 21
**Tasks:**
- Implemented `DuelingQNetwork` (3-layer CNN → separate value and advantage streams) and the `BattleshipCNN` feature extractor in `neural_ai.py`.
- Added strict action masking so the agent can never select an already-shot cell.

**Notes:** Masking is enforced both in training and at inference — invalid actions get −inf Q-values.

---

### Day 6 — Sat, Aug 22
**Tasks:**
- Built the training pipeline (`train.py`): configurable timesteps, periodic evaluation during training, best-model checkpointing.
- Ran a long training session and saved the first production model (`battleship_dqn.zip`, ~27 MB).

**Notes:** Checkpointing the best model by evaluation win-rate is essential — the final timestep is not always the best policy.

---

### Day 7 — Mon, Aug 24
**Tasks:**
- Hyperparameter tuning: learning rate, replay buffer size, batch size, exploration schedule.
- Logged win rates after each configuration to compare runs.

**Notes:** Kept a small experiment log table — learning rate had the biggest effect on stability.

---

### Day 8 — Tue, Aug 25
**Tasks:**
- Wrote the evaluation harness (`evaluate.py`): plays the trained agent against Random, Hunt & Target, and Probability AIs; measures win rate and average shots-to-win.

**Notes:** Decided on shot-efficiency (avg shots per win) as the main metric, not just win rate — it separates strong agents better.

---

### Day 9 — Wed, Aug 26
**Tasks:**
- Literature study: action masking in RL and greedy vs stochastic inference; why a deterministic greedy policy becomes predictable game after game.
- Noted de-prediction ideas for the fallback heuristic: per-game parity-phase flipping, sampling the opening shot from a band, jitter on near-ties.

**Notes:** An AI that always opens on the same cell is exploitable — variance needs to be designed in, not random by accident.

---

### Day 10 — Thu, Aug 27
**Tasks:**
- Ran model comparison experiments: 4-channel vs 6-channel observations — the heatmap-prior channel clearly improved efficiency.
- Added a global model cache so the network weights load once instead of on every move (large inference speed-up).

**Notes:** Inference latency went from noticeable to instant after caching.

---

### Day 11 — Fri, Aug 28
**Tasks:**
- Started the real-time multiplayer module: WebSocket-based human-vs-human play on localhost.
- Backend: game manager and room handling. Frontend: multiplayer lobby, game page, spectator view, and a "watch AI battle" page.

**Notes:** Biggest architectural change of the project — moving from request/response to a stateful socket-based game loop.

---

### Day 12 — Sat, Aug 29
**Tasks:**
- Completed the multiplayer connection setup: frontend env config, Vite proxy settings.
- Wrote `test_ws.py` to verify the socket message flow.

**Notes:** Room join/rejoin logic needs careful handling of stale sockets.

---

### Day 13 — Mon, Aug 31
**Tasks:**
- End-to-end integration testing of multiplayer; fixed frontend–backend connection configuration (`api.js`, server config).
- Added an automated full-game multiplayer test (`test_multiplayer_full.py`).

**Notes:** Automated smoke tests for the socket flow caught two race conditions immediately.

---

### Day 14 — Tue, Sep 1
**Tasks:**
- Config cleanup: one-command backend start script (`start_backend.bat`), lobby and app config changes.
- Organized the RL code into its own package.

**Notes:** Repo is now cleanly split: game engine, AI strategies, RL training, multiplayer.

---

### Day 15 — Wed, Sep 2
**Tasks:**
- Fixed human-vs-human bugs (turn synchronisation, game-over handling) and sped up gameplay updates.
- Improved ship placement logic and the attack strategy of the baseline AIs (base / probability / hunt-target) and the placement UI.

**Notes:** Fast, snappy turns make the multiplayer mode feel completely different.

---

### Day 16 — Thu, Sep 3
**Tasks:**
- Final training round with the improved reward system; saved the best model (`battleship_dqn_best.zip`, ~54 MB).
- Built the tournament runner (`tournament.py`): round-robin between all AI strategies (Random, Hunt & Target, Probability, Neural).

**Notes:** The neural agent with the heatmap-prior observation beats the strongest heuristic AI on shot efficiency.

---

### Day 17 — Fri, Sep 4
**Tasks:**
- Created the report generator (`report.py`): exports the AI-vs-AI tournament results to Excel and HTML (`ai_vs_ai_results.xlsx` / `.html`) with win rates, accuracy and shots-to-win stats.
- Fixed accuracy calculation and stats display; final testing and documentation of results.

**Notes:** Project milestone: game engine + 4 AI strategies (including a trained DQN) + multiplayer + automated evaluation report, all working end-to-end.
