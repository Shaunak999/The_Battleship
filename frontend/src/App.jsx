import { useState } from "react";
import Home from "./pages/Home";
import Game from "./pages/Game";
import WatchAiBattle from "./pages/WatchAiBattle";
import MultiplayerLobby from "./pages/MultiplayerLobby";
import MultiplayerGame from "./pages/MultiplayerGame";
import SpectatorView from "./pages/SpectatorView";

export default function App() {
  // Detect ?join=GAMEID in the URL so the friend's link auto-fills the Join tab.
  const urlParams = new URLSearchParams(window.location.search);
  const joinCode = urlParams.get("join")?.toUpperCase() || null;

  const [mode, setMode] = useState(joinCode ? "lan_multiplayer" : null);
  const [mpConfig, setMpConfig] = useState(null);

  if (!mode) {
    return <Home onSelectMode={setMode} />;
  }

  if (mode === "watch_ai") {
    return <WatchAiBattle onExit={() => setMode(null)} />;
  }

  if (mode === "lan_multiplayer") {
    if (!mpConfig) {
      return (
        <MultiplayerLobby
          onExit={() => { setMode(null); window.history.replaceState({}, "", "/"); }}
          onJoin={(config) => setMpConfig(config)}
          initialTab={joinCode ? "join" : "create"}
          initialGameId={joinCode || ""}
        />
      );
    }

    if (mpConfig.role === "spectator") {
      return (
        <SpectatorView
          gameId={mpConfig.gameId}
          onExit={() => { setMode(null); setMpConfig(null); }}
        />
      );
    }

    return (
      <MultiplayerGame
        gameId={mpConfig.gameId}
        role={mpConfig.role}
        playerName={mpConfig.playerName}
        onExit={() => { setMode(null); setMpConfig(null); }}
      />
    );
  }

  return <Game mode={mode} onExit={() => setMode(null)} />;
}
