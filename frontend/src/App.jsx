import { useState } from "react";
import Home from "./pages/Home";
import Game from "./pages/Game";
import WatchAiBattle from "./pages/WatchAiBattle";

export default function App() {
  const [mode, setMode] = useState(null); // null | "human_vs_human" | "human_vs_ai" | "watch_ai"

  if (!mode) {
    return <Home onSelectMode={setMode} />;
  }

  if (mode === "watch_ai") {
    return <WatchAiBattle onExit={() => setMode(null)} />;
  }

  return <Game mode={mode} onExit={() => setMode(null)} />;
}