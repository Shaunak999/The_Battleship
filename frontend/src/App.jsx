import { useState } from "react";
import Home from "./pages/Home";
import Game from "./pages/Game";

export default function App() {
  const [mode, setMode] = useState(null); // null | "human_vs_human" | "human_vs_ai"

  if (!mode) {
    return <Home onSelectMode={setMode} />;
  }

  return <Game mode={mode} onExit={() => setMode(null)} />;
}