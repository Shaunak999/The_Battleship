export default function Home({ onSelectMode }) {
  return (
    <div className="home-hero">
      <h1>BATTLESHIP</h1>

      <div className="home-actions">
        <button className="btn" onClick={() => onSelectMode("lan_multiplayer")}>
          Human vs Human
        </button>
        <button className="btn" onClick={() => onSelectMode("human_vs_ai")}>
          Human vs Computer
        </button>
        <button className="btn" onClick={() => onSelectMode("watch_ai")}>
          Computer vs Computer
        </button>
      </div>
    </div>
  );
}
