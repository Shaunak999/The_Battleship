export default function Home({ onSelectMode }) {
  return (
    <div className="home-hero">
      <h1>BATTLESHIP</h1>
      <p style={{ color: "var(--color-text-muted)" }}>
        Blue is your fleet. Red marks a hit.
      </p>

      <div className="home-actions">
        <button className="btn" onClick={() => onSelectMode("human_vs_human")}>
          Human vs Human
        </button>
        <button className="btn" onClick={() => onSelectMode("human_vs_ai")}>
          Human vs AI
        </button>
      </div>
    </div>
  );
}