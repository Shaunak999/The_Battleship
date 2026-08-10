import { useEffect, useState } from "react";
import { getStrategies } from "../services/api";

export default function StrategySelector({ onSelect }) {
  const [strategies, setStrategies] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;

    getStrategies()
      .then((data) => {
        if (!cancelled) setStrategies(data);
      })
      .catch((err) => {
        if (!cancelled) setError(err.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  if (loading) return <p>Loading AI strategies...</p>;

  if (error) {
    return (
      <p className="error-text">
        Couldn't reach the backend to load AI strategies: {error}
      </p>
    );
  }

  return (
    <div>
      <h3>Choose Computer Strategy</h3>
      <div className="strategy-options">
        {strategies.map((s) => (
          <button
            key={s.key}
            className="btn secondary"
            onClick={() => onSelect(s.key)}
            title={s.description}
          >
            {s.name}
          </button>
        ))}
      </div>
    </div>
  );
}