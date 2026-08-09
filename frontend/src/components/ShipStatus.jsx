const ALL_SHIPS = [
  { name: "Carrier", size: 5 },
  { name: "Battleship", size: 4 },
  { name: "Cruiser", size: 3 },
  { name: "Submarine", size: 3 },
  { name: "Destroyer", size: 2 },
];

/**
 * remainingShips: string[] of ship names still afloat, as returned by
 * the backend's remaining_ships(). A ship not in this list is sunk.
 */
export default function ShipStatus({ title, remainingShips }) {
  return (
    <div className="card" style={{ minWidth: 180 }}>
      <h3>{title}</h3>
      <ul className="ship-status-list">
        {ALL_SHIPS.map((ship) => {
          const sunk = !remainingShips.includes(ship.name);
          return (
            <li key={ship.name} className={sunk ? "sunk" : ""}>
              <span>{ship.name}</span>
              <span>{ship.size}</span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}