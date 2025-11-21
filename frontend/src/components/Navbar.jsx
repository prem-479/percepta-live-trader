import { Link } from "react-router-dom";

export default function Navbar() {
  return (
    <nav style={{
      padding: "12px",
      background: "#111",
      color: "white",
      display: "flex",
      gap: "20px"
    }}>
      <Link to="/" style={{ color: "white" }}>Dashboard</Link>
      <Link to="/signals" style={{ color: "white" }}>Signals</Link>
      <Link to="/news" style={{ color: "white" }}>News</Link>
      <Link to="/risk" style={{ color: "white" }}>Risk</Link>
      <Link to="/coach" style={{ color: "white" }}>Coach</Link>
      <Link to="/portfolio" style={{ color: "white" }}>Portfolio</Link>
    </nav>
  );
}
