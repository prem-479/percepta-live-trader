import { Link } from "react-router-dom";

export default function Navbar() {
  return (
    <nav style={{ background: "#111", padding: "10px", color: "white" }}>
      <Link to="/" style={{ marginRight: "20px" }}>Dashboard</Link>
      <Link to="/signals" style={{ marginRight: "20px" }}>Signals</Link>
      <Link to="/news" style={{ marginRight: "20px" }}>News</Link>
      <Link to="/risk" style={{ marginRight: "20px" }}>Risk</Link>
      <Link to="/coach" style={{ marginRight: "20px" }}>Coach</Link>
      <Link to="/portfolio">Portfolio</Link>
    </nav>
  );
}
