import LiveSignals from "./pages/LiveSignals";

function App() {
  return (
    <div style={{ padding: "40px", background: "#121212", minHeight: "100vh" }}>
      <h1 style={{ color: "white" }}>Percepta Live Trading Dashboard</h1>
      <LiveSignals />
    </div>
  );
}

export default App;
