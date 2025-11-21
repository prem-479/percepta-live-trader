import React, { useState } from "react";
import { fetchLiveSignals } from "../services/signalsApi";
import SignalCard from "./SignalCard";

export default function LiveSignals() {
  const [symbol, setSymbol] = useState("AAPL");
  const [data, setData] = useState(null);

  const loadData = async () => {
    const res = await fetchLiveSignals(symbol);
    setData(res);
  };

  return (
    <div style={{ marginTop: "30px", color: "white" }}>
      <h2>📡 Live Trading Signals</h2>

      <input
        value={symbol}
        onChange={(e) => setSymbol(e.target.value)}
        style={{ padding: "8px", marginRight: "10px" }}
      />
      <button onClick={loadData} style={{ padding: "8px 16px" }}>
        Fetch
      </button>

      <SignalCard data={data} />
    </div>
  );
}
