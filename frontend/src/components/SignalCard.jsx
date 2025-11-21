import React from "react";

export default function SignalCard({ data }) {
  if (!data) return <p style={{ color: "white" }}>Loading...</p>;

  return (
    <div
      style={{
        padding: "20px",
        background: "#1e1e1e",
        borderRadius: "12px",
        marginTop: "20px",
        width: "350px",
        color: "white",
      }}
    >
      <h3>📈 {data.symbol} Signals</h3>
      <p><b>Last Price:</b> {data.last_price}</p>
      <p><b>Trend Signal:</b> {data.trend_signal ?? "N/A"}</p>
      <p><b>Bollinger:</b> {data.bollinger_signal}</p>
    </div>
  );
}
