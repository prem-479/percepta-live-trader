import axios from "axios";

const API_BASE = "http://127.0.0.1:8000";

export async function getLiveSignals(symbol = "AAPL") {
  const res = await axios.get(`${API_BASE}/signals/live?symbol=${symbol}`);
  return res.data;
}
