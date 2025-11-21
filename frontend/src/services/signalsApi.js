import axios from "axios";

const API_BASE = "http://127.0.0.1:8000";

export async function fetchLiveSignals(symbol) {
  try {
    const response = await axios.get(`${API_BASE}/signals/live`, {
      params: { symbol },
    });
    return response.data;
  } catch (error) {
    console.error("Error fetching live signals:", error);
    return null;
  }
}
