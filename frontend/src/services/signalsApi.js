import axios from "axios";

// Use VITE env variable so it works on localhost + production
const API_BASE = import.meta.env.VITE_BACKEND_URL || "http://127.0.0.1:8000";

// === GET PREDICTION ===
export async function fetchPrediction(symbol) {
  try {
    const response = await axios.get(`${API_BASE}/predict`, {
      params: { symbol },
    });
    return response.data;
  } catch (error) {
    console.error("❌ Error fetching prediction:", error);
    return null;
  }
}

// === GET RISK ANALYSIS ===
export async function fetchRisk(symbol) {
  try {
    const response = await axios.get(`${API_BASE}/risk`, {
      params: { symbol },
    });
    return response.data;
  } catch (error) {
    console.error("❌ Error fetching risk:", error);
    return null;
  }
}

// === GET BACKEND HEALTH ===
export async function checkBackend() {
  try {
    const response = await axios.get(`${API_BASE}/health`);
    return response.data;
  } catch (error) {
    console.error("❌ Backend offline:", error);
    return null;
  }
}
