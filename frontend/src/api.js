// src/api.js
import axios from "axios";

const API_BASE = "http://localhost:8000";

export async function health() {
  const res = await axios.get(`${API_BASE}/health`);
  return res.data;
}

export async function queryRag({ question, topK }) {
  const res = await axios.post(`${API_BASE}/api/query`, {
    question,
    top_k: topK,
  });
  return res.data; // { answer, top_retrieval_score, sources[] }
}

export async function fetchEvalResults() {
  const res = await axios.get(`${API_BASE}/api/eval-results`);
  return res.data; // { results, summary }
}