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

export async function uploadPdf(file) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await axios.post(`${API_BASE}/api/upload`, formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data; // { message, filename }
}

export async function resetIndex() {
  const res = await axios.post(`${API_BASE}/api/reset-index`);
  return res.data; // { message, vector_store_docs }
}