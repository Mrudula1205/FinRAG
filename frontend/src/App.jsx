import React, { useEffect, useState } from "react";
import { health, queryRag, uploadPdf } from "./api";
import "./App.css";

function App() {
  const [healthStatus, setHealthStatus] = useState("Connecting...");
  const [healthDocs, setHealthDocs] = useState(null);

  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [sending, setSending] = useState(false);
  const [topK, setTopK] = useState(5);
  const [topScore, setTopScore] = useState(null);

  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState("");
  const [uploading, setUploading] = useState(false);

  const refreshHealth = async () => {
    try {
      const data = await health();
      setHealthStatus("Ready");
      setHealthDocs(data.vector_store_docs);
    } catch {
      setHealthStatus("Offline");
    }
  };

  useEffect(() => {
    refreshHealth();
  }, []);

  const addMessage = (role, text, extra = {}) => {
    setMessages((prev) => [...prev, { role, text, ...extra }]);
  };

  const handleSend = async (q) => {
    const trimmed = (q ?? question).trim();
    if (!trimmed || sending) return;

    setQuestion("");
    setSending(true);
    setTopScore(null);
    addMessage("user", trimmed);

    try {
      const data = await queryRag({ question: trimmed, topK });
      addMessage("assistant", data.answer, { sources: data.sources || [] });
      setTopScore(data.top_retrieval_score);
    } catch (e) {
      addMessage("assistant", `Request failed: ${e.message || "Unknown error"}`);
    } finally {
      setSending(false);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setUploadStatus("Select a PDF first.");
      return;
    }
    try {
      setUploading(true);
      setUploadStatus("Uploading file and starting ingestion...");
      const res = await uploadPdf(selectedFile);
      setUploadStatus(res.message || "Upload accepted.");
    } catch (e) {
      setUploadStatus(`Upload failed: ${e.response?.data?.detail || e.message}`);
    } finally {
      setUploading(false);
    }
  };

  const exampleQuestions = [
    "What are the primary risk factors in this 10-K?",
    "How does management describe liquidity and capital resources?",
    "What legal or regulatory exposure is discussed?",
    "What does the filing say about competition and market position?",
  ];

  return (
    <div className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Financial RAG Workspace</p>
          <h1>10-K Filing Assistant</h1>
          <p className="hero-subtitle">
            Upload filings, index them, and ask grounded questions with cited passages.
          </p>
        </div>
        <div className="status-cluster">
          <div className={`status-pill ${healthStatus === "Ready" ? "ok" : "warn"}`}>
            {healthStatus}
            {healthDocs !== null && `  ${healthDocs} docs`}
          </div>
          <button className="ghost-btn" onClick={refreshHealth}>
            Refresh
          </button>
        </div>
      </header>

      <main className="workspace">
        <aside className="panel left-panel">
          <section className="card">
            <h2>1. Upload Filing</h2>
            <p className="muted">
              Choose a 10-K PDF and index it into the vector database.
            </p>
            <label className="file-picker">
              <span>Choose PDF</span>
              <input
                type="file"
                accept="application/pdf"
                onChange={(e) => setSelectedFile(e.target.files?.[0] ?? null)}
              />
            </label>
            <p className="filename">{selectedFile?.name || "No file selected"}</p>
            <button className="primary-btn" onClick={handleUpload} disabled={uploading}>
              {uploading ? "Uploading..." : "Upload and Index"}
            </button>
            {uploadStatus && <p className="upload-status">{uploadStatus}</p>}
          </section>

          <section className="card">
            <h2>2. Retrieval Settings</h2>
            <div className="setting-row">
              <label htmlFor="top-k">Top-K</label>
              <select
                id="top-k"
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
              >
                <option value={3}>3</option>
                <option value={5}>5</option>
                <option value={8}>8</option>
              </select>
            </div>
            <p className="muted">
              Lower Top-K keeps prompts smaller and reduces token-limit errors.
            </p>
            {topScore !== null && (
              <div className="metric-box">Top retrieval score: {topScore}</div>
            )}
          </section>

          <section className="card">
            <h2>Prompt Starters</h2>
            <div className="prompt-list">
              {exampleQuestions.map((q) => (
                <button key={q} className="prompt-btn" onClick={() => handleSend(q)}>
                  {q}
                </button>
              ))}
            </div>
          </section>
        </aside>

        <section className="panel chat-panel">
          <div className="chat-header">
            <h2>Ask Questions</h2>
            <p className="muted">Answers are generated from retrieved filing chunks.</p>
          </div>

          <div className="chat-messages">
            {messages.length === 0 ? (
              <div className="empty-state">
                <p>No conversation yet.</p>
                <p className="muted">Upload a document and ask your first question.</p>
              </div>
            ) : (
              messages.map((m, i) => <MessageBubble key={i} message={m} />)
            )}
          </div>

          <div className="composer">
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask about risks, liquidity, operations, legal exposure, or strategy..."
              rows={3}
              onKeyDown={(e) => {
                if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
                  e.preventDefault();
                  handleSend();
                }
              }}
            />
            <div className="composer-row">
              <button className="primary-btn" onClick={() => handleSend()} disabled={sending}>
                {sending ? "Generating..." : "Ask"}
              </button>
              <span className="hint">Ctrl/Cmd + Enter to send</span>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

function MessageBubble({ message }) {
  const { role, text, sources = [] } = message;
  const isUser = role === "user";

  return (
    <div className={`message ${isUser ? "user" : "assistant"}`}>
      <div className="bubble-wrap">
        <p className="message-role">{isUser ? "You" : "Assistant"}</p>
        <div className="bubble">{text}</div>
        {!isUser && sources.length > 0 && (
          <details className="sources">
            <summary>{sources.length} source(s)</summary>
            {sources.map((s, i) => (
              <div key={i} className="source-card">
                <div className="source-header">
                  <span className="source-ref">
                    {s.section && s.section !== "N/A" ? s.section : s.source}
                  </span>
                  <span className="source-score">score {s.score}</span>
                </div>
                <div className="source-preview">{s.preview}</div>
              </div>
            ))}
          </details>
        )}
      </div>
    </div>
  );
}

export default App;