// src/App.jsx
import React, { useEffect, useState } from "react";
import { health, queryRag, uploadPdf } from "./api";
import "./App.css";

function App() {
  const [healthStatus, setHealthStatus] = useState("Connecting…");
  const [healthDocs, setHealthDocs] = useState(null);

  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [sending, setSending] = useState(false);
  const [topK, setTopK] = useState(5);
  const [topScore, setTopScore] = useState(null);

  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState("");

  // Health check once
  useEffect(() => {
    (async () => {
      try {
        const data = await health();
        setHealthStatus(`✓ ${data.vector_store_docs} docs`);
        setHealthDocs(data.vector_store_docs);
      } catch {
        setHealthStatus("Offline");
      }
    })();
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
      addMessage("assistant", `⚠️ Error: ${e.message || "Request failed"}`);
    } finally {
      setSending(false);
    }
  };

  const exampleQuestions = [
    "What are the main risk factors mentioned in this 10-K?",
    "How does the company describe its competitive strengths in this filing?",
    "What does the 10-K say about regulatory or legal risks?",
    "Summarise the company's liquidity and capital resources position described in this 10-K.",
  ];

  return (
    <div className="app">
      <header className="header">
        <div className="logo">
          <span className="logo-icon">📄</span>
          <div>
            <h1>10-K Filing Assistant</h1>
            <p className="tagline">
              Ask questions · Get cited answers · Powered by RAG
            </p>
          </div>
        </div>
        <div className="health">
          {healthStatus}
          {healthDocs !== null && <span> · {healthDocs} docs</span>}
        </div>
      </header>

      <main className="chat">
            <section className="upload-panel">
              <h2>Upload a 10-K PDF</h2>
              <p className="upload-help">
                Choose a filing PDF to index. The server will parse and add it to
                the vector store; this may take a minute for large documents.
              </p>
              <div className="upload-controls">
                <input
                  type="file"
                  accept="application/pdf"
                  onChange={(e) => {
                    const f = e.target.files?.[0] ?? null;
                    setSelectedFile(f);
                  }}
                />
                <button
                  onClick={async () => {
                    if (!selectedFile) {
                      setUploadStatus("Please select a PDF first.");
                      return;
                    }
                    try {
                      setUploadStatus("Uploading and starting ingestion…");
                      const res = await uploadPdf(selectedFile);
                      setUploadStatus(res.message || "Upload started.");
                    } catch (e) {
                      setUploadStatus(
                        `Upload failed: ${e.response?.data?.detail || e.message}`
                      );
                    }
                  }}
                >
                  Upload & Index
                </button>
              </div>
              {uploadStatus && (
                <p className="upload-status">{uploadStatus}</p>
              )}
            </section>

          <div className="chat-messages">
            {messages.length === 0 && (
              <div className="welcome-card">
                <h2>Ask me anything about the 10-K</h2>
                <p>
                  I retrieve relevant sections from the filing, then generate a
                  cited answer.
                </p>
                <p className="examples-label">Try asking:</p>
                <div className="examples">
                  {exampleQuestions.map((q) => (
                    <button
                      key={q}
                      className="example-btn"
                      onClick={() => handleSend(q)}
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((m, i) => (
              <MessageBubble key={i} message={m} />
            ))}
          </div>

          <div className="chat-input">
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask a question about the 10-K filing…"
              rows={2}
              onKeyDown={(e) => {
                if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
                  e.preventDefault();
                  handleSend();
                }
              }}
            />
            <div className="input-row">
              <button onClick={() => handleSend()} disabled={sending}>
                {sending ? "Sending…" : "Send"}
              </button>
              <label>
                Top-K:
                <select
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                >
                  <option value={3}>3</option>
                  <option value={5}>5</option>
                  <option value={8}>8</option>
                </select>
              </label>
              {topScore !== null && (
                <span className="score-pill">
                  Top retrieval score: {topScore}
                </span>
              )}
            </div>
          </div>
      </main>
    </div>
  );
}

function MessageBubble({ message }) {
  const { role, text, sources = [] } = message;
  const isUser = role === "user";

  return (
    <div className={`message ${isUser ? "user" : "assistant"}`}>
      <div className="bubble">{text}</div>
      {!isUser && sources.length > 0 && (
        <details className="sources">
          <summary>{sources.length} source(s) used</summary>
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
  );
}

export default App;