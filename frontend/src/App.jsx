import { useState, useEffect } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, CartesianGrid, ResponsiveContainer } from "recharts";
import "./App.css";

const API = "http://127.0.0.1:8000";
const SAMPLE = "public void serialize(LittleEndianOutput out) {out.writeShort(field_1_vcenter);}";

function ResultCard({ r }) {
  return (
    <div className="card">
      <div className="card-head">
        <strong>{r.model}</strong>
        <span className={r.final_valid ? "badge ok" : "badge bad"}>
          {r.final_valid ? "Valid C# syntax" : "Syntax errors"}
        </span>
      </div>
      <pre>{r.translation}</pre>
      <div className="meta">
        {r.seconds}s
        {r.retried && (r.fixed_by_retry ? " · fixed by retry loop" : " · retry could not fix it")}
      </div>
      {r.retried && r.fixed_by_retry && (
        <details><summary>First attempt (had syntax errors)</summary><pre>{r.first_attempt}</pre></details>
      )}
    </div>
  );
}

function Translator() {
  const [code, setCode] = useState(SAMPLE);
  const [models, setModels] = useState([]);
  const [model, setModel] = useState("codet5-base");
  const [retry, setRetry] = useState(true);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch(`${API}/models`).then(r => r.json()).then(d => {
      setModels(d.available);
      if (d.available.length && !d.available.includes("codet5-base")) setModel(d.available[0]);
    }).catch(() => setError("Cannot reach the backend. Is the server running on port 8000?"));
  }, []);

  async function run(all) {
    setLoading(true); setError(""); setResults([]);
    try {
      const res = await fetch(`${API}/${all ? "compare" : "translate"}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code, model, retry }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Request failed");
      setResults(all ? data.results : [data]);
    } catch (e) { setError(e.message); }
    setLoading(false);
  }

  return (
    <div>
      <textarea value={code} onChange={e => setCode(e.target.value)} placeholder="Paste a Java method here..." />
      <div className="controls">
        <select value={model} onChange={e => setModel(e.target.value)}>
          {models.map(m => <option key={m} value={m}>{m}</option>)}
        </select>
        <label><input type="checkbox" checked={retry} onChange={e => setRetry(e.target.checked)} /> Retry if syntax is invalid</label>
        <button disabled={loading} onClick={() => run(false)}>Translate</button>
        <button className="secondary" disabled={loading} onClick={() => run(true)}>Compare all models</button>
        {loading && <span>Working... (the first run of each model is slower)</span>}
      </div>
      {error && <div className="error">{error}</div>}
      <div className="grid">{results.map(r => <ResultCard key={r.model} r={r} />)}</div>
    </div>
  );
}

function Results() {
  const [rows, setRows] = useState([]);
  const [error, setError] = useState("");
  useEffect(() => {
    fetch(`${API}/results`).then(r => r.json()).then(setRows)
      .catch(() => setError("Cannot reach the backend."));
  }, []);
  const f = v => (v === null || v === undefined ? "-" : v);
  return (
    <div>
      {error && <div className="error">{error}</div>}
      <table>
        <thead><tr><th>Model</th><th>BLEU</th><th>Exact match %</th><th>CodeBLEU</th><th>Valid syntax %</th></tr></thead>
        <tbody>
          {rows.map(r => (
            <tr key={r.model}><td>{r.model}</td><td>{f(r.BLEU)}</td><td>{f(r.ExactMatch)}</td><td>{f(r.CodeBLEU)}</td><td>{f(r.valid_syntax)}</td></tr>
          ))}
        </tbody>
      </table>
      <div className="chart">
        <ResponsiveContainer width="100%" height={340}>
          <BarChart data={rows}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="model" />
            <YAxis domain={[40, 100]} />
            <Tooltip />
            <Legend />
            <Bar dataKey="BLEU" fill="#2f6fed" />
            <Bar dataKey="CodeBLEU" fill="#11a36a" />
            <Bar dataKey="ExactMatch" name="Exact match" fill="#f0a030" />
            <Bar dataKey="valid_syntax" name="Valid syntax" fill="#9a5bd6" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function About() {
  return (
    <div className="card">
      <h3>About this project</h3>
      <p>This system translates Java methods to C# with encoder-decoder Transformer models fine-tuned on the CodeXGLUE Java-C# dataset (10,300 training pairs, 1,000 test pairs).</p>
      <p>Models compared: CodeT5-small, CodeT5-base and PLBART-base. Metrics: BLEU (sacreBLEU), exact match, CodeBLEU and syntactic validity (tree-sitter).</p>
      <p>A retry step regenerates up to 10 beam-search candidates when the first output is not valid C# syntax and keeps the first valid one. Valid syntax does not guarantee correct behaviour.</p>
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState("translate");
  return (
    <div className="app">
      <h1>Java to C# Code Translator</h1>
      <div className="sub">Compare Transformer models for automated code translation</div>
      <div className="tabs">
        {[["translate", "Translator"], ["results", "Results"], ["about", "About"]].map(([k, label]) => (
          <button key={k} className={tab === k ? "active" : ""} onClick={() => setTab(k)}>{label}</button>
        ))}
      </div>
      {tab === "translate" && <Translator />}
      {tab === "results" && <Results />}
      {tab === "about" && <About />}
    </div>
  );
}