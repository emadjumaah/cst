/**
 * main.ts — CST Reasoning browser demo
 *
 * Honest scope: loads pre-tokenized held-out syllogisms (EN + AR, 8 moods)
 * and scores the 4 language-specific yes/no candidate continuations using
 * the trained ONNX model via onnxruntime-web. No NL tokenizer in the
 * browser — the model was trained on CST tokens, so the demo feeds CST
 * tokens, exactly as the Python eval does.
 */

import * as ort from "onnxruntime-web";

// ─── Types & constants ────────────────────────────────────────────
type Lang = "en" | "ar";

interface Example {
  id: string;
  lang: Lang;
  category: string;
  question: string;
  answer: "yes" | "no";
  prefix_tokens: string[]; // [BOS] q [EOS] [BOS] cot1 [EOS] ... [BOS] cotN [EOS]
}

interface ModelEntry {
  label: string;
  file: string;
  params_M: number;
  max_len: number;
  session: ort.InferenceSession | null;
}

const MODELS: Record<"fast" | "accurate", ModelEntry> = {
  fast: {
    label: "Fast",
    file: "model_logic_small.onnx",
    params_M: 2.05,
    max_len: 128,
    session: null,
  },
  accurate: {
    label: "Accurate",
    file: "model_logic_large.onnx",
    params_M: 11.1,
    max_len: 256,
    session: null,
  },
};

const CAND_TOKENS: Record<Lang, { yes: string[]; no: string[] }> = {
  en: { yes: ["[BOS]", "LIT:yes", "[EOS]"], no: ["[BOS]", "REL:neg", "[EOS]"] },
  ar: { yes: ["[BOS]", "ROOT:ن.ع.م", "[EOS]"], no: ["[BOS]", "STR:neg:general", "[EOS]"] },
};

// ─── State ────────────────────────────────────────────────────────
let vocab: Map<string, number> = new Map();
let PAD = 0;
let examples: Example[] = [];
let filtered: Example[] = [];
let current: Example | null = null;
let activeMode: "fast" | "accurate" = "fast";

// ─── Setup ort runtime ────────────────────────────────────────────
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/";
ort.env.wasm.numThreads = 1;

// ─── Boot ─────────────────────────────────────────────────────────
async function boot() {
  setStatus("loading vocab + examples…");
  const [vj, ej] = await Promise.all([
    fetch("./vocab-reasoning.json").then((r) => r.json() as Promise<Record<string, number>>),
    fetch("./examples.json").then((r) => r.json() as Promise<Example[]>),
  ]);
  vocab = new Map(Object.entries(vj).map(([k, v]) => [k, v as number]));
  PAD = vocab.get("[PAD]") ?? 0;
  examples = ej;
  populateFilters();
  applyFilters();

  setStatus("loading fast model (2.4 MB)…");
  try {
    MODELS.fast.session = await ort.InferenceSession.create(`./${MODELS.fast.file}`, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    setStatus(`ready · vocab ${vocab.size} · ${examples.length} examples`, "ok");
  } catch (e) {
    console.error(e);
    setStatus(`failed to load model: ${(e as Error).message}`, "err");
    return;
  }

  bindUI();
  nextExample();
}

// ─── UI ───────────────────────────────────────────────────────────
function $(id: string): HTMLElement {
  return document.getElementById(id)!;
}

function setStatus(msg: string, cls: "" | "ok" | "err" = "") {
  const el = $("status");
  el.textContent = msg;
  el.className = "status " + cls;
}

function populateFilters() {
  const moods = Array.from(new Set(examples.map((e) => e.category))).sort();
  const sel = $("mood-sel") as HTMLSelectElement;
  for (const m of moods) {
    const o = document.createElement("option");
    o.value = m;
    o.textContent = m;
    sel.appendChild(o);
  }
}

function applyFilters() {
  const lang = ($("lang-sel") as HTMLSelectElement).value;
  const mood = ($("mood-sel") as HTMLSelectElement).value;
  filtered = examples.filter(
    (e) => (lang === "all" || e.lang === lang) && (mood === "all" || e.category === mood),
  );
  if (!filtered.length) filtered = examples;
}

function nextExample() {
  if (!filtered.length) return;
  current = filtered[Math.floor(Math.random() * filtered.length)];
  renderExample(current);
  clearResult();
}

function renderExample(ex: Example) {
  const q = $("question");
  q.textContent = ex.question;
  q.className = "prose" + (ex.lang === "ar" ? " ar" : "");
  const g = $("gold");
  g.textContent = ex.answer;
  g.className = "badge " + (ex.answer === "yes" ? "yes" : "no");
  // prefix token chips
  const view = $("prefix-view");
  view.innerHTML = "";
  for (const t of ex.prefix_tokens) {
    const s = document.createElement("span");
    s.className = "tok" + (t === "[BOS]" ? " bos" : t === "[EOS]" ? " eos" : "");
    s.textContent = t;
    view.appendChild(s);
  }
}

function clearResult() {
  $("pred").textContent = "—";
  $("pred").className = "badge big";
  $("scores").querySelector("tbody")!.innerHTML = "";
  $("meta").textContent = "";
}

function bindUI() {
  ($("next-btn") as HTMLButtonElement).onclick = nextExample;
  ($("run-btn") as HTMLButtonElement).onclick = run;
  for (const sel of ["lang-sel", "mood-sel"]) {
    ($(sel) as HTMLSelectElement).onchange = () => {
      applyFilters();
      nextExample();
    };
  }
  ($("mode-fast") as HTMLButtonElement).onclick = () => switchMode("fast");
  ($("mode-accurate") as HTMLButtonElement).onclick = () => switchMode("accurate");
}

async function switchMode(m: "fast" | "accurate") {
  if (m === activeMode) return;
  activeMode = m;
  $("mode-fast").classList.toggle("active", m === "fast");
  $("mode-accurate").classList.toggle("active", m === "accurate");
  const entry = MODELS[m];
  if (!entry.session) {
    setStatus(`loading ${entry.label.toLowerCase()} model…`);
    disableButtons(true);
    try {
      entry.session = await ort.InferenceSession.create(`./${entry.file}`, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
      setStatus(`${entry.label} ready · ${entry.params_M} M params`, "ok");
    } catch (e) {
      setStatus(`failed: ${(e as Error).message}`, "err");
    } finally {
      disableButtons(false);
    }
  } else {
    setStatus(`${entry.label} active · ${entry.params_M} M params`, "ok");
  }
}

function disableButtons(b: boolean) {
  for (const id of ["mode-fast", "mode-accurate", "run-btn", "next-btn"]) {
    ($(id) as HTMLButtonElement).disabled = b;
  }
}

// ─── Inference ────────────────────────────────────────────────────
function toIds(toks: string[]): number[] {
  const unk = vocab.get("[UNK]") ?? 1;
  return toks.map((t) => vocab.get(t) ?? unk);
}

async function scoreCandidate(
  sess: ort.InferenceSession,
  prefixIds: number[],
  candIds: number[],
  maxLen: number,
): Promise<number> {
  let seq = prefixIds.concat(candIds);
  if (seq.length > maxLen) seq = seq.slice(-maxLen);
  const n = Math.min(candIds.length, seq.length);

  const padded = new BigInt64Array(maxLen).fill(BigInt(PAD));
  for (let i = 0; i < seq.length; i++) padded[i] = BigInt(seq[i]);
  const input = new ort.Tensor("int64", padded, [1, maxLen]);
  const out = await sess.run({ [sess.inputNames[0]]: input });
  const logits = out[sess.outputNames[0]].data as Float32Array;
  const V = vocab.size;

  // log-softmax per position, then sum over candidate positions.
  const T = seq.length;
  let total = 0;
  for (let off = 0; off < n; off++) {
    const t = T - n + off;
    const prev = t - 1;
    if (prev < 0) continue;
    const base = prev * V;
    // logsumexp
    let max = -Infinity;
    for (let v = 0; v < V; v++) if (logits[base + v] > max) max = logits[base + v];
    let sumExp = 0;
    for (let v = 0; v < V; v++) sumExp += Math.exp(logits[base + v] - max);
    const lse = max + Math.log(sumExp);
    total += logits[base + seq[t]] - lse;
  }
  return total;
}

async function run() {
  if (!current) return;
  const entry = MODELS[activeMode];
  if (!entry.session) {
    setStatus("model not loaded", "err");
    return;
  }
  disableButtons(true);
  setStatus("scoring…");
  await new Promise((r) => setTimeout(r, 0));

  const t0 = performance.now();
  const prefixIds = toIds(current.prefix_tokens);
  const cands = CAND_TOKENS[current.lang];
  const [yScore, nScore] = await Promise.all([
    scoreCandidate(entry.session, prefixIds, toIds(cands.yes), entry.max_len),
    scoreCandidate(entry.session, prefixIds, toIds(cands.no), entry.max_len),
  ]);
  const elapsed = performance.now() - t0;

  const pred: "yes" | "no" = yScore > nScore ? "yes" : "no";
  const predEl = $("pred");
  predEl.textContent = pred;
  predEl.className = "badge big " + (pred === "yes" ? "yes" : "no");

  const tbody = $("scores").querySelector("tbody")!;
  tbody.innerHTML = "";
  for (const [k, score, toks] of [
    ["yes", yScore, cands.yes] as const,
    ["no", nScore, cands.no] as const,
  ]) {
    const tr = document.createElement("tr");
    tr.className = k === pred ? "winner" : "loser";
    tr.innerHTML = `<td>${k}</td><td class="tok-inline">${toks.join(" ")}</td><td>${score.toFixed(3)}</td>`;
    tbody.appendChild(tr);
  }

  const correct = pred === current.answer;
  $("meta").innerHTML =
    `<strong style="color:${correct ? "var(--ok)" : "var(--bad)"}">${correct ? "✓ correct" : "✗ wrong"}</strong> · ` +
    `${entry.label} (${entry.params_M} M) · ${elapsed.toFixed(1)} ms · margin ${Math.abs(yScore - nScore).toFixed(2)}`;
  setStatus(`${entry.label} ready`, "ok");
  disableButtons(false);
}

boot().catch((e) => {
  console.error(e);
  document.body.innerHTML = `<pre style="padding:24px;color:#f88">boot failed: ${(e as Error).message}</pre>`;
});
