import { LogicTokenizer, bucketOf, type Bucket } from "./logic-tokenizer";
import { LogicModel, type TaskType } from "./model";
import { humanizeSegment, humanizeAnswer } from "./humanize";

const BUCKET_LABEL: Record<Bucket, string> = {
  SPECIAL: "special",
  L: "logic",
  Q: "quantifier",
  R: "relation",
  T: "time",
  M: "modal",
  RO: "role",
  C: "concept",
  A: "arithmetic",
  S: "structure",
  V: "variable",
  N: "number",
};

interface Example {
  label: string;
  text: string;
}

// NOTE: text formats below mirror the training corpus
// (reasoning/data/generators/prop_logic.py + syllogisms.py + algebra_engine).
// Changing the wording (e.g. "T" instead of "true") moves the input out of
// distribution and the model will under-perform.
const EXAMPLES: Record<TaskType, Example[]> = {
  prop: [
    { label: "Evaluate q", text: "Given p=true, q=true, r=false, evaluate: q" },
    { label: "Conjunction", text: "Given p=false, q=false, evaluate: (q and p)" },
    { label: "Implication T", text: "Given p=true, q=true, evaluate: (p implies q)" },
    { label: "Implication F", text: "Given p=true, q=false, evaluate: (p implies q)" },
    { label: "Negation", text: "Given p=true, q=false, evaluate: (not (p and q))" },
  ],
  syllogism: [
    {
      label: "Barbara (valid)",
      text: "All engineers are professionals. All programmers are engineers. Does it follow that: All programmers are professionals.",
    },
    {
      label: "Celarent (valid)",
      text: "No animals are living things. All birds are animals. Does it follow that: No birds are living things.",
    },
    {
      label: "Darii (valid)",
      text: "All doctors are educated. Some adults are doctors. Does it follow that: Some adults are educated.",
    },
    {
      label: "Illicit minor",
      text: "All cats are animals. All cats are mammals. Does it follow that: All mammals are animals.",
    },
  ],
  algebra: [
    { label: "Root + pattern A", text: "do the write" },
    { label: "Root + pattern B", text: "Please do send" },
    { label: "Root + pattern C", text: "do the read" },
    { label: "Root + pattern D", text: "do the teach" },
  ],
};

const TASK_LABEL: Record<TaskType, string> = {
  prop: "Prop logic",
  syllogism: "Syllogism",
  algebra: "Algebra",
};

let tk: LogicTokenizer;
let model: LogicModel;
let activeTask: TaskType = "prop";

async function boot() {
  tk = await LogicTokenizer.load();
  model = new LogicModel(tk);
  await model.init();
  renderVocabBadge();
  renderTabs();
  renderExamples();
  bindInput();
  run(); // initial example
}

function renderVocabBadge() {
  const el = document.getElementById("vocab-badge")!;
  el.textContent = `${tk.vocab.size} tokens`;
}

function renderTabs() {
  const bar = document.getElementById("tabs")!;
  bar.innerHTML = "";
  (["prop", "syllogism", "algebra"] as TaskType[]).forEach((task) => {
    const btn = document.createElement("button");
    btn.className = "tab" + (task === activeTask ? " active" : "");
    btn.textContent = TASK_LABEL[task];
    btn.onclick = () => {
      activeTask = task;
      renderTabs();
      renderExamples();
      (document.getElementById("input") as HTMLTextAreaElement).value = EXAMPLES[task][0].text;
      run();
    };
    bar.appendChild(btn);
  });
}

function renderExamples() {
  const el = document.getElementById("examples")!;
  el.innerHTML = "";
  for (const ex of EXAMPLES[activeTask]) {
    const chip = document.createElement("button");
    chip.className = "chip";
    chip.textContent = ex.label;
    chip.onclick = () => {
      (document.getElementById("input") as HTMLTextAreaElement).value = ex.text;
      run();
    };
    el.appendChild(chip);
  }
  const note = document.getElementById("task-note")!;
  if (activeTask === "algebra") {
    note.textContent =
      "⚠ Experimental: the 151-token logic-v0.1 vocab collapses all semantic roots/resources/actions to C:CONCEPT, so answers will look like structured CST shapes with repeated C:CONCEPT slots. Shape is learned; content is not.";
    note.style.display = "block";
  } else {
    note.textContent = "";
    note.style.display = "none";
  }
}

function bindInput() {
  const ta = document.getElementById("input") as HTMLTextAreaElement;
  ta.value = EXAMPLES[activeTask][0].text;
  document.getElementById("run-btn")!.addEventListener("click", run);
  ta.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") run();
  });
}

let busy = false;
function setBusy(b: boolean) {
  busy = b;
  const btn = document.getElementById("run-btn") as HTMLButtonElement;
  btn.disabled = b;
  btn.textContent = b ? "Running…" : "Run";
  document.body.style.cursor = b ? "progress" : "";
  document.querySelectorAll<HTMLButtonElement>(".tab, .chip").forEach((el) => (el.disabled = b));
}

async function run() {
  if (busy) return;
  const text = (document.getElementById("input") as HTMLTextAreaElement).value;
  if (!text.trim()) {
    renderAll([], null, "");
    return;
  }
  setBusy(true);
  // Yield to the browser so the disabled/label change paints before
  // the synchronous ONNX-WASM work blocks the main thread.
  await new Promise((r) => setTimeout(r, 0));
  try {
    const toks = tk.fromFormal(text, true);
    const result = await model.infer(activeTask, text);
    renderAll(toks, result.answer, result.backend, result.elapsedMs, result.cot);
  } finally {
    setBusy(false);
  }
}

function tokenChip(t: string): HTMLElement {
  const el = document.createElement("span");
  const b = bucketOf(t);
  el.className = `tok b-${b}`;
  el.textContent = t;
  el.title = BUCKET_LABEL[b];
  return el;
}

function renderAll(
  toks: string[],
  answer: string | null,
  backend: string,
  elapsed = 0,
  cot: string[][] = [],
) {
  const toksEl = document.getElementById("tokens")!;
  toksEl.innerHTML = "";
  for (const t of toks) toksEl.appendChild(tokenChip(t));

  const idsEl = document.getElementById("ids")!;
  idsEl.textContent = tk.toIds(toks).join("  ");

  const ansEl = document.getElementById("answer")!;
  ansEl.textContent = answer ?? "—";
  ansEl.className =
    "answer " +
    (answer === "true" || answer === "valid"
      ? "ok"
      : answer === "false" || answer === "invalid"
        ? "bad"
        : answer === "unknown" || answer === "undetermined"
          ? "unk"
          : "eq");

  const glossEl = document.getElementById("answer-gloss")!;
  glossEl.textContent = answer === null ? "" : humanizeAnswer(activeTask, answer);

  document.getElementById("meta")!.textContent =
    answer === null ? "" : `${backend} · ${elapsed.toFixed(1)} ms · ${toks.length} tokens`;

  // Human-readable CoT (deduplicated, final step highlighted).
  const humanEl = document.getElementById("cot-human")!;
  humanEl.innerHTML = "";
  const phrases: string[] = [];
  for (const step of cot) {
    const p = humanizeSegment(step);
    if (!p) continue;
    if (phrases.length && phrases[phrases.length - 1] === p) continue; // collapse repeats
    phrases.push(p);
  }
  if (!phrases.length) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "—";
    humanEl.appendChild(li);
  } else {
    phrases.forEach((p, i) => {
      const li = document.createElement("li");
      if (i === phrases.length - 1) li.className = "final";
      li.textContent = p;
      humanEl.appendChild(li);
    });
  }

  const cotEl = document.getElementById("cot")!;
  cotEl.innerHTML = "";
  for (const step of cot) {
    const block = document.createElement("div");
    block.className = "cot-block";

    const row = document.createElement("div");
    row.className = "cot-row";
    for (const t of step) row.appendChild(tokenChip(t));
    block.appendChild(row);

    const gloss = document.createElement("div");
    gloss.className = "cot-gloss";
    const human = humanizeSegment(step);
    gloss.textContent = human ? `≈ ${human}` : "";
    if (human) block.appendChild(gloss);

    cotEl.appendChild(block);
  }
}

boot().catch((err) => {
  document.body.innerHTML = `<pre style="padding:24px;color:#f88">boot failed: ${err.message}</pre>`;
  console.error(err);
});
