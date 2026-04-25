/**
 * model.ts — inference backend.
 *
 * Tries to load the trained ONNX (edge/demo-logic/public/model_logic.onnx)
 * on init. If present, inference runs through the real 10.79M-param
 * transformer via onnxruntime-web. If not, falls back to deterministic
 * stubs so the UI still works.
 *
 * Model contract (from colab_train_reasoning):
 *   - input  "ids"    int64[batch, seq]  (seq = 256, padded with [PAD])
 *   - output "logits" float32[batch, seq, vocab_size]
 *   - trained on:  [BOS]q[EOS] [BOS]cot1[EOS] [BOS]cot2[EOS] … [BOS]ans[EOS]
 */

import type { LogicTokenizer } from "./logic-tokenizer";

export type TaskType = "prop" | "syllogism" | "algebra";

export interface InferResult {
  answer: string;
  cot: string[][];
  backend: "stub" | "onnx";
  elapsedMs: number;
}

const MAX_LEN = 256;
const MAX_NEW_TOKENS = 48;

type OrtSession = {
  run: (feeds: Record<string, unknown>) => Promise<Record<string, { data: Float32Array }>>;
};

export class LogicModel {
  private session: OrtSession | null = null;
  private ort: typeof import("onnxruntime-web") | null = null;

  constructor(private tk: LogicTokenizer) {}

  async init(): Promise<void> {
    try {
      const ort = await import("onnxruntime-web");
      this.ort = ort;
      const session = await ort.InferenceSession.create("model_logic.onnx", {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
      this.session = session as unknown as OrtSession;
      // eslint-disable-next-line no-console
      console.log("[LogicModel] ONNX loaded — trained 10.79M transformer active");
    } catch (e) {
      // eslint-disable-next-line no-console
      console.warn("[LogicModel] ONNX unavailable, using stubs:", e);
      this.session = null;
    }
  }

  hasOnnx(): boolean {
    return this.session !== null;
  }

  async infer(task: TaskType, text: string): Promise<InferResult> {
    const t0 = performance.now();
    if (this.session && this.ort) {
      try {
        const res = await this.onnxInfer(task, text);
        return { ...res, backend: "onnx", elapsedMs: performance.now() - t0 };
      } catch (e) {
        // eslint-disable-next-line no-console
        console.warn("[LogicModel] ONNX inference failed, fallback:", e);
      }
    }
    let res: Omit<InferResult, "elapsedMs" | "backend">;
    switch (task) {
      case "prop":
        res = this.stubProp(text);
        break;
      case "syllogism":
        res = this.stubSyllogism(text);
        break;
      case "algebra":
        res = this.stubAlgebra(text);
        break;
    }
    return { ...res, backend: "stub", elapsedMs: performance.now() - t0 };
  }

  // ─── ONNX inference ─────────────────────────────────────────────
  private async onnxInfer(
    task: TaskType,
    text: string,
  ): Promise<Omit<InferResult, "elapsedMs" | "backend">> {
    const ort = this.ort!;
    const sess = this.session!;
    const V = this.tk.vocab;

    const promptToks = ["[BOS]", ...this.tk.fromFormal(text), "[EOS]"];
    const promptIds = this.tk.toIds(promptToks);
    const generated: number[] = [];
    // Training format: [BOS]q[EOS] [BOS]cot1[EOS] ... [BOS]cotN[EOS] [BOS]ans[EOS]
    // We stop after enough segments, OR a PAD, OR budget exhausted.
    let segmentsClosed = 0;
    const MAX_SEGMENTS = 6; // plenty for CoT + answer on all three tasks

    for (let step = 0; step < MAX_NEW_TOKENS; step++) {
      const ctx = [...promptIds, ...generated].slice(-MAX_LEN);
      const padded = new BigInt64Array(MAX_LEN).fill(BigInt(V.PAD));
      for (let j = 0; j < ctx.length; j++) padded[j] = BigInt(ctx[j]);

      const input = new ort.Tensor("int64", padded, [1, MAX_LEN]);
      const out = await sess.run({ ids: input });
      const logits = out.logits.data;
      const vocabSize = V.size;
      const lastReal = ctx.length - 1;
      const start = lastReal * vocabSize;

      let best = 0;
      let bestScore = -Infinity;
      for (let v = 0; v < vocabSize; v++) {
        const sc = logits[start + v];
        if (sc > bestScore) {
          bestScore = sc;
          best = v;
        }
      }

      // Hard stops.
      if (best === V.PAD) break;
      generated.push(best);
      if (best === V.EOS) {
        segmentsClosed++;
        if (segmentsClosed >= MAX_SEGMENTS) break;
      }
    }

    const outTokens = this.tk.toTokens(generated);
    const segments: string[][] = [];
    let cur: string[] = [];
    let inSeg = false;
    for (const t of outTokens) {
      if (t === "[BOS]") {
        inSeg = true;
        cur = [];
        continue;
      }
      if (t === "[EOS]") {
        if (inSeg && cur.length) segments.push(cur);
        inSeg = false;
        continue;
      }
      if (inSeg) cur.push(t);
    }
    if (inSeg && cur.length) segments.push(cur);

    const answerToks = segments.length ? segments[segments.length - 1] : [];
    const cotSegs = segments.slice(0, -1);
    const answer = formatAnswer(task, answerToks);

    const cot: string[][] = [
      this.tk.fromFormal(text),
      ...cotSegs,
      answerToks.length ? answerToks : ["[UNK]"],
    ];
    return { answer, cot };
  }

  // ─── Stub: prop logic ───────────────────────────────────────────
  private stubProp(text: string): Omit<InferResult, "elapsedMs" | "backend"> {
    const env = new Map<string, boolean>();
    const assignRe = /([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(true|false|t|f|1|0)/gi;
    let m: RegExpExecArray | null;
    while ((m = assignRe.exec(text))) {
      const name = m[1].toLowerCase();
      const v = m[2].toLowerCase();
      env.set(name, v === "true" || v === "t" || v === "1");
    }
    const splitIdx = Math.max(text.lastIndexOf(";"), text.toLowerCase().lastIndexOf("then"));
    const exprText = splitIdx >= 0 ? text.slice(splitIdx + 1) : text;
    const toks = this.tk.fromFormal(exprText).filter((t) => !t.startsWith("A:"));
    let answer = "unknown";
    try {
      const v = evalProp(toks, env);
      if (v !== null) answer = v ? "true" : "false";
    } catch {
      /* unknown */
    }
    const cot: string[][] = [];
    for (const [k, v] of env)
      cot.push([`V:${k[0].toUpperCase()}`, "R:EQUALS", v ? "L:TRUE" : "L:FALSE"]);
    cot.push(toks);
    cot.push([answer === "true" ? "L:TRUE" : answer === "false" ? "L:FALSE" : "[UNK]"]);
    return { answer, cot };
  }

  // ─── Stub: syllogism ────────────────────────────────────────────
  private stubSyllogism(text: string): Omit<InferResult, "elapsedMs" | "backend"> {
    const sentences = text
      .split(/[.\n]+/)
      .map((s) => s.trim())
      .filter(Boolean);
    const parsed = sentences.map(parseCategorical);
    const premises = parsed.slice(0, -1);
    const conclusion = parsed[parsed.length - 1];
    let answer = "undetermined";
    if (premises.every(Boolean) && conclusion) {
      answer = judgeSyllogism(premises as CatProp[], conclusion) ? "valid" : "invalid";
    }
    const cot: string[][] = [];
    for (const p of parsed) if (p) cot.push(catToLogic(p));
    cot.push([answer === "valid" ? "L:TRUE" : answer === "invalid" ? "L:FALSE" : "[UNK]"]);
    return { answer, cot };
  }

  // ─── Stub: algebra ──────────────────────────────────────────────
  private stubAlgebra(text: string): Omit<InferResult, "elapsedMs" | "backend"> {
    const lower = text.trim().toLowerCase();
    const cot: string[][] = [this.tk.fromFormal(text)];
    let answer = "unknown";
    const solveM = lower.match(/solve\s+(-?\d+)\s*\*?\s*([a-z])\s*([+-])\s*(-?\d+)\s*=\s*(-?\d+)/);
    if (solveM) {
      const [, aS, v, op, bS, cS] = solveM;
      const a = Number(aS),
        b = (op === "-" ? -1 : 1) * Number(bS),
        c = Number(cS);
      if (a !== 0) {
        const x = (c - b) / a;
        answer = `${v} = ${Number.isInteger(x) ? x : x.toFixed(3)}`;
        cot.push(["A:SOLVE", "V:X"]);
        cot.push(["R:EQUALS", intBucket(c - b), "A:DIV", intBucket(a)]);
      }
    } else if (lower.startsWith("simplify")) {
      answer = simplifyPoly(lower.slice("simplify".length).trim());
      cot.push(["A:SIMPLIFY"]);
    } else if (lower.startsWith("evaluate") || lower.startsWith("eval")) {
      const expr = lower.replace(/^(evaluate|eval)\s+/, "");
      try {
        if (/^[\d+\-*/().\s]+$/.test(expr)) {
          // eslint-disable-next-line no-new-func
          const v = Function(`"use strict";return (${expr})`)();
          answer = String(v);
          cot.push(["A:EVAL"]);
        }
      } catch {
        /* noop */
      }
    }
    cot.push([toNumToken(answer)]);
    return { answer, cot };
  }
}

// ─── Answer formatting from model tokens ──────────────────────────

function formatAnswer(task: TaskType, answerToks: string[]): string {
  if (!answerToks.length) return "unknown";
  const joined = answerToks.join(" ");
  if (task === "prop") {
    if (answerToks.includes("L:TRUE")) return "true";
    if (answerToks.includes("L:FALSE")) return "false";
  }
  if (task === "syllogism") {
    if (answerToks.includes("V:Y") || answerToks.includes("L:TRUE")) return "yes (valid)";
    if (answerToks.includes("V:N") || answerToks.includes("L:FALSE")) return "no (invalid)";
  }
  return joined;
}

// ─── Prop-logic evaluator ──────────────────────────────────────────
function evalProp(toks: string[], env: Map<string, boolean>): boolean | null {
  let i = 0;
  const peek = () => toks[i];
  const eat = () => toks[i++];
  function atom(): boolean | null {
    const t = peek();
    if (t === "L:LPAREN") {
      eat();
      const v = expr();
      if (peek() === "L:RPAREN") eat();
      return v;
    }
    if (t === "L:NOT") {
      eat();
      const v = atom();
      return v === null ? null : !v;
    }
    if (t === "L:TRUE") {
      eat();
      return true;
    }
    if (t === "L:FALSE") {
      eat();
      return false;
    }
    if (t && t.startsWith("V:")) {
      eat();
      const name = t.slice(2).toLowerCase();
      return env.has(name) ? env.get(name)! : null;
    }
    eat();
    return null;
  }
  function expr(): boolean | null {
    let lhs = atom();
    while (i < toks.length) {
      const op = peek();
      if (op === "L:AND") {
        eat();
        const rhs = atom();
        lhs = lhs === null || rhs === null ? null : lhs && rhs;
        continue;
      }
      if (op === "L:OR") {
        eat();
        const rhs = atom();
        lhs = lhs === null || rhs === null ? null : lhs || rhs;
        continue;
      }
      if (op === "L:IMPL") {
        eat();
        const rhs = expr();
        lhs = lhs === null || rhs === null ? null : !lhs || rhs;
        continue;
      }
      if (op === "L:IFF") {
        eat();
        const rhs = expr();
        lhs = lhs === null || rhs === null ? null : lhs === rhs;
        continue;
      }
      break;
    }
    return lhs;
  }
  return expr();
}

// ─── Categorical syllogism parser + judge ─────────────────────────
type Quant = "A" | "E" | "I" | "O";
interface CatProp {
  q: Quant;
  subj: string;
  pred: string;
}

function parseCategorical(s: string): CatProp | null {
  const lower = s.toLowerCase().trim();
  let m = lower.match(/^all\s+([a-z]+)\s+are\s+([a-z]+)/);
  if (m) return { q: "A", subj: m[1], pred: m[2] };
  m = lower.match(/^no\s+([a-z]+)\s+are\s+([a-z]+)/);
  if (m) return { q: "E", subj: m[1], pred: m[2] };
  m = lower.match(/^some\s+([a-z]+)\s+are\s+not\s+([a-z]+)/);
  if (m) return { q: "O", subj: m[1], pred: m[2] };
  m = lower.match(/^some\s+([a-z]+)\s+are\s+([a-z]+)/);
  if (m) return { q: "I", subj: m[1], pred: m[2] };
  m = lower.match(/^therefore,?\s+(.+)/);
  if (m) return parseCategorical(m[1]);
  return null;
}

function catToLogic(c: CatProp): string[] {
  const q = { A: "Q:ALL", E: "Q:NO", I: "Q:SOME", O: "Q:SOME" }[c.q];
  const arr = [q, "C:CONCEPT", "R:IS", "C:CONCEPT"];
  if (c.q === "O") arr.splice(3, 0, "L:NOT");
  return arr;
}

function judgeSyllogism(premises: CatProp[], concl: CatProp): boolean {
  const terms = new Set<string>();
  [...premises, concl].forEach((p) => {
    terms.add(p.subj);
    terms.add(p.pred);
  });
  if (terms.size !== 3) return false;
  const [S, M, P] = [
    concl.subj,
    ...[...terms].filter((t) => t !== concl.subj && t !== concl.pred),
    concl.pred,
  ];
  const regions = [0, 1, 2, 3, 4, 5, 6];
  const inSet = (r: number, t: string): boolean => {
    const bits = [S, M, P].map((_, i) => ((r >> i) & 1) === 1);
    if (t === S) return bits[0];
    if (t === M) return bits[1];
    return bits[2];
  };
  const satisfies = (p: CatProp, pop: boolean[]): boolean => {
    const has = (t: string) => regions.some((r) => pop[r] && inSet(r, t));
    const hasIn = (t: string, u: string) =>
      regions.some((r) => pop[r] && inSet(r, t) && inSet(r, u));
    const hasOut = (t: string, u: string) =>
      regions.some((r) => pop[r] && inSet(r, t) && !inSet(r, u));
    const allIn = (t: string, u: string) =>
      regions.every((r) => !pop[r] || !inSet(r, t) || inSet(r, u));
    const allOut = (t: string, u: string) =>
      regions.every((r) => !pop[r] || !inSet(r, t) || !inSet(r, u));
    if (p.q === "A") return has(p.subj) ? allIn(p.subj, p.pred) : true;
    if (p.q === "E") return allOut(p.subj, p.pred);
    if (p.q === "I") return hasIn(p.subj, p.pred);
    if (p.q === "O") return hasOut(p.subj, p.pred);
    return false;
  };
  for (let mask = 1; mask < 128; mask++) {
    const pop = regions.map((r) => ((mask >> r) & 1) === 1);
    if (!premises.every((p) => satisfies(p, pop))) continue;
    if (!satisfies(concl, pop)) return false;
  }
  return true;
}

// ─── Algebra helpers ──────────────────────────────────────────────
function intBucket(n: number): string {
  if (n === 0) return "N:ZERO";
  if (n === 1) return "N:ONE";
  if (n < 0) return "N:NEG";
  if (n <= 100) return "N:SMALL";
  return "N:LARGE";
}

function toNumToken(ans: string): string {
  const n = Number(ans);
  return Number.isFinite(n) ? intBucket(n) : "[UNK]";
}

function simplifyPoly(s: string): string {
  let xCoef = 0,
    c = 0;
  const tokens = s.match(/[+-]?\s*(\d+)?\s*\*?\s*([a-z])?/g) ?? [];
  for (const raw of tokens) {
    const t = raw.replace(/\s+/g, "");
    if (!t) continue;
    const m = t.match(/^([+-]?)(\d*)([a-z])?$/);
    if (!m) continue;
    const sign = m[1] === "-" ? -1 : 1;
    const num = m[2] === "" ? 1 : Number(m[2]);
    if (m[3]) xCoef += sign * num;
    else if (m[2] !== "") c += sign * num;
  }
  const parts: string[] = [];
  if (xCoef !== 0) parts.push(xCoef === 1 ? "x" : xCoef === -1 ? "-x" : `${xCoef}x`);
  if (c !== 0) parts.push((c > 0 && parts.length ? "+ " : "") + c);
  return parts.length ? parts.join(" ") : "0";
}
