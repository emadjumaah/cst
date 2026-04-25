/**
 * humanize.ts — pretty-print CST-logic tokens as English prose.
 *
 * Used to render a human-readable view of the CoT next to the raw
 * token chips. Not semantically exact — just enough to let a reader
 * follow what the 10.79M model is saying at a glance.
 */

import type { TaskType } from "./model";

const ATOM: Record<string, string> = {
  // logic constants / ops
  "L:TRUE": "true",
  "L:FALSE": "false",
  "L:AND": "and",
  "L:OR": "or",
  "L:NOT": "not",
  "L:IMPL": "→",
  "L:IFF": "↔",
  "L:THEREFORE": "∴",
  "L:BECAUSE": "because",
  "L:LPAREN": "(",
  "L:RPAREN": ")",

  // quantifiers
  "Q:ALL": "all",
  "Q:SOME": "some",
  "Q:NO": "no",
  "Q:NONE": "none",
  "Q:EXISTS": "∃",
  "Q:UNIQUE": "exactly one",
  "Q:MOST": "most",
  "Q:FEW": "few",

  // relations
  "R:IS": "is",
  "R:ISA": "is a",
  "R:HAS": "has",
  "R:EQUALS": "=",
  "R:NE": "≠",
  "R:GT": ">",
  "R:LT": "<",
  "R:GE": "≥",
  "R:LE": "≤",
  "R:BEFORE": "before",
  "R:AFTER": "after",
  "R:IN": "in",
  "R:PART_OF": "part of",

  // arithmetic
  "A:PLUS": "+",
  "A:MINUS": "−",
  "A:TIMES": "×",
  "A:DIV": "÷",
  "A:POW": "^",
  "A:SOLVE": "solve",
  "A:SIMPLIFY": "simplify",
  "A:FACTOR": "factor",
  "A:EXPAND": "expand",
  "A:EVAL": "evaluate",

  // time / modal
  "T:PAST": "(past)",
  "T:PRESENT": "(present)",
  "T:FUTURE": "(future)",
  "T:ALWAYS": "always",
  "T:NEVER": "never",
  "T:SOMETIMES": "sometimes",
  "M:MUST": "must",
  "M:MAY": "may",
  "M:CAN": "can",
  "M:SHOULD": "should",

  // numbers
  "N:ZERO": "0",
  "N:ONE": "1",
  "N:SMALL": "«small n»",
  "N:LARGE": "«large n»",
  "N:NEG": "«negative»",
  "N:FRAC": "«fraction»",
  "N:REAL": "«real»",
  "N:INT": "«int»",

  // structure
  "S:PREMISE": "premise:",
  "S:CONCLUSION": "conclusion:",
  "S:STEP": "step:",
  "S:ASSUME": "assume",
  "S:PROVE": "prove",
  "S:CASE": "case",
  "S:DEFINE": "define",
  "S:QED": "∎",

  // specials
  "[BOS]": "",
  "[EOS]": "",
  "[PAD]": "",
  "[UNK]": "?",
  "[SEP]": ";",
  "[Q]": "(question)",
};

/** Render one segment (already split on BOS/EOS) as a short English phrase. */
export function humanizeSegment(toks: string[]): string {
  const parts: string[] = [];
  for (const t of toks) {
    if (!t || t.startsWith("[")) continue;
    if (t in ATOM) {
      parts.push(ATOM[t]);
      continue;
    }
    // Category prefixes we didn't hard-code.
    if (t.startsWith("V:")) parts.push(t.slice(2).toLowerCase());
    else if (t.startsWith("C:")) parts.push(t.slice(2).toLowerCase());
    else if (t.startsWith("RO:")) parts.push(`[${t.slice(3).toLowerCase()}]`);
    else parts.push(t);
  }
  // light cleanup: collapse whitespace, tighten spacing around parens.
  return parts
    .join(" ")
    .replace(/\s+/g, " ")
    .replace(/\(\s+/g, "(")
    .replace(/\s+\)/g, ")")
    .replace(/\s+([,;.])/g, "$1")
    .trim();
}

/** One-line gloss of the final answer for the info banner. */
export function humanizeAnswer(task: TaskType, answer: string): string {
  const a = answer.trim().toLowerCase();
  if (task === "prop") {
    if (a === "true") return "The expression evaluates to TRUE.";
    if (a === "false") return "The expression evaluates to FALSE.";
  }
  if (task === "syllogism") {
    if (a.startsWith("yes")) return "The conclusion follows from the premises (valid).";
    if (a.startsWith("no")) return "The conclusion does NOT follow (invalid).";
  }
  if (task === "algebra") {
    if (answer && !answer.includes(":") && answer !== "unknown") {
      return `Result: ${answer}`;
    }
  }
  return answer ? `Answer: ${answer}` : "No answer produced.";
}
