/**
 * logic-tokenizer.ts — browser port of edge/logic_tokenizer.py
 *
 * Same 151-token closed vocabulary, same projection tables, same formal
 * regex pass. Vocab JSONs are fetched at runtime from /logic/vocab/ so
 * they stay in lockstep with the Python source of truth.
 */

export type Bucket =
  | "SPECIAL" | "L" | "Q" | "R" | "T" | "M"
  | "RO" | "C" | "A" | "S" | "V" | "N";

export function bucketOf(tok: string): Bucket {
  if (tok.startsWith("[")) return "SPECIAL";
  const p = tok.slice(0, tok.indexOf(":")) as Bucket;
  return p || "SPECIAL";
}

const VOCAB_FILES = [
  "specials", "operators", "quantifiers", "relations", "time_modal",
  "roles", "concepts", "arithmetic", "structure", "variables", "numbers",
] as const;

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch ${url}: ${res.status}`);
  return res.json() as Promise<T>;
}

export interface LogicVocab {
  tokenToId: Map<string, number>;
  idToToken: Map<number, string>;
  size: number;
  PAD: number;
  UNK: number;
  BOS: number;
  EOS: number;
}

export async function loadVocab(baseUrl = "logic/vocab"): Promise<LogicVocab> {
  const tokenToId = new Map<string, number>();
  const idToToken = new Map<number, string>();

  // specials — list, insertion order
  const specials = await fetchJson<string[]>(`${baseUrl}/specials.json`);
  for (const t of specials) {
    idToToken.set(tokenToId.size, t);
    tokenToId.set(t, tokenToId.size);
  }

  // other buckets — dict, keys sorted for determinism
  for (const name of VOCAB_FILES.slice(1)) {
    const data = await fetchJson<Record<string, string>>(`${baseUrl}/${name}.json`);
    for (const key of Object.keys(data).sort()) {
      if (tokenToId.has(key)) throw new Error(`duplicate logic token: ${key}`);
      idToToken.set(tokenToId.size, key);
      tokenToId.set(key, tokenToId.size);
    }
  }

  const get = (t: string) => {
    const id = tokenToId.get(t);
    if (id === undefined) throw new Error(`missing special: ${t}`);
    return id;
  };
  return {
    tokenToId, idToToken, size: tokenToId.size,
    PAD: get("[PAD]"), UNK: get("[UNK]"),
    BOS: get("[BOS]"), EOS: get("[EOS]"),
  };
}

// ─── CST-standard → logic projection (mirror of Python) ───────────────

const STD_REL_TO_LOGIC: Record<string, string | null> = {
  "REL:and": "L:AND", "REL:or": "L:OR", "REL:but": "L:AND",
  "REL:not": "L:NOT", "REL:neither": "L:NOT",
  "REL:if": "L:IMPL", "REL:then": "L:IMPL", "REL:iff": "L:IFF",
  "REL:therefore": "L:THEREFORE", "REL:because": "L:BECAUSE",
  "REL:so": "L:THEREFORE",
  "REL:all": "Q:ALL", "REL:every": "Q:ALL", "REL:each": "Q:ALL",
  "REL:some": "Q:SOME", "REL:any": "Q:SOME",
  "REL:no": "Q:NO", "REL:none": "Q:NONE",
  "REL:before": "R:BEFORE", "REL:after": "R:AFTER", "REL:in": "R:IN",
  "REL:has": "R:HAS", "REL:of": "R:PART_OF", "REL:is": "R:IS",
};

const STD_STR_TO_LOGIC: Record<string, string | null> = {
  "STR:neg:general": "L:NOT", "STR:neg:past": "L:NOT",
  "STR:neg:future": "L:NOT", "STR:neg:nominal": "L:NOT",
  "STR:cond:likely": "L:IMPL", "STR:cond:hypo": "L:IMPL",
  "STR:cond:counter": "L:IMPL",
  "STR:future": "T:FUTURE", "STR:question": "[Q]",
  "STR:emphasis": null,
};

const STD_FEAT_TO_LOGIC: Record<string, string | null> = {
  "FEAT:asp:p": "T:PAST", "FEAT:asp:i": "T:PRESENT", "FEAT:asp:c": "T:PRESENT",
};

const STD_NUM_TO_LOGIC: Record<string, string> = {
  "NUM:zero": "N:ZERO", "NUM:one": "N:ONE",
  "NUM:small": "N:SMALL", "NUM:large": "N:LARGE",
  "NUM:year": "N:LARGE", "NUM:percent": "N:SMALL",
  "NUM:neg": "N:NEG", "NUM:frac": "N:FRAC", "NUM:real": "N:REAL",
};

const FIELD_TO_CONCEPT: Record<string, string> = {
  person: "C:PERSON", family: "C:PERSON", body: "C:OBJECT",
  social: "C:GROUP", social_g: "C:GROUP",
  place: "C:PLACE", nature: "C:PLACE",
  move: "C:ACTION", make: "C:ACTION", create: "C:ACTION",
  destroy: "C:ACTION", fight: "C:ACTION", build: "C:ACTION",
  change: "C:PROCESS", enable: "C:PROCESS",
  give: "C:ACTION", take: "C:ACTION", gather: "C:ACTION",
  send: "C:ACTION", speak: "C:ACTION", fix: "C:ACTION",
  want: "C:STATE", feel: "C:EMOTION",
  think: "C:IDEA", know: "C:IDEA", art: "C:IDEA",
  force: "C:PROPERTY", govern: "C:RULE", decide: "C:ACTION",
  size: "C:SIZE", color: "C:COLOR", time: "C:TIME_POINT",
  quality: "C:PROPERTY",
  food: "C:OBJECT", animal: "C:ANIMAL", material: "C:OBJECT",
  trade: "C:ACTION", dwell: "C:PLACE", contain: "C:WHOLE",
  connect: "C:RELATION", exist: "C:STATE",
};

const CMP_ROLE_TO_ROLE: Record<string, string | null> = {
  agent: "RO:AGENT", patient: "RO:PATIENT", place: "RO:LOCATION",
  instance: null, state: null, mutual: null, process: null,
  causer: "RO:CAUSE", seeker: "RO:AGENT", quality: null,
  intensifier: null,
};

function projectStandardToken(tok: string): string[] {
  if (["[BOS]", "[EOS]", "[PAD]", "[UNK]", "[SEP]"].includes(tok)) return [tok];

  if (tok in STD_REL_TO_LOGIC) {
    const m = STD_REL_TO_LOGIC[tok]; return m ? [m] : [];
  }
  if (tok in STD_STR_TO_LOGIC) {
    const m = STD_STR_TO_LOGIC[tok]; return m ? [m] : [];
  }
  if (tok in STD_FEAT_TO_LOGIC) {
    const m = STD_FEAT_TO_LOGIC[tok]; return m ? [m] : [];
  }
  if (tok.startsWith("FEAT:")) return [];

  if (tok in STD_NUM_TO_LOGIC) return [STD_NUM_TO_LOGIC[tok]];
  if (tok.startsWith("NUM:")) return ["N:INT"];

  if (tok.startsWith("TIME:")) return ["C:TIME_POINT"];

  if (tok.startsWith("CMP:")) {
    const parts = tok.split(":");
    if (parts.length === 3) {
      const [, field, role] = parts;
      const concept = FIELD_TO_CONCEPT[field] ?? "C:CONCEPT";
      const roleTok = CMP_ROLE_TO_ROLE[role];
      return roleTok ? [concept, roleTok] : [concept];
    }
    return ["C:CONCEPT"];
  }

  if (tok.startsWith("ROOT:")) {
    const field = tok.slice(5);
    return [FIELD_TO_CONCEPT[field] ?? "C:CONCEPT"];
  }
  if (tok.startsWith("NE:")) return ["C:PERSON"];
  if (tok.startsWith("FOREIGN:")) return ["C:CONCEPT"];
  if (tok.startsWith("LIT:")) return ["[UNK]"];

  return ["[UNK]"];
}

// ─── Formal-logic / algebra regex pass ────────────────────────────────

interface TokPattern { re: RegExp; tok: string; }

const SIMPLE_OPS: TokPattern[] = [
  // Order matters: longer / more specific before shorter.
  { re: /^(↔|<->|<=>)/, tok: "L:IFF" },
  { re: /^(→|->|=>)/, tok: "L:IMPL" },
  { re: /^(∧|&&|&)/, tok: "L:AND" },
  { re: /^(∨|\|\||\|)/, tok: "L:OR" },
  { re: /^(≠|!=)/, tok: "R:NE" },
  { re: /^(¬|~|!)/, tok: "L:NOT" },
  { re: /^(==|=)/, tok: "R:EQUALS" },
  { re: /^(≥|>=)/, tok: "R:GE" },
  { re: /^(≤|<=)/, tok: "R:LE" },
  { re: /^>/, tok: "R:GT" },
  { re: /^</, tok: "R:LT" },
  { re: /^\+/, tok: "A:PLUS" },
  { re: /^-/, tok: "A:MINUS" },
  { re: /^(×|\*)/, tok: "A:TIMES" },
  { re: /^(÷|\/)/, tok: "A:DIV" },
  { re: /^\^/, tok: "A:POW" },
  { re: /^∀/, tok: "Q:ALL" },
  { re: /^∃/, tok: "Q:SOME" },
  { re: /^∴/, tok: "L:THEREFORE" },
  { re: /^∵/, tok: "L:BECAUSE" },
  { re: /^\(/, tok: "L:LPAREN" },
  { re: /^\)/, tok: "L:RPAREN" },
];

const KEYWORD_TO_LOGIC: Record<string, string> = {
  and: "L:AND", or: "L:OR", not: "L:NOT", but: "L:AND",
  if: "L:IMPL", then: "L:IMPL", iff: "L:IFF",
  therefore: "L:THEREFORE", so: "L:THEREFORE",
  because: "L:BECAUSE", since: "L:BECAUSE",
  all: "Q:ALL", every: "Q:ALL", each: "Q:ALL", any: "Q:SOME",
  forall: "Q:ALL", for_all: "Q:ALL",
  some: "Q:SOME", no: "Q:NO", none: "Q:NONE",
  exists: "Q:EXISTS", unique: "Q:UNIQUE",
  most: "Q:MOST", few: "Q:FEW",
  is: "R:IS", are: "R:IS", was: "R:IS", were: "R:IS",
  isa: "R:ISA", has: "R:HAS", have: "R:HAS",
  equals: "R:EQUALS",
  true: "L:TRUE", false: "L:FALSE",
  simplify: "A:SIMPLIFY", expand: "A:EXPAND",
  factor: "A:FACTOR", solve: "A:SOLVE", evaluate: "A:EVAL",
  before: "R:BEFORE", after: "R:AFTER",
  must: "M:MUST", may: "M:MAY",
  can: "M:CAN", should: "M:SHOULD",
  always: "T:ALWAYS", never: "T:NEVER",
  sometimes: "T:SOMETIMES",
  past: "T:PAST", future: "T:FUTURE", present: "T:PRESENT",
  premise: "S:PREMISE", conclusion: "S:CONCLUSION",
  step: "S:STEP", assume: "S:ASSUME",
  prove: "S:PROVE", qed: "S:QED",
  case: "S:CASE", define: "S:DEFINE",
};

function integerBucket(n: number): string {
  if (n === 0) return "N:ZERO";
  if (n === 1) return "N:ONE";
  if (n < 0) return "N:NEG";
  if (n <= 100) return "N:SMALL";
  return "N:LARGE";
}

export function tokenizeFormal(text: string, vocab: LogicVocab): string[] {
  const out: string[] = [];
  let i = 0;
  const s = text;
  while (i < s.length) {
    const ch = s[i];
    if (/\s/.test(ch) || ch === "." || ch === "," || ch === ";" || ch === ":") { i++; continue; }

    // simple ops
    let matched = false;
    for (const { re, tok } of SIMPLE_OPS) {
      const m = s.slice(i).match(re);
      if (m) { out.push(tok); i += m[0].length; matched = true; break; }
    }
    if (matched) continue;

    // integer
    const intM = s.slice(i).match(/^\d+/);
    if (intM) { out.push(integerBucket(Number(intM[0]))); i += intM[0].length; continue; }

    // identifier
    const idM = s.slice(i).match(/^[A-Za-z_][A-Za-z0-9_]*/);
    if (idM) {
      const raw = idM[0];
      const lower = raw.toLowerCase();
      if (lower in KEYWORD_TO_LOGIC) {
        out.push(KEYWORD_TO_LOGIC[lower]);
      } else {
        const vSlot = `V:${raw[0].toUpperCase()}`;
        out.push(vocab.tokenToId.has(vSlot) ? vSlot : "C:CONCEPT");
      }
      i += raw.length; continue;
    }

    // unknown char — skip
    i++;
  }
  return out;
}

// ─── Collapse + public API ────────────────────────────────────────────

const COLLAPSIBLE = new Set([
  "L:AND", "L:OR", "L:NOT", "L:SEP",
  "T:PAST", "T:PRESENT", "T:FUTURE", "[Q]",
]);

function collapse(toks: string[], vocab: LogicVocab): string[] {
  const out: string[] = [];
  for (const t of toks) {
    const safe = vocab.tokenToId.has(t) ? t : "[UNK]";
    if (out.length && out[out.length - 1] === safe && COLLAPSIBLE.has(safe)) continue;
    out.push(safe);
  }
  return out;
}

export class LogicTokenizer {
  constructor(public vocab: LogicVocab) {}

  static async load(baseUrl = "logic/vocab"): Promise<LogicTokenizer> {
    return new LogicTokenizer(await loadVocab(baseUrl));
  }

  fromStandard(tokens: string[], addBosEos = false): string[] {
    const projected: string[] = [];
    if (addBosEos) projected.push("[BOS]");
    for (const tok of tokens) for (const p of projectStandardToken(tok)) projected.push(p);
    if (addBosEos) projected.push("[EOS]");
    return collapse(projected, this.vocab);
  }

  fromFormal(text: string, addBosEos = false): string[] {
    const raw = tokenizeFormal(text, this.vocab);
    const projected = addBosEos ? ["[BOS]", ...raw, "[EOS]"] : raw;
    return collapse(projected, this.vocab);
  }

  toIds(tokens: string[]): number[] {
    return tokens.map(t => this.vocab.tokenToId.get(t) ?? this.vocab.UNK);
  }

  toTokens(ids: number[]): string[] {
    return ids.map(i => this.vocab.idToToken.get(i) ?? "[UNK]");
  }
}
