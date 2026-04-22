import type { Token, TokenizerOutput, CoverageStats } from "./types.ts";
import { normalize } from "./normalizer.ts";
import { detectStructure } from "./structureDetector.ts";
import { detectEntities, isEntity } from "./ner.ts";
import { decompose } from "./morphology.ts";
import { emitToken } from "./emitter.ts";
import { Vocabulary } from "./vocabulary.ts";

// @ts-ignore compromise has no typed exports for this pattern
import nlp from "compromise";

export type {
  Token,
  TokenizerOutput,
  CoverageStats,
  VocabEntry,
} from "./types.ts";
export { Vocabulary } from "./vocabulary.ts";
export { getSemanticField, SEMANTIC_FIELDS } from "./semanticFields.ts";

/**
 * Contextual Semantic Tokenizer (CST)
 *
 * Converts English text into typed semantic tokens:
 *   CMP:field:role  — composed (best)
 *   ROOT:field      — semantic field only
 *   REL:relation    — logical relation
 *   STR:marker      — sentence structure
 *   LIT:surface     — literal fallback
 */
export class CSTTokenizer {
  private vocab = new Vocabulary();
  private lemmaCache = new Map<string, string>();

  // ── Core API ────────────────────────────────────────

  tokenize(text: string): TokenizerOutput {
    // Stage 1 — Normalize
    const normalized = normalize(text);

    // Stage 2 — Sentence structure detection
    const structure = detectStructure(normalized);

    // Stage 3 — Split into words
    const words = this.splitWords(normalized);

    // Stage 4 — NER (detect entities before morphology)
    const entityMap = detectEntities(text);

    // Stage 5+6+7 — For each word: decompose → map field → emit
    const doc = nlp(normalized);
    const tokens: Token[] = [];

    // Emit STR tokens first
    for (const strVal of structure.tokens) {
      tokens.push({
        type: "STR",
        value: strVal,
        surface: "",
        id: 0,
        confidence: 1.0,
      });
    }

    // Process each word
    for (const word of words) {
      if (!word || word.length === 0) continue;

      // Skip pure punctuation (including en-dash, em-dash, ellipsis)
      if (/^[.,!?;:'"()\-–—…\/]+$/.test(word)) continue;

      // Skip contraction fragments and single-letter noise
      if (
        /^(s|t|re|ll|ve|d|m|don|won|didn|doesn|isn|wasn|aren|couldn|shouldn|wouldn|haven|hasn|weren|ain|e|u|o|x)$/.test(
          word,
        )
      )
        continue;

      // Get lemma from compromise
      const lemma = this.getLemma(word, doc);

      // Check entity
      const isEnt = isEntity(word, entityMap);

      // Morphological decomposition
      const dec = decompose(word, lemma);

      // Emit token
      const token = emitToken(word, lemma, isEnt, dec);
      tokens.push(token);
    }

    // Assign vocabulary IDs
    const ids = this.vocab.assignIds(tokens);

    return {
      tokens,
      ids,
      coverage: this.computeCoverage(tokens),
    };
  }

  tokenizeBatch(texts: string[]): TokenizerOutput[] {
    return texts.map((t) => this.tokenize(t));
  }

  encode(text: string): number[] {
    return this.tokenize(text).ids;
  }

  decode(ids: number[]): Token[] {
    return ids.map((id) => {
      const entry = this.vocab.getToken(id);
      if (!entry) {
        return {
          type: "SPECIAL" as const,
          value: "[UNK]",
          surface: "",
          id,
          confidence: 0,
        };
      }
      return {
        type: entry.type,
        value: entry.token,
        surface: entry.gloss ?? "",
        id: entry.id,
        confidence: 1.0,
      };
    });
  }

  // ── Vocabulary management ──────────────────────────

  getVocab(): Vocabulary {
    return this.vocab;
  }

  getVocabSize(): number {
    return this.vocab.size;
  }

  saveVocab(path: string): void {
    this.vocab.save(path);
  }

  loadVocab(path: string): void {
    this.vocab.load(path);
  }

  // ── Coverage analysis ──────────────────────────────

  getCoverage(texts: string[]): CoverageStats {
    const totals: CoverageStats = {
      total: 0,
      cmp: 0,
      root: 0,
      str: 0,
      rel: 0,
      lit: 0,
      unk: 0,
      structured: 0,
      ratio: 0,
    };

    for (const text of texts) {
      const output = this.tokenize(text);
      const c = output.coverage;
      totals.total += c.total;
      totals.cmp += c.cmp;
      totals.root += c.root;
      totals.str += c.str;
      totals.rel += c.rel;
      totals.lit += c.lit;
      totals.unk += c.unk;
    }

    totals.structured = totals.cmp + totals.root + totals.str + totals.rel;
    totals.ratio = totals.total > 0 ? totals.structured / totals.total : 0;
    return totals;
  }

  // ── Internal helpers ───────────────────────────────

  private splitWords(text: string): string[] {
    // Split on whitespace and punctuation, keeping words only
    return text
      .replace(/([.,!?;:'"()\-])/g, " $1 ")
      .split(/\s+/)
      .filter((w) => w.length > 0);
  }

  private getLemma(word: string, doc: any): string {
    const cached = this.lemmaCache.get(word);
    if (cached !== undefined) return cached;

    // Use compromise to find the lemma/root form
    const termDoc = nlp(word);
    const verbs = termDoc.verbs();
    if (verbs.length > 0) {
      const inf = verbs.toInfinitive().text();
      if (inf && inf.length > 0) {
        const result = inf.toLowerCase();
        this.lemmaCache.set(word, result);
        return result;
      }
    }

    const nouns = termDoc.nouns();
    if (nouns.length > 0) {
      const sing = nouns.toSingular().text();
      if (sing && sing.length > 0) {
        const result = sing.toLowerCase();
        this.lemmaCache.set(word, result);
        return result;
      }
    }

    const result = word.toLowerCase();
    this.lemmaCache.set(word, result);
    return result;
  }

  private computeCoverage(tokens: Token[]): CoverageStats {
    const stats: CoverageStats = {
      total: 0,
      cmp: 0,
      root: 0,
      str: 0,
      rel: 0,
      lit: 0,
      unk: 0,
      structured: 0,
      ratio: 0,
    };

    for (const t of tokens) {
      stats.total++;
      switch (t.type) {
        case "CMP":
          stats.cmp++;
          break;
        case "ROOT":
          stats.root++;
          break;
        case "STR":
          stats.str++;
          break;
        case "REL":
          stats.rel++;
          break;
        case "LIT":
          stats.lit++;
          break;
        case "SPECIAL":
          stats.unk++;
          break;
      }
    }

    stats.structured = stats.cmp + stats.root + stats.str + stats.rel;
    stats.ratio = stats.total > 0 ? stats.structured / stats.total : 0;
    return stats;
  }
}
