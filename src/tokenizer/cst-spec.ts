/**
 * CST Specification v1.0 — Contextual Semantic Tokenization
 *
 * This is the single source of truth for the CST token format.
 * All language-specific analyzers (English, Arabic, etc.) MUST produce
 * tokens conforming to this spec.
 *
 * Architecture:
 *   Input text → Language Analyzer (per-language) → CST Tokens (universal)
 *
 * The analyzer is a language-specific plugin. The output format is universal.
 */

// ═══════════════════════════════════════════════════════════════
// 1. TOKEN TYPES
// ═══════════════════════════════════════════════════════════════

/**
 * Every CST token has one of these types.
 * Ordered by semantic richness (CMP > ROOT > REL > STR > LIT).
 */
export type CSTTokenType = "CMP" | "ROOT" | "REL" | "STR" | "LIT" | "SPECIAL";

/**
 * CMP:<field>:<role>  — Composed: semantic field + morphological role (richest)
 * ROOT:<field>        — Semantic field only (no role detected)
 * REL:<relation>      — Relational word (preposition, conjunction, quantifier, etc.)
 * STR:<marker>        — Sentence-level structure (question, negation, etc.)
 * LIT:<surface>       — Surface fallback (named entities, unmapped words)
 * SPECIAL             — Control tokens: [PAD], [UNK], [BOS], [EOS], [SEP]
 */

// ═══════════════════════════════════════════════════════════════
// 2. SPECIAL TOKENS (reserved IDs, all languages)
// ═══════════════════════════════════════════════════════════════

export const SPECIAL_TOKENS = {
  "[PAD]": 0,
  "[UNK]": 1,
  "[BOS]": 2,
  "[EOS]": 3,
  "[SEP]": 4,
} as const;

// ═══════════════════════════════════════════════════════════════
// 3. SEMANTIC FIELDS (universal, language-independent)
// ═══════════════════════════════════════════════════════════════

/**
 * The canonical set of semantic fields. Every ROOT: and CMP: token
 * MUST use one of these field names.
 *
 * Both English and Arabic map to the same fields:
 *   English: "write" → ROOT:write (via lemma lookup)
 *   Arabic:  "ك.ت.ب" → ROOT:write (via trilateral root)
 *
 * 58 fields organized by category.
 */
export const SEMANTIC_FIELDS = [
  // ── Cognition & Communication ──
  "know", // learn, understand, study, discover
  "think", // reason, analyze, believe, consider
  "speak", // say, tell, talk, explain, discuss
  "write", // record, document, publish, inscribe
  "see", // look, watch, observe, notice
  "feel", // love, hate, fear, hope, emotion
  "decide", // choose, select, determine, resolve

  // ── Action & Creation ──
  "make", // build, create, produce, fabricate
  "create", // invent, originate, establish, found
  "destroy", // break, damage, demolish, ruin
  "change", // transform, modify, alter, evolve
  "fix", // repair, restore, heal, mend
  "work", // labor, operate, perform, execute
  "enable", // help, support, allow, assist, facilitate

  // ── Movement & Transfer ──
  "move", // go, travel, walk, run, transport
  "send", // transmit, dispatch, deliver, broadcast
  "give", // share, donate, grant, distribute
  "take", // seize, capture, steal, acquire
  "gather", // collect, assemble, meet, merge
  "hold", // maintain, preserve, sustain, grip
  "open", // close, shut, lock, unlock
  "hide", // conceal, cover, reveal, expose
  "connect", // link, attach, bind, separate

  // ── Existence & State ──
  "exist", // live, survive, appear, occur, become
  "rest", // sleep, relax, pause, calm
  "want", // desire, need, require, demand, seek

  // ── Social & Power ──
  "govern", // rule, control, lead, manage, politics, law
  "fight", // battle, attack, defend, war, military
  "trade", // buy, sell, economy, business, finance
  "social", // community, society, organization, culture
  "possess", // own, have, keep, property

  // ── Domain Knowledge ──
  "science", // experiment, theory, research, biology, chemistry
  "health", // medicine, disease, treatment, body function
  "tech", // computer, software, digital, engineering
  "art", // music, painting, film, theater, literature
  "sport", // game, match, athletic, competition

  // ── Physical World ──
  "nature", // earth, sea, mountain, forest, ecology
  "weather", // storm, rain, temperature, climate
  "animal", // bird, fish, mammal, insect
  "plant", // tree, flower, seed, agriculture
  "body", // hand, head, eye, heart, organ
  "food", // eat, drink, taste, meal, cuisine
  "material", // metal, glass, wood, fabric, stone
  "color", // red, blue, green, spectrum

  // ── Space, Time & Measure ──
  "time", // schedule, period, era, date, duration
  "place", // location, region, city, country, geography
  "dwell", // house, building, room, architecture
  "structure", // tower, wall, bridge, infrastructure, layout
  "size", // large, small, number, quantity, amount
  "measure", // count, weigh, calculate, distance
  "quality", // good, bad, strong, weak, ability

  // ── Classification & Reference ──
  "person", // man, woman, child, family, individual
  "name", // label, term, symbol, sign, mark
  "contain", // include, consist, involve, category, type
  "force", // push, pull, pressure, gravity, impact
] as const;

export type SemanticField = (typeof SEMANTIC_FIELDS)[number];

// ═══════════════════════════════════════════════════════════════
// 4. MORPHOLOGICAL ROLES (for CMP tokens)
// ═══════════════════════════════════════════════════════════════

/**
 * Roles describe the grammatical function derived from morphology.
 *
 * English: detected via prefix/suffix (un-break-able → negate, possible)
 * Arabic:  detected via وزن/pattern (فاعل → agent, مفعول → patient)
 *
 * CMP format: CMP:<field>:<role>
 * Example:    CMP:write:agent = "writer" (EN) / "كاتب" (AR)
 */
export const MORPHOLOGICAL_ROLES = [
  // ── From affixes (both languages) ──
  "agent", // doer: -er/-or (EN), فاعل (AR)
  "patient", // receiver: -ee (EN), مفعول (AR)
  "instance", // act/result: -tion/-ment (EN), فِعال (AR)
  "state", // quality: -ness/-ity (EN), فُعولة (AR)
  "place", // location: -ery/-ory (EN), مَفعَلة (AR)
  "possible", // capability: -able/-ible (EN), فعيل (AR)

  // ── English-origin ──
  "negate", // reversal: un-/dis-/non-/-less
  "repeat", // again: re-
  "before", // prior: pre-
  "wrong", // error: mis-
  "excess", // over: over-
  "mutual", // together: co- (EN), تفاعل (AR)
  "exceed", // surpass: out-
  "has", // possessing: -ful
  "manner", // how: -ly
  "quality", // adjective: -al
  "past", // completed: -ed
  "plural", // multiple: -s

  // ── Arabic-origin (أوزان) ──
  "intensifier", // فعّال — intensive doer
  "causer", // مُفعِل — causative
  "seeker", // مُستفعِل — one who seeks/requests
  "process", // مُفاعَلة — reciprocal action
] as const;

export type MorphRole = (typeof MORPHOLOGICAL_ROLES)[number];

// ═══════════════════════════════════════════════════════════════
// 5. RELATION CATEGORIES (for REL tokens)
// ═══════════════════════════════════════════════════════════════

/**
 * REL tokens capture grammatical relationships between content words.
 * The category set is universal; each language maps its own function
 * words to these categories.
 *
 * English: "in" → REL:in, "because" → REL:causes
 * Arabic:  "في" → REL:in, "لأن" → REL:causes
 */
export const RELATION_CATEGORIES = {
  // ── Spatial ──
  in: "spatial",
  at: "spatial",
  on: "spatial",
  to: "spatial",
  from: "spatial",
  by: "spatial",
  for: "spatial",
  of: "spatial",
  with: "spatial",
  without: "spatial",
  between: "spatial",
  among: "spatial",
  through: "spatial",
  across: "spatial",
  along: "spatial",
  around: "spatial",
  within: "spatial",
  upon: "spatial",
  above: "spatial",
  below: "spatial",
  under: "spatial",
  over: "spatial",
  near: "spatial",
  beside: "spatial",
  behind: "spatial",
  beyond: "spatial",
  infront: "spatial", // AR: أمام (in front of, spatial) — distinct from before (temporal)
  against: "spatial",
  into: "spatial",
  onto: "spatial",
  about: "spatial",

  // ── Temporal ──
  before: "temporal",
  after: "temporal",
  during: "temporal",
  until: "temporal",
  now: "temporal",
  then: "temporal",
  already: "temporal",
  still: "temporal",
  yet: "temporal",
  once: "temporal",
  since: "temporal", // AR: منذ (temporal source) — distinct from from (spatial source)

  // ── Logical ──
  causes: "logical",
  condition: "logical",
  despite: "logical",
  contrast: "logical",
  like: "logical",
  unlike: "logical",
  except: "logical",
  instead: "logical",
  per: "logical",
  via: "logical",
  versus: "logical",
  including: "logical",
  compare: "logical",

  // ── Conjunctive ──
  and: "conjunctive",
  or: "conjunctive",
  but: "conjunctive",
  nor: "conjunctive",

  // ── Quantifier ──
  all: "quantifier",
  each: "quantifier",
  every: "quantifier",
  some: "quantifier",
  any: "quantifier",
  both: "quantifier",
  many: "quantifier",
  much: "quantifier",
  more: "quantifier",
  most: "quantifier",
  few: "quantifier",
  less: "quantifier",
  other: "quantifier",
  another: "quantifier",
  several: "quantifier",
  only: "quantifier",
  enough: "quantifier",
  least: "quantifier",

  // ── Referential ──
  that: "referential",
  which: "referential",
  who: "referential",
  whom: "referential",
  what: "referential",
  where: "referential",
  when: "referential",
  how: "referential",
  why: "referential",
  this: "referential",
  these: "referential",
  those: "referential",
  there: "referential",
  here: "referential",

  // ── Pronominal ──
  possess: "pronominal", // their, our, his, her, its, my, your
  them: "pronominal",
  us: "pronominal",
  him: "pronominal",
  themselves: "pronominal",
  itself: "pronominal",
  himself: "pronominal",
  herself: "pronominal",
  ourselves: "pronominal",
  yourself: "pronominal",
  either: "pronominal",
  neither: "pronominal",

  // ── Epistemic / Adverbial ──
  also: "adverbial",
  even: "adverbial",
  just: "adverbial",
  emphasis: "adverbial",
  general: "adverbial",
  usual: "adverbial",
  frequent: "adverbial",
  sometimes: "adverbial",
  always: "adverbial",
  never: "adverbial",
  maybe: "adverbial",
  almost: "adverbial",
  exactly: "adverbial",
  quite: "adverbial",
  well: "adverbial",
  indeed: "adverbial",

  // ── Misc ──
  negate: "misc",
  being: "misc",
  been: "misc",
  having: "misc",
  according: "misc",
  together: "misc",
  apart: "misc",
  away: "misc",
  back: "misc",
  up: "misc",
  down: "misc",
  out: "misc",
  off: "misc",
  as: "misc",
  none: "misc",
  complete: "misc",
  able: "misc",
  permit: "misc",
  quantity: "misc",
  cause: "misc",
  far: "misc",
  likely: "misc",
} as const;

// ═══════════════════════════════════════════════════════════════
// 6. STRUCTURE MARKERS (for STR tokens)
// ═══════════════════════════════════════════════════════════════

/**
 * STR tokens mark sentence-level properties.
 * Detected from the full sentence BEFORE word-by-word processing.
 * Emitted at the front of the token sequence (after [BOS]).
 *
 * Language-specific detection rules, universal marker names.
 */
export const STRUCTURE_MARKERS = [
  "question", // EN: trailing ?, AR: trailing ؟ or هل/ما/أ prefix

  // Negation — split in Arabic so the model can learn verb mood / case.
  // English: not/never/can't all → STR:negation.
  // Arabic:  لا → STR:neg:general (indicative/imperative scope)
  //          لم → STR:neg:past    (governs jussive)
  //          لن → STR:neg:future  (governs subjunctive)
  //          ليس → STR:neg:nominal (governs accusative predicate)
  "negation",
  "neg:general",
  "neg:past",
  "neg:future",
  "neg:nominal",

  // Conditional — split in Arabic so the model learns mood / likelihood.
  // English: if/unless/when all → STR:condition.
  // Arabic:  إذا  → STR:cond:likely  (realistic condition)
  //          لو   → STR:cond:hypo    (hypothetical / unreal)
  //          لولا → STR:cond:counter (counterfactual, if-not-for)
  "condition",
  "cond:likely",
  "cond:hypo",
  "cond:counter",

  "future", // EN: will/shall/going to, AR: سوف/سـ
  "past", // EN: was/were/had/did, AR: كان/قد + past verb
  "emphasis", // EN: trailing !, AR: trailing ! or إنّ/لقد
] as const;

export type StructureMarker = (typeof STRUCTURE_MARKERS)[number];

// ═══════════════════════════════════════════════════════════════
// 7. TOKEN EMISSION PRIORITY
// ═══════════════════════════════════════════════════════════════

/**
 * When processing each word, apply this priority (highest first):
 *
 * 1. Named Entity        → LIT:<surface>     (confidence 1.0)
 * 2. Numeric literal      → ROOT:size         (confidence 0.9)
 * 3. Relation word        → REL:<relation>    (confidence 1.0)
 * 4. Field + role found   → CMP:<field>:<role> (confidence 0.9)
 * 5. Field only           → ROOT:<field>      (confidence 0.8)
 * 6. Surface fallback     → LIT:<surface>     (confidence 0.5)
 *
 * STR tokens are detected first (whole-sentence) and prepended.
 */

// ═══════════════════════════════════════════════════════════════
// 8. LANGUAGE ANALYZER CONTRACT
// ═══════════════════════════════════════════════════════════════

/**
 * Every language analyzer must implement this interface.
 *
 * The analyzer receives raw text and produces CST tokens.
 * Internal implementation varies by language:
 *   - English: compromise.js → prefix/suffix tables
 *   - Arabic: camel-tools → trilateral root → field lookup + وزن → role
 *   - Future: any language with morphological analysis
 *
 * The output MUST conform to:
 *   - Token types: CMP, ROOT, REL, STR, LIT, SPECIAL only
 *   - Field names: from SEMANTIC_FIELDS only
 *   - Role names: from MORPHOLOGICAL_ROLES only
 *   - Structure markers: from STRUCTURE_MARKERS only
 *   - Special tokens: [PAD]=0, [UNK]=1, [BOS]=2, [EOS]=3, [SEP]=4
 */
export interface CSTToken {
  type: CSTTokenType;
  value: string; // full token string, e.g. "CMP:write:agent"
  field?: SemanticField; // semantic field (for CMP/ROOT)
  role?: MorphRole; // morphological role (for CMP)
  surface: string; // original word
  id: number; // vocab ID
  confidence: number; // 0.0–1.0
}

export interface CSTAnalyzerOutput {
  tokens: CSTToken[];
  ids: number[];
  text: string; // original input
}

export interface CSTAnalyzer {
  tokenize(text: string): CSTAnalyzerOutput;
  getVocabSize(): number;
}

// ═══════════════════════════════════════════════════════════════
// 9. FIELD ALIGNMENT NOTES
// ═══════════════════════════════════════════════════════════════

/**
 * Field unification (v1.0):
 *
 * English "consume" → merged into "food" (eating/drinking is food domain)
 * English "health"  → kept (medical domain, distinct from body)
 * Arabic  "food"    → canonical (replaces English "consume")
 * Arabic  "help"    → merged into "enable" (both mean facilitate/assist)
 * Arabic  "kind"    → merged into "contain" (classification/type)
 * Arabic  "sign"    → merged into "name" (label/symbol/mark)
 * Arabic  "take"    → kept (seize/capture, distinct from "possess")
 *
 * Result: 58 universal fields (see SEMANTIC_FIELDS above)
 */

// ═══════════════════════════════════════════════════════════════
// 10. ARABIC REL MAPPING (FUNC → REL alignment)
// ═══════════════════════════════════════════════════════════════

/**
 * The Arabic tokenizer previously used FUNC:<type> tokens.
 * Under the unified spec, these map to REL:<relation> tokens:
 *
 * FUNC:PREP → REL:<specific-prep>
 *   في → REL:in, من → REL:from, إلى → REL:to, على → REL:on
 *   عن → REL:about, مع → REL:with, بين → REL:between
 *   حول → REL:around, خلال → REL:through, حتى → REL:until
 *   عند → REL:at, فوق → REL:above, تحت → REL:under
 *   أمام → REL:before, خلف → REL:behind, بعد → REL:after
 *   قبل → REL:before, دون → REL:without, ضد → REL:against
 *   عبر → REL:across, ضمن → REL:within, منذ → REL:from
 *   نحو → REL:to, لدى → REL:at
 *
 * FUNC:CONJ → REL:<specific-conj>
 *   و → REL:and, أو → REL:or, لكن → REL:but, ثم → REL:then
 *   بل → REL:instead, إذا → REL:condition, لو → REL:condition
 *   حيث → REL:where, كما → REL:like, بينما → REL:contrast
 *   حين → REL:when, عندما → REL:when, مثل → REL:like
 *
 * FUNC:PRON → REL:<referential/pronominal>
 *   هو/هي/هم → LIT (personal pronouns → LIT like English)
 *   هذا/هذه/ذلك → REL:this/REL:these/REL:those
 *   الذي/التي/الذين → REL:which/REL:who
 *   ما → REL:what
 *
 * FUNC:DET → REL:<quantifier>
 *   كل → REL:all, بعض → REL:some, أي → REL:any
 *   جميع → REL:all, معظم → REL:most, أكثر → REL:more
 *   كثير → REL:many, قليل → REL:few, عدة → REL:several
 *   غير → REL:unlike, أحد → REL:some, نفس → LIT
 *
 * FUNC:NEG → STR:negation (sentence-level, not per-word)
 *   لا/لم/لن/ليس → triggers STR:negation
 *
 * FUNC:AUX → LIT (auxiliary verbs → LIT like English is/was/have)
 *   كان/يكون/أصبح/ظل/بات/صار → LIT:<word>
 *
 * FUNC:PART → LIT or STR
 *   قد/لقد → STR:emphasis or STR:past
 *   سوف → STR:future
 *   إن/أن/إنّ → LIT (subordinators)
 *
 * FUNC:NUM → ROOT:size (like English numerals)
 *
 * FUNC:ADV → REL:<specific>
 *   أيضاً → REL:also, فقط → REL:only, جداً → REL:emphasis
 *   تقريباً → REL:almost, حالياً → REL:now
 */
