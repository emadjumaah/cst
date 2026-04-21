# CST Arabic Coverage — How the Tokenizer Works

> CST takes the algebraic structure of Arabic morphology and makes it the
> universal token format. Arabic is not just another language in CST —
> **Arabic is the blueprint.**

This document specifies exactly how an Arabic sentence is turned into a
sequence of CST tokens by [`edge/arabic_tokenizer.py`](../edge/arabic_tokenizer.py),
what each token type means, and how the pipeline round-trips back to
Arabic at inference time.

---

## 1. End-to-End Pipeline

```
                 ┌────────────────────────────────────────┐
                 │           Arabic sentence              │
                 │     وسيكتبُ الأطفالُ رسالةً للمعلمة       │
                 └────────────────────────────────────────┘
                                    │
                                    ▼
                 ┌────────────────────────────────────────┐
                 │  1. Normalize (strip diacritics, kash- │
                 │     ida; keep letters + shadda)        │
                 └────────────────────────────────────────┘
                                    │
                                    ▼
                 ┌────────────────────────────────────────┐
                 │  2. Sentence-level STR detection       │
                 │     لم/لن/ليس, إذا/لو, هل, ؟, !        │
                 └────────────────────────────────────────┘
                                    │
                                    ▼
                 ┌────────────────────────────────────────┐
                 │  3. Word-by-word analysis              │
                 │     CAMeL Tools → root/pattern/pos     │
                 │                   prc0..prc3, enc0,    │
                 │                   asp/per/gen/num      │
                 └────────────────────────────────────────┘
                                    │
                                    ▼
                 ┌────────────────────────────────────────┐
                 │  4. Per-word token emission            │
                 │  [prc2 → prc1 → prc0 → prc3]           │
                 │  core (CMP/ROOT/LIT)                   │
                 │  [FEAT gender/num | verb asp+pgn]      │
                 │  [FEAT:pron enc0]                      │
                 └────────────────────────────────────────┘
                                    │
                                    ▼
                 ┌────────────────────────────────────────┐
                 │  Integer id stream (GPT-2 causal LM)   │
                 │  model learns next-token prediction    │
                 └────────────────────────────────────────┘
                                    │
                     inference      ▼     lookup tables
                 ┌────────────────────────────────────────┐
                 │  5. Edge renderer                      │
                 │     primary token → Arabic word        │
                 │     FEAT tokens → morphological affix  │
                 └────────────────────────────────────────┘
                                    │
                                    ▼
                               Arabic text
```

Training objective is **causal LM on CST ids**: at each position the
model predicts the next CST token given all previous ones. The model
never sees raw text during training — it operates entirely in CST space.

---

## 2. Token Types

CST has five token classes plus five special tokens. Every Arabic word
decomposes into a sequence drawn from exactly these classes.

| Class  | Shape                        | Role                                                     |
| ------ | ---------------------------- | -------------------------------------------------------- |
| `CMP`  | `CMP:<field>:<role>`         | Content word with detected pattern (agent / patient / …) |
| `ROOT` | `ROOT:<field>`               | Content word, root matched but pattern unknown           |
| `REL`  | `REL:<relation>`             | Grammatical relation (preposition / conjunction / …)     |
| `STR`  | `STR:<marker>`               | Sentence-level structure (negation / condition / …)      |
| `FEAT` | `FEAT:<attribute>[:<value>]` | Morphological feature (definiteness / gender / pronoun)  |
| `LIT`  | `LIT:<surface>`              | Literal surface word (pronoun / aux / unknown / NER)     |

Special tokens `[PAD] [UNK] [BOS] [EOS] [SEP]` occupy ids 0–4.

### The Algebra

```
Root  × Pattern                = Meaning
ك.ت.ب × فاعل                    = كاتب (writer)
ك.ت.ب × مفعول                   = مكتوب (written)
ك.ت.ب × مفعلة                   = مكتبة (library)

Root  × Pattern                → CMP token
ك.ت.ب × فاعل                   → CMP:write:agent
ك.ت.ب × مفعول                  → CMP:write:patient
ك.ت.ب × مفعلة                  → CMP:write:place
```

### Worked Example

```
Input:     وسيكتبُ الأطفالُ رسالةً للمعلمة
Words:     وسيكتب    الأطفال    رسالة     للمعلمة

Word 1: وسيكتب  (and will-he-writes)
  prc2 = wa_conj            → REL:and
  prc1 = sa_fut             → STR:future
  root = ك.ت.ب              → field = write (imperfective verb,
  pos  = verb                  no pattern match → ROOT not CMP)
  asp  = i, per=3, gen=m, num=s
                            → ROOT:write
                            → FEAT:asp:i
                            → (3ms is default pgn → skipped)

Word 2: الأطفال  (the-children)
  prc0 = Al_det             → FEAT:def
  root = ط.ف.ل → person      → (no pattern match)
  pos  = noun, gen=m, num=p  → ROOT:person
                            → FEAT:p

Word 3: رسالة  (letter)
  pattern = فعالة → state    → CMP:send:state   (ر.س.ل → send)
  gen = f, num = s           → FEAT:f

Word 4: للمعلمة  (for-the-teacher-f)
  prc1 = li_prep            → REL:for
  prc0 = Al_det             → FEAT:def
  pattern = مفعّلة → causer   → CMP:know:causer (ع.ل.م → know)
  gen = f, num = s          → FEAT:f

Final token stream:
  [BOS] STR:future REL:and ROOT:write FEAT:asp:i
        FEAT:def ROOT:person FEAT:p
        CMP:send:state FEAT:f
        REL:for FEAT:def CMP:know:causer FEAT:f
  [EOS]
```

The sentence-level `STR:future` is emitted once at the head because it
scopes the whole clause; the per-word FEAT tokens scope only the word
they follow.

---

## 3. CMP Patterns — الأوزان

Arabic's وزن system maps directly to CMP roles. The tokenizer matches
the analyzer-provided `pattern` field against this table; vowel
diacritics are stripped, shadda (ّ) is preserved.

### Noun / Participle Patterns

| Pattern   | Role        | Example         | CST Token           |
| --------- | ----------- | --------------- | ------------------- |
| فَاعِل    | agent       | كاتب (writer)   | `CMP:write:agent`   |
| فَاعِلة   | agent (f)   | كاتبة           | `CMP:write:agent`   |
| فَاعِلون  | agent (pl)  | كاتبون          | `CMP:write:agent`   |
| فَاعِلات  | agent (fpl) | كاتبات          | `CMP:write:agent`   |
| فَواعِل   | agent (bpl) | كواتب           | `CMP:write:agent`   |
| مَفْعُول  | patient     | مكتوب (written) | `CMP:write:patient` |
| مَفْعُولة | patient (f) | مكتوبة          | `CMP:write:patient` |
| مَفْعَلة  | place       | مكتبة (library) | `CMP:write:place`   |
| مَفاعِل   | place (pl)  | مكاتب           | `CMP:write:place`   |
| مِفْعال   | instrument  | مفتاح (key)     | `CMP:open:place`    |
| فَعِيل    | quality     | كبير (big)      | `CMP:size:quality`  |
| فَعْلى    | quality (f) | كبرى            | `CMP:size:quality`  |

> Gender and number are recovered from CAMeL's `gen` / `num` features
> and re-emitted as `FEAT:f`, `FEAT:p`, `FEAT:d` tokens (see §6), so
> كاتب / كاتبة / كاتبات still decode to distinct surface forms even
> though they share the `CMP:write:agent` core.

### Verbal Nouns (المصادر)

| Pattern     | Role                 | Example           | CST Token              |
| ----------- | -------------------- | ----------------- | ---------------------- |
| فِعال       | instance             | كِتاب (book)      | `CMP:write:instance`   |
| فُعول       | instance             | دُخول (entry)     | `CMP:move:instance`    |
| فَعْل       | instance             | عِلْم (knowledge) | `CMP:know:instance`    |
| فِعالة      | state                | كِتابة (writing)  | `CMP:write:state`      |
| تَفْعيل     | instance (Form II)   | تَعليم (teaching) | `CMP:know:instance`    |
| اِنْفِعال   | instance (Form VII)  | انكسار            | `CMP:destroy:instance` |
| اِفْتِعال   | instance (Form VIII) | اجتماع            | `CMP:gather:instance`  |
| اِسْتِفْعال | instance (Form X)    | استخدام           | `CMP:work:instance`    |

### Derived Roles

| Pattern     | Role          | Example           | CST Token               |
| ----------- | ------------- | ----------------- | ----------------------- |
| فَعَّال     | intensifier   | كتّاب             | `CMP:write:intensifier` |
| مُفَعِّل    | causer (II)   | مُعلِّم (teacher) | `CMP:know:causer`       |
| مُسْتَفْعِل | seeker (X)    | مُستخدِم (user)   | `CMP:work:seeker`       |
| تَفاعُل     | mutual (VI)   | تعاوُن            | `CMP:enable:mutual`     |
| مُفاعَلة    | process (III) | مُكاتَبة          | `CMP:write:process`     |

Words whose root matches a semantic field but whose pattern is not in
this table fall back to `ROOT:<field>` (see §4). Conjugated verbs almost
always land here, with their aspect / person / gender / number carried
in companion FEAT tokens.

---

## 4. ROOT Tokens — الجذور

When a root is recognized but no pattern match → `ROOT:<field>`.

58 semantic fields total, mapped from Arabic trilateral roots. A
representative slice:

| Field | Sample Roots        |
| ----- | ------------------- |
| write | ك.ت.ب، خ.ط.ط، س.ج.ل |
| know  | ع.ل.م، ع.ر.ف، د.ر.س |
| speak | ق.و.ل، ك.ل.م، ح.د.ث |
| move  | م.ش.ي، ذ.ه.ب، ر.ج.ع |
| feel  | ح.ب.ب، ش.ع.ر، ح.ز.ن |

See the `_add("<field>", …)` calls at the top of
[`edge/arabic_tokenizer.py`](../edge/arabic_tokenizer.py) for the full 500+
root map.

### Weak Root Wildcards

Weak roots (containing و/ي/ا) morph across conjugations. CST builds a
wildcard index: every position occupied by a weak letter is mirrored
with `#`.

```
و.ج.د   → exist
#.ج.د   → exist   (و replaced by wildcard)
#.#.د   → exist   (double-weak fallback)
```

This lets conjugated forms whose weak consonants shift or drop still
resolve to the same field.

---

## 5. STR Tokens — Sentence-Level Structure

STR tokens mark whole-sentence properties. They are emitted **once** at
the head of the sequence (right after `[BOS]`), before the first word
token.

### Negation — split by grammatical scope

Each Arabic negation particle governs a different verb mood / nominal
case. They are **intentionally not merged** so the model can learn
Arabic syntax and produce the correct surface form at generation time.

| Arabic | Governs                 | CST Token         |
| ------ | ----------------------- | ----------------- |
| لا     | indicative / imperative | `STR:neg:general` |
| لم     | jussive verb            | `STR:neg:past`    |
| لن     | subjunctive verb        | `STR:neg:future`  |
| ليس    | accusative predicate    | `STR:neg:nominal` |

ما is ambiguous (negation / relative / exclamatory) and is handled
word-locally via the CAMeL POS tag — see §9.

### Conditionals — split by modality

| Arabic | Modality                    | CST Token          |
| ------ | --------------------------- | ------------------ |
| إذا    | likely / realistic          | `STR:cond:likely`  |
| لو     | hypothetical / unreal       | `STR:cond:hypo`    |
| لولا   | counterfactual (if-not-for) | `STR:cond:counter` |

### Other STR Markers

| Trigger              | CST Token      |
| -------------------- | -------------- |
| هل, trailing ؟ or ?  | `STR:question` |
| سوف                  | `STR:future`   |
| قد + past verb       | `STR:past`     |
| إنّ, لقد, trailing ! | `STR:emphasis` |

`STR:future` can also come from a word-attached proclitic (`prc1 =
sa_fut` / `Ha_fut`, e.g. سيكتب → `STR:future ROOT:write …`).

---

## 6. FEAT Tokens — Morphological Features

FEAT tokens expose sub-word morphology that a flat BPE tokenizer
collapses. They are emitted **immediately after** the content token
they modify.

| Token                | Meaning                               | Emitted when               |
| -------------------- | ------------------------------------- | -------------------------- |
| `FEAT:def`           | definite (ال)                         | `prc0 = Al_det`            |
| `FEAT:f`             | feminine noun / adj                   | `gen = f` and pos ≠ verb   |
| `FEAT:p`             | plural noun / adj                     | `num = p` and pos ≠ verb   |
| `FEAT:d`             | dual noun / adj                       | `num = d` and pos ≠ verb   |
| `FEAT:asp:p`         | perfective verb                       | pos = verb, `asp = p`      |
| `FEAT:asp:i`         | imperfective verb                     | pos = verb, `asp = i`      |
| `FEAT:asp:c`         | imperative verb                       | pos = verb, `asp = c`      |
| `FEAT:1s`…`FEAT:3fp` | verb subject person-gender-number     | verbs; default 3ms omitted |
| `FEAT:pron:3ms`…     | pronominal enclitic (ه / ها / هم / …) | `enc0 ≠ 0`                 |

The pgn / pron tags use a compact code: first char is person (1/2/3),
middle char is gender (m/f — omitted for 1st person and non-gendered
duals/plurals), last char is number (s/p/d). Examples:

| Code  | Meaning               |
| ----- | --------------------- |
| `1s`  | I                     |
| `1p`  | we                    |
| `2ms` | you (m.sg)            |
| `3fs` | she / her / it (f.sg) |
| `3mp` | they / them (m.pl)    |
| `3fp` | they / them (f.pl)    |
| `3d`  | they / them (dual)    |

---

## 7. REL Tokens — حروف الجر، العطف، والاستفهام

Every Arabic function word that expresses a grammatical relationship
between content words maps to a REL token. Function words that appear
as fused clitics (بـ، لـ، كـ، و، ف) also produce REL tokens through the
clitic decomposition in §8.

### Prepositions

| Arabic | CST Token     | Arabic | CST Token     |
| ------ | ------------- | ------ | ------------- |
| في     | `REL:in`      | فوق    | `REL:above`   |
| من     | `REL:from`    | تحت    | `REL:under`   |
| إلى    | `REL:to`      | أمام   | `REL:infront` |
| على    | `REL:on`      | خلف    | `REL:behind`  |
| عن     | `REL:about`   | بعد    | `REL:after`   |
| مع     | `REL:with`    | قبل    | `REL:before`  |
| بين    | `REL:between` | دون    | `REL:without` |
| حول    | `REL:around`  | ضد     | `REL:against` |
| خلال   | `REL:through` | عبر    | `REL:across`  |
| منذ    | `REL:since`   | ضمن    | `REL:within`  |
| حتى    | `REL:until`   | لأجل   | `REL:for`     |
| نحو    | `REL:to`      | لدى    | `REL:at`      |
| عند    | `REL:at`      |        |               |

> **Splits worth remembering.** منذ is kept distinct from من (temporal
> vs. generic source) and أمام is kept distinct from قبل (spatial vs.
> temporal). Collapsing them erases meaning the model can never recover
> at generation time.

### Conjunctions

| Arabic             | CST Token      |
| ------------------ | -------------- |
| و                  | `REL:and`      |
| أو, أم             | `REL:or`       |
| ثم                 | `REL:then`     |
| لكن, لكنّ          | `REL:but`      |
| بل                 | `REL:instead`  |
| إذ                 | `REL:as`       |
| كي                 | `REL:for`      |
| حيث                | `REL:where`    |
| لأن                | `REL:causes`   |
| بينما              | `REL:contrast` |
| كما, مثل, كأنّ     | `REL:like`     |
| حين, عندما, لما    | `REL:when`     |
| لعلّ               | `REL:maybe`    |
| إلا, سوى, عدا, خلا | `REL:except`   |
| إنما, فقط          | `REL:only`     |

### Demonstratives & Relatives

| Arabic                | CST Token   |
| --------------------- | ----------- |
| هذا, هذه              | `REL:this`  |
| ذلك, تلك              | `REL:those` |
| هؤلاء                 | `REL:these` |
| الذي, التي            | `REL:which` |
| الذين, اللذين, اللاتي | `REL:who`   |

### Quantifiers

| Arabic         | CST Token     |
| -------------- | ------------- |
| كل, جميع, سائر | `REL:all`     |
| بعض, أحد       | `REL:some`    |
| أي             | `REL:any`     |
| كلا            | `REL:both`    |
| معظم, أغلب     | `REL:most`    |
| عدة            | `REL:several` |
| كثير           | `REL:many`    |
| قليل           | `REL:few`     |
| أكثر           | `REL:more`    |
| أقل            | `REL:less`    |
| غير            | `REL:unlike`  |

### Adverbs

| Arabic | CST Token      |
| ------ | -------------- |
| أيضا   | `REL:also`     |
| جدا    | `REL:emphasis` |
| تقريبا | `REL:almost`   |
| حاليا  | `REL:now`      |

---

## 8. Clitic Decomposition

Arabic orthography fuses particles onto content words. The sequence
`وسبللمعلمين` ("and by / for / for the teachers") is one graphical
word but four morphemes. CAMeL Tools exposes them as features; CST
emits each as its own token so the model sees the real grammatical
structure.

| Feature | What it carries              | Example values                                                             |
| ------- | ---------------------------- | -------------------------------------------------------------------------- |
| `prc2`  | conjunction proclitic        | `wa_conj`, `fa_conj`, `wa_sub`                                             |
| `prc1`  | preposition / future / emph. | `bi_prep`, `li_prep`, `ka_prep`, `sa_fut`, `Ha_fut`, `la_emph`, `min_prep` |
| `prc0`  | article / attached negation  | `Al_det`, `mA_neg`, `lA_neg`                                               |
| `prc3`  | question particle            | `>a_ques`                                                                  |
| `enc0`  | pronominal enclitic          | `3ms_poss`, `1s_dobj`, `3fp_pron`                                          |

Emission order is outer → inner: **prc2 → prc1 → prc0 → prc3**. Full
mapping tables live in [`edge/arabic_tokenizer.py`](../edge/arabic_tokenizer.py)
(`PRC0_TOKENS`, `PRC1_TOKENS`, `PRC2_TOKENS`, `PRC3_TOKENS`, `enc0_feat`).

### Worked example: `وبالكتابه`

> و + ب + ال + كتاب + ه = "and with his book"

| Feature | Value                    | Emits                |
| ------- | ------------------------ | -------------------- |
| `prc2`  | `wa_conj`                | `REL:and`            |
| `prc1`  | `bi_prep`                | `REL:with`           |
| `prc0`  | `Al_det`                 | `FEAT:def`           |
| core    | root ك.ت.ب, pattern فعال | `CMP:write:instance` |
| `enc0`  | `3ms_poss`               | `FEAT:pron:3ms`      |

```
وبالكتابه → REL:and REL:with FEAT:def CMP:write:instance FEAT:pron:3ms
```

Five tokens for one orthographic word — but every piece of meaning is
preserved and the model sees each morpheme as a first-class unit.

---

## 9. Fast-Path Words

A small set of words bypass morphological analysis because their
semantics are fixed:

- **STR triggers** (`ARABIC_STR_TRIGGERS` in `edge/arabic_tokenizer.py`) — emitted
  once at sentence level (see §5), consumed at the word position.
- **REL fixed map** (`ARABIC_REL_MAP`) — prepositions, conjunctions,
  quantifiers listed in §7 emit their REL directly. These are
  standalone-word forms; their clitic-fused variants (بـ، لـ، …) flow
  through the prc1/prc2 pipeline instead.
- **LIT particles** (`ARABIC_LIT_WORDS`) — personal pronouns,
  auxiliaries, subordinators, vocative يا.
- **Numerals** (`ARABIC_NUMERALS`) — all emit `ROOT:size`.

### Disambiguating ما

ما is ambiguous (negation / interrogative / relative). The tokenizer
uses the CAMeL POS to choose:

| POS                             | Emits             |
| ------------------------------- | ----------------- |
| `part_neg`                      | `STR:neg:general` |
| `pron_interrog`, `adv_interrog` | `REL:what`        |
| `pron_rel`, `adv_rel`           | `REL:which`       |
| anything else                   | `REL:what`        |

Note: ما is also listed in `ARABIC_REL_MAP` as a safety fallback when no
analysis is available, which lands on `REL:what`.

---

## 10. Fallback Hierarchy

For each content word the tokenizer walks this priority ladder:

| Step | Condition                                     | Token emitted            |
| ---- | --------------------------------------------- | ------------------------ |
| 1    | Word is a fast-path STR / REL / LIT / numeral | `STR:` / `REL:` / `LIT:` |
| 2    | CAMeL gives no usable analysis                | `LIT:<surface>`          |
| 3    | `pos = noun_prop` (named entity)              | `LIT:<surface>`          |
| 4    | Root known + pattern matches table            | `CMP:<field>:<role>`     |
| 5    | Root known + pos matches `POS_TO_ROLE`        | `CMP:<field>:<role>`     |
| 6    | Root known, no role                           | `ROOT:<field>`           |
| 7    | Root unknown                                  | `LIT:<surface>`          |

FEAT tokens are attached only after steps 4–6 succeed (content words
with recovered morphology).

---

## 11. How Decoding Works (Edge Inference)

Training runs entirely in CST-id space (GPT-2 causal LM over
`train-100000.jsonl`). At inference the model emits a stream of CST
ids which must be rendered back to Arabic by the edge demo.

The lookup tables are produced by
[`edge/build_lookups.py`](../edge/build_lookups.py):

- **`word2tok.json`** — Arabic surface word → most-common CST token
  _sequence_ (full sequence, including FEAT tokens).
- **`tok2word.json`** — primary CST token → most-common Arabic surface
  word. FEAT tokens are skipped on this side because they are affixes,
  not standalone words.

The edge renderer reads the generated token stream and:

1. Emits the surface word for each primary token via `tok2word`.
2. Treats FEAT tokens as modifiers: `FEAT:def` prepends ال, `FEAT:pron:3ms`
   appends ه, etc. (v1 renderer is intentionally simple; v2 can use a
   proper morphological generator.)

This means the tokenizer is **not byte-reversible** — CST is lossy by
design, like BPE is lossy about casing. The goal is faithful _semantic_
round-trip, not exact orthographic round-trip. The model's job at
inference is to pick the CST tokens whose composition produces natural
Arabic, not to regenerate the literal input.

---

## 12. Expected Token Distribution

Over 100k Arabic Wikipedia sentences, typical distribution:

| Class    | Share     | Notes                                    |
| -------- | --------- | ---------------------------------------- |
| `CMP:*`  | 25 – 35 % | Content words where pattern matched      |
| `ROOT:*` | 15 – 20 % | Conjugated verbs / un-patterned forms    |
| `REL:*`  | 15 – 20 % | Function words + decomposed prepositions |
| `FEAT:*` | 15 – 25 % | New — morphological richness             |
| `STR:*`  | 3 – 6 %   | Sentence-level markers                   |
| `LIT:*`  | 10 – 15 % | Pronouns / auxiliaries / NER / unknowns  |

FEAT's large share reflects the new clitic and feature decomposition;
it replaces the long `LIT:<fused-word>` tail that older versions
produced.

---

## 13. What We Intentionally Exclude

| Feature                 | Reason                                                       |
| ----------------------- | ------------------------------------------------------------ |
| إعراب (case endings)    | Dropped in modern Arabic, unreliable in unvoweled text       |
| Broken plural surface   | Collapsed into `CMP + FEAT:p`; surface form comes via lookup |
| Full dependency parsing | Beyond tokenizer scope — model learns syntax from context    |

---

## 14. File Map

| File                                              | Role                                                      |
| ------------------------------------------------- | --------------------------------------------------------- |
| `edge/arabic_tokenizer.py`                        | **Canonical Arabic CST tokenizer library** (data + class) |
| `edge/training/tokenize_1m.py`                    | Experiment CLI: tokenize 1M Wikipedia sentences           |
| `edge/training/tests/test_tokenizer.py`           | Unit tests (mocked analyzer, no DB needed)                |
| `edge/build_lookups.py`                           | word ↔ token-sequence lookup tables                       |
| `src/tokenizer/cst-spec.ts`                       | TypeScript spec: token classes, categories                |
| `data/tokenized/cst-ar/train-*.jsonl`             | Produced training data                                    |
| `edge/demo/public/model/{word2tok,tok2word}.json` | Lookup tables shipped to the browser demo                 |

Legacy scripts under `training/arabic_experiment*.py` use the older
`FUNC:PREP` / `FUNC:CONJ` labeling scheme and are **not** aligned with
the current spec. Use `edge/arabic_tokenizer.py` as the source of
truth; `edge/training/tokenize_1m.py` is only a driver for the 1M
Wikipedia experiment — write new experiment scripts as thin wrappers
that import from `edge/arabic_tokenizer.py`, do not copy its contents.

---

## 15. One-Sentence Summary

> Every Arabic surface word decomposes into a **prefix stack of
> clitics** (REL / STR / FEAT:def) plus a **core content token** (CMP /
> ROOT / LIT) plus an **inflection tail** (FEAT gender / number /
> aspect / person / pronominal enclitic). The model never sees opaque
> fused words — it sees the algebra.

---

## 16. Companion Tool: `arabic-algebra-engine`

The sister project [`arabic-algebra-engine`](../../arabic-algebra/arabic-algebra-engine/)
is a zero-parameter, deterministic **symbolic encoder** that maps short
Arabic (or English) intent phrases into the same root × pattern algebra
described above. It is **not a competitor to the CST tokenizer** — it is
a producer of CST-compatible token sequences for cases where the input
is a short command/intent rather than free-running prose.

Relevance for CST:

| What the engine provides                              | How CST consumes it                                             |
| ----------------------------------------------------- | --------------------------------------------------------------- |
| 820 hand-curated triconsonantal roots × 29 domains    | Expanded candidate set for the CST semantic-field map           |
| Root → resource/domain labels                         | Supervised labels for field-level token classification          |
| `toCST(token)` bridge returning `[BOS] … [EOS]` seqs  | Direct training data for the reasoning-level tokenizer `T_R^ar` |
| Deterministic encoder (no CAMeL dependency)           | Lightweight fallback when CAMeL analysis is unavailable         |
| 74 action rules + agent/decomposer + domain packs     | Ground-truth CoT traces for the reasoning-model training set    |

The bridge lives at
[`src/engine/core/cst_bridge.ts`](../../arabic-algebra/arabic-algebra-engine/src/engine/core/cst_bridge.ts)
and is exported as `toCST` from the package root:

```ts
import { encodeLocal, toCST } from "arabic-algebra-engine";

const token = encodeLocal("أرسل التقرير إلى المدير غدًا");
const cst   = toCST(token);
// cst.tokens → ["[BOS]", "CMP:message:patient", "LIT:time:tomorrow", …, "[EOS]"]
```

The output shape matches this document's reasoning-level contract
exactly: `ROOT:<field>` / `CMP:<field>:<role>` / `REL:<type>` /
`STR:<marker>` / `LIT:<value>`. That means the engine's emissions can
be concatenated directly with the tokenizer's output for joint training
without any further re-mapping.

