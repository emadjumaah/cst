"""Import labeled Arabic roots from the Arabic Algebra Engine workspace.

Parses the 14 ``roots*.ts`` files in
``../arabic-algebra/arabic-algebra-engine/src/engine/data/`` and emits
``edge/artifacts/aae_roots.json`` — a mapping ``"ك.ت.ب" -> "write"`` that
``arabic_tokenizer.py`` loads at import time to enlarge
``ARABIC_ROOT_TO_FIELD``.

Mapping strategy
----------------
Each AAE ``RootData`` entry carries ``covers`` (free-text English gloss)
and ``keywords`` (list of English/Arabic synonyms). We match those
strings against the 55 CST field names already defined in
``arabic_tokenizer.py``; the first match wins. Entries with no match are
logged and skipped (they can be hand-mapped later).

Run:
    python edge/import_aae_roots.py

Output:
    edge/artifacts/aae_roots.json   — {root_dotted: cst_field}
    edge/artifacts/aae_roots_unmatched.json   — list of unmapped entries
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
AAE_DATA = (HERE / "../../arabic-algebra/arabic-algebra-engine/src/engine/data").resolve()
OUT = HERE / "artifacts"
OUT.mkdir(parents=True, exist_ok=True)

# ── CST fields currently defined in arabic_tokenizer.py ─────────
CST_FIELDS = [
    "write", "know", "speak", "think", "see", "feel", "move", "give", "take",
    "make", "destroy", "change", "exist", "time", "place", "possess", "trade",
    "fight", "enable", "govern", "create", "force", "body", "food", "nature",
    "weather", "animal", "plant", "color", "size", "measure", "connect",
    "contain", "open", "hold", "hide", "gather", "send", "social", "dwell",
    "need", "want", "decide", "fix", "rest", "person", "name", "art",
    "science", "tech", "material", "structure", "quality", "sport", "work",
]

# Lemma → field shortcuts (kept minimal; the generic matcher handles most).
# When the covers/keywords text contains one of these tokens, map to field.
LEMMA_TO_FIELD: dict[str, str] = {
    # body / health
    "health": "body", "heal": "body", "cure": "body", "disease": "body",
    "illness": "body", "sick": "body", "pain": "body", "medicine": "body",
    "doctor": "body", "blood": "body", "heart": "body",
    # speak
    "say": "speak", "tell": "speak", "ask": "speak", "question": "speak",
    "answer": "speak", "call": "speak", "shout": "speak", "word": "speak",
    "language": "speak", "voice": "speak",
    # know
    "knowledge": "know", "learn": "know", "understand": "know", "study": "know",
    "teach": "know", "school": "know", "memory": "know", "wisdom": "know",
    # think
    "reason": "think", "consider": "think", "believe": "think", "mind": "think",
    "idea": "think", "thought": "think", "philosophy": "think",
    # feel
    "love": "feel", "hate": "feel", "happy": "feel", "sad": "feel", "fear": "feel",
    "anger": "feel", "joy": "feel", "hope": "feel", "desire": "feel",
    "passion": "feel", "emotion": "feel",
    # move
    "go": "move", "come": "move", "walk": "move", "run": "move", "travel": "move",
    "journey": "move", "road": "move", "path": "move", "arrive": "move",
    "leave": "move", "depart": "move",
    # see
    "look": "see", "watch": "see", "observe": "see", "view": "see", "eye": "see",
    "vision": "see", "sight": "see",
    # create / make
    "build": "make", "construct": "make", "produce": "make", "form": "create",
    "invent": "create", "design": "create", "generate": "create",
    # destroy
    "kill": "destroy", "break": "destroy", "ruin": "destroy", "death": "destroy",
    "damage": "destroy", "burn": "destroy",
    # change
    "transform": "change", "modify": "change", "alter": "change",
    "develop": "change", "grow": "change", "evolve": "change",
    # exist / being
    "be": "exist", "being": "exist", "presence": "exist", "life": "exist",
    "alive": "exist", "truth": "exist", "reality": "exist",
    # time
    "year": "time", "day": "time", "hour": "time", "moment": "time",
    "history": "time", "period": "time", "epoch": "time", "age": "time",
    # place
    "land": "place", "country": "place", "city": "place", "region": "place",
    "area": "place", "location": "place", "home": "dwell", "house": "dwell",
    # trade / economy
    "buy": "trade", "sell": "trade", "money": "trade", "market": "trade",
    "price": "trade", "wealth": "trade", "cost": "trade", "business": "trade",
    # govern / authority
    "rule": "govern", "king": "govern", "state": "govern", "law": "govern",
    "power": "govern", "authority": "govern", "command": "govern",
    "leader": "govern", "politics": "govern",
    # fight / war
    "war": "fight", "battle": "fight", "attack": "fight", "defend": "fight",
    "enemy": "fight", "soldier": "fight", "weapon": "fight", "conflict": "fight",
    # food
    "eat": "food", "drink": "food", "meal": "food", "hungry": "food",
    "bread": "food", "taste": "food",
    # nature / weather
    "earth": "nature", "sea": "nature", "mountain": "nature", "river": "nature",
    "tree": "plant", "flower": "plant", "seed": "plant", "fruit": "plant",
    "rain": "weather", "wind": "weather", "sun": "weather", "cloud": "weather",
    "hot": "weather", "cold": "weather",
    # animal
    "bird": "animal", "fish": "animal", "dog": "animal", "horse": "animal",
    # person / social
    "man": "person", "woman": "person", "child": "person", "people": "social",
    "family": "social", "tribe": "social", "group": "social", "society": "social",
    # measure / size
    "number": "measure", "count": "measure", "weight": "measure",
    "long": "size", "short": "size", "big": "size", "small": "size",
    # art / science / tech
    "music": "art", "song": "art", "dance": "art", "paint": "art",
    "research": "science", "experiment": "science", "test": "science",
    "technology": "tech", "machine": "tech", "computer": "tech",
    # material / structure
    "iron": "material", "stone": "material", "wood": "material", "metal": "material",
    "order": "structure", "system": "structure", "class": "structure",
    # quality
    "good": "quality", "bad": "quality", "beautiful": "quality", "clean": "quality",
    # connect / contain / gather
    "link": "connect", "join": "connect", "bind": "connect",
    "hold": "hold", "carry": "hold", "collect": "gather", "assemble": "gather",
    "cover": "contain", "full": "contain", "empty": "contain",
    # send / give / take
    "give": "give", "gift": "give", "grant": "give", "offer": "give",
    "take": "take", "seize": "take", "steal": "take",
    "send": "send", "deliver": "send", "mail": "send",
    # possess / want / need
    "have": "possess", "own": "possess", "lose": "possess",
    "want": "want", "wish": "want", "desire": "want",
    "need": "need", "must": "need",
    # rest / work / sport
    "sleep": "rest", "stop": "rest", "pause": "rest",
    "work": "work", "do": "work", "act": "work",
    "play": "sport", "game": "sport", "win": "sport",
    # decide / fix
    "choose": "decide", "decide": "decide", "judge": "decide",
    "repair": "fix", "correct": "fix",
    # hide / open
    "hide": "hide", "secret": "hide", "cover": "hide",
    "open": "open", "close": "open", "door": "open",
    # name
    "name": "name", "title": "name",
    # force
    "strong": "force", "weak": "force", "push": "force", "pull": "force",
    # character (map to feel/quality)
    "character": "quality", "virtue": "quality", "honor": "quality",
}

# ── AAE domain → CST field default (fallback if lemma matching fails) ─
AAE_DOMAIN_DEFAULT = {
    "communication": "speak",
    "cognition": "know",
    "creation": "create",
    "transformation": "change",
    "movement": "move",
    "perception": "see",
    "possession": "possess",
    "social": "social",
    "abstract": "think",
    "body": "body",
    "health": "body",
    "character": "quality",
    "culture": "art",
    "economy": "trade",
    "everyday": None,           # no sensible default; require lemma match
    "food": "food",
    "home": "dwell",
    "governance": "govern",
    "nature": "nature",
    "physical": "material",
    "relations": "connect",
    "science": "science",
}

# ── Regex to extract RootData entries from .ts files ─────────────
ENTRY_RE = re.compile(
    r"\{\s*"
    r'arabic:\s*"([^"]+)"\s*,\s*'
    r'latin:\s*"[^"]*"\s*,\s*'
    r'domain:\s*"([^"]*)"\s*,\s*'
    r'semanticField:\s*"([^"]*)"\s*,\s*'
    r'resource:\s*"([^"]*)"\s*,\s*'
    r'covers:\s*"([^"]*)"\s*,\s*'
    r"keywords:\s*\[([^\]]*)\]",
    re.DOTALL,
)


def to_dotted_root(arabic: str) -> str | None:
    """Convert ``"كتب"`` → ``"ك.ت.ب"``. Skip anything that isn't 3-4 letters."""
    # Strip diacritics, spaces, leading particles
    s = re.sub(r"[\u064B-\u0652\u0670\s]", "", arabic)
    letters = [c for c in s if "\u0600" <= c <= "\u06FF"]
    if not (3 <= len(letters) <= 4):
        return None
    return ".".join(letters)


def parse_keywords(raw: str) -> list[str]:
    return [m.group(1) for m in re.finditer(r'"([^"]+)"', raw)]


def classify(domain: str, covers: str, semantic_field: str, keywords: list[str]) -> str | None:
    """Return a CST field name for an AAE entry, or None if unmatched."""
    # Assemble English search text from covers + first few keywords + field
    parts = [covers.lower(), semantic_field.lower()]
    for kw in keywords[:6]:
        kw_low = kw.lower()
        # Skip pure Arabic keywords (they don't carry English lemma info).
        if any("\u0600" <= c <= "\u06FF" for c in kw_low):
            continue
        parts.append(kw_low)
    text = " ".join(parts)

    # Direct field name hit (e.g., "write, record, document" contains "write")
    for field in CST_FIELDS:
        if re.search(rf"\b{field}\b", text):
            return field
    # Lemma shortcut table
    tokens = re.findall(r"[a-z]+", text)
    for tok in tokens:
        if tok in LEMMA_TO_FIELD:
            return LEMMA_TO_FIELD[tok]
    # Fallback: AAE domain default
    return AAE_DOMAIN_DEFAULT.get(domain)


def main() -> int:
    if not AAE_DATA.exists():
        print(f"ERROR: AAE data dir not found: {AAE_DATA}", file=sys.stderr)
        return 1

    files = sorted(AAE_DATA.glob("roots*.ts"))
    print(f"Scanning {len(files)} AAE root files in {AAE_DATA}")

    mapping: dict[str, str] = {}
    unmatched: list[dict] = []
    domain_hist: Counter[str] = Counter()
    field_hist: Counter[str] = Counter()
    collisions: list[tuple[str, str, str]] = []

    for f in files:
        src = f.read_text(encoding="utf-8")
        hits = list(ENTRY_RE.finditer(src))
        for m in hits:
            arabic, domain, sem_field, resource, covers, kw_raw = m.groups()
            keywords = parse_keywords(kw_raw)
            dotted = to_dotted_root(arabic)
            if not dotted:
                continue
            field = classify(domain, covers, sem_field, keywords)
            if field is None:
                unmatched.append({
                    "file": f.name, "arabic": arabic, "dotted": dotted,
                    "domain": domain, "semanticField": sem_field,
                    "covers": covers, "keywords": keywords[:4],
                })
                continue
            domain_hist[domain] += 1
            field_hist[field] += 1
            if dotted in mapping and mapping[dotted] != field:
                collisions.append((dotted, mapping[dotted], field))
                # keep first
                continue
            mapping[dotted] = field

    # Sort mapping for stable output
    out_path = OUT / "aae_roots.json"
    out_path.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    unmatched_path = OUT / "aae_roots_unmatched.json"
    unmatched_path.write_text(
        json.dumps(unmatched, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nMapped  roots : {len(mapping)}")
    print(f"Unmapped roots : {len(unmatched)}")
    print(f"Collisions     : {len(collisions)}")
    print(f"\nPer-field counts:")
    for field, n in field_hist.most_common():
        print(f"  {field:12s} {n}")
    print(f"\nPer-domain counts:")
    for domain, n in domain_hist.most_common():
        print(f"  {domain:20s} {n}")
    print(f"\nWrote {out_path.relative_to(HERE.parent)}")
    print(f"Wrote {unmatched_path.relative_to(HERE.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
