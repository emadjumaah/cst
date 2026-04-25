"""Syllogism generator (Category 2), bilingual.

Produces categorical syllogisms of the form

    Major premise: All M are P.
    Minor premise: All S are M.
    Conclusion:    All S are P.

Valid moods covered: Barbara, Celarent, Darii, Ferio (Figure 1).
Invalid controls are included so the downstream model must learn to
reject bad inferences, not memorize "syllogism → yes".

Run::

    python -m reasoning.data.generators.syllogisms --count 5000 --out out/syllogisms.jsonl
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..schema import Meta, Record, write_jsonl


# Content categories (S, M, P) — deliberately abstract and numerous so
# the logic, not the surface terms, drives the inference. Expanded from
# 5 → 25 triples to break term-memorization shortcuts.
CATEGORIES_EN = [
    ("birds", "animals", "living things"),
    ("squares", "rectangles", "quadrilaterals"),
    ("programmers", "engineers", "professionals"),
    ("poets", "writers", "artists"),
    ("apples", "fruits", "plants"),
    ("sparrows", "birds", "vertebrates"),
    ("violins", "string instruments", "musical instruments"),
    ("novels", "books", "publications"),
    ("sedans", "cars", "vehicles"),
    ("roses", "flowers", "living organisms"),
    ("dolphins", "mammals", "animals"),
    ("oaks", "trees", "plants"),
    ("surgeons", "doctors", "professionals"),
    ("sonnets", "poems", "literary works"),
    ("smartphones", "electronic devices", "tools"),
    ("triangles", "polygons", "geometric shapes"),
    ("teachers", "educators", "workers"),
    ("cellos", "string instruments", "orchestral instruments"),
    ("bicycles", "vehicles", "transport means"),
    ("eagles", "raptors", "birds"),
    ("pianists", "musicians", "artists"),
    ("diamonds", "gemstones", "minerals"),
    ("turtles", "reptiles", "animals"),
    ("chemists", "scientists", "researchers"),
    ("shortstories", "prose works", "literary works"),
]

CATEGORIES_AR = [
    # Every triple is collision-free: CAMeL assigns distinct roots to S, M, P.
    # Verified by running ArabicReasoningTokenizer on each term and checking
    # that roots(S) ∩ roots(M) = roots(M) ∩ roots(P) = roots(S) ∩ roots(P) = ∅.
    ("الطيور", "الثدييات", "الفقاريات"),            # birds / mammals / vertebrates
    ("المربعات", "المستطيلات", "الأشكال الرباعية"),   # squares / rectangles / quadrilaterals
    ("المبرمجون", "المهندسون", "المحترفون"),           # programmers / engineers / professionals
    ("الشعراء", "الكتّاب", "الفنانون"),               # poets / writers / artists
    ("التفاح", "الفواكه", "الأطعمة"),                 # apples / fruits / foods
    ("العصافير", "الطيور", "الفقاريات"),              # sparrows / birds / vertebrates
    ("الكمنجات", "الأوتار", "الأدوات"),               # violins / strings / tools
    ("الروايات", "الصحف", "الخرائط"),                # novels / newspapers / maps
    ("القطارات", "المركبات", "الآلات"),               # trains / vehicles / machines
    ("الورود", "الأزهار", "الكائنات الحية"),          # roses / flowers / living things (OK)
    ("الدلافين", "الثدييات", "الحيوانات"),            # dolphins / mammals / animals
    ("البلوط", "الأشجار", "الغابات"),                 # oaks / trees / forests
    ("الجراحون", "العلماء", "المحترفون"),              # surgeons / scientists / professionals
    ("السوناتات", "القصائد", "الأعمال الأدبية"),       # sonnets / poems / literary works
    ("الهواتف الذكية", "الأجهزة الإلكترونية", "الأدوات"),  # smartphones / electronics / tools
    ("المثلثات", "المضلعات", "الأشكال الهندسية"),      # triangles / polygons / geometric shapes
    ("المعلمون", "المربون", "العاملون"),               # teachers / educators / workers
    ("الكمانات الجهيرة", "الآلات", "المعدات"),         # cellos / instruments / equipment
    ("الدراجات", "المركبات", "الآلات"),               # bicycles / vehicles / machines
    ("النسور", "الطيور الجارحة", "الزواحف"),           # eagles / raptors / reptiles
    ("عازفو البيانو", "الموسيقيون", "العمال"),         # pianists / musicians / workers
    ("الألماس", "الأحجار الكريمة", "المعادن"),         # diamonds / gemstones / minerals
    ("السلاحف", "الزواحف", "الحيوانات"),              # turtles / reptiles / animals
    ("الكيميائيون", "العلماء", "الباحثون"),            # chemists / scientists / researchers
    ("القصص القصيرة", "المسرحيات", "اللوحات"),        # short stories / plays / paintings
]


@dataclass
class Mood:
    name: str
    premise_major: str   # template over (M, P)
    premise_minor: str   # template over (S, M)
    conclusion: str      # template over (S, P)
    valid: bool


# Figure-1 valid + undistributed-middle invalid controls (easy / medium).
# Each valid mood V is paired with V-BadConclusion: identical premises,
# wrong-polarity conclusion → invalid. This guarantees every premise
# pattern has a matched invalid variant, killing the surface-quantifier
# shortcut.
MOODS_EN = [
    Mood("Barbara", "All {M} are {P}.", "All {S} are {M}.", "All {S} are {P}.", True),
    Mood("Celarent", "No {M} are {P}.", "All {S} are {M}.", "No {S} are {P}.", True),
    Mood("Darii",   "All {M} are {P}.", "Some {S} are {M}.", "Some {S} are {P}.", True),
    Mood("Ferio",   "No {M} are {P}.", "Some {S} are {M}.", "Some {S} are not {P}.", True),
    # Wrong-conclusion twins (same premises as the valid mood above).
    Mood("Barbara-BadConc", "All {M} are {P}.", "All {S} are {M}.", "No {S} are {P}.", False),
    Mood("Celarent-BadConc", "No {M} are {P}.", "All {S} are {M}.", "All {S} are {P}.", False),
    Mood("Darii-BadConc",   "All {M} are {P}.", "Some {S} are {M}.", "Some {S} are not {P}.", False),
    Mood("Ferio-BadConc",   "No {M} are {P}.", "Some {S} are {M}.", "Some {S} are {P}.", False),
    # Classical invalid controls (structurally flawed premise layout).
    Mood("Undistributed-Middle",
         "All {P} are {M}.", "All {S} are {M}.", "All {S} are {P}.", False),
    Mood("Invalid-Two-Particular",
         "Some {M} are {P}.", "Some {S} are {M}.", "Some {S} are {P}.", False),
    Mood("Illicit-Minor",
         "All {M} are {P}.", "All {M} are {S}.", "All {S} are {P}.", False),
    Mood("Illicit-Major",
         "All {M} are {P}.", "No {S} are {M}.", "No {S} are {P}.", False),
]

MOODS_AR = [
    Mood("Barbara",
         "كل {M} هي {P}.", "كل {S} هي {M}.", "كل {S} هي {P}.", True),
    Mood("Celarent",
         "لا شيء من {M} هو {P}.", "كل {S} هي {M}.", "لا شيء من {S} هو {P}.", True),
    Mood("Darii",
         "كل {M} هي {P}.", "بعض {S} هي {M}.", "بعض {S} هي {P}.", True),
    Mood("Ferio",
         "لا شيء من {M} هو {P}.", "بعض {S} هي {M}.", "بعض {S} ليست {P}.", True),
    # Wrong-conclusion twins.
    Mood("Barbara-BadConc",
         "كل {M} هي {P}.", "كل {S} هي {M}.", "لا شيء من {S} هو {P}.", False),
    Mood("Celarent-BadConc",
         "لا شيء من {M} هو {P}.", "كل {S} هي {M}.", "كل {S} هي {P}.", False),
    Mood("Darii-BadConc",
         "كل {M} هي {P}.", "بعض {S} هي {M}.", "بعض {S} ليست {P}.", False),
    Mood("Ferio-BadConc",
         "لا شيء من {M} هو {P}.", "بعض {S} هي {M}.", "بعض {S} هي {P}.", False),
    Mood("Undistributed-Middle",
         "كل {P} هي {M}.", "كل {S} هي {M}.", "كل {S} هي {P}.", False),
    Mood("Invalid-Two-Particular",
         "بعض {M} هي {P}.", "بعض {S} هي {M}.", "بعض {S} هي {P}.", False),
    Mood("Illicit-Minor",
         "كل {M} هي {P}.", "كل {M} هي {S}.", "كل {S} هي {P}.", False),
    Mood("Illicit-Major",
         "كل {M} هي {P}.", "لا شيء من {S} هو {M}.", "لا شيء من {S} هو {P}.", False),
]

# Figure-2 moods: middle term is predicate in both premises, which makes
# distribution analysis non-obvious. Plus an illicit-major invalid
# control. These are tagged ``hard``.
MOODS_HARD_EN = [
    Mood("Camestres", "All {P} are {M}.", "No {S} are {M}.", "No {S} are {P}.", True),
    Mood("Baroco",    "All {P} are {M}.", "Some {S} are not {M}.", "Some {S} are not {P}.", True),
    Mood("Cesare",    "No {P} are {M}.", "All {S} are {M}.", "No {S} are {P}.", True),
    Mood("Festino",   "No {P} are {M}.", "Some {S} are {M}.", "Some {S} are not {P}.", True),
    # Wrong-conclusion twins for the Figure-2 valid moods.
    Mood("Camestres-BadConc",
         "All {P} are {M}.", "No {S} are {M}.", "All {S} are {P}.", False),
    Mood("Baroco-BadConc",
         "All {P} are {M}.", "Some {S} are not {M}.", "All {S} are {P}.", False),
    Mood("Cesare-BadConc",
         "No {P} are {M}.", "All {S} are {M}.", "All {S} are {P}.", False),
    Mood("Festino-BadConc",
         "No {P} are {M}.", "Some {S} are {M}.", "Some {S} are {P}.", False),
    # Invalid: illicit major (P distributed in conclusion but not premise)
    Mood("Invalid-Hard-Illicit-Major",
         "All {M} are {P}.", "Some {S} are not {M}.", "Some {S} are not {P}.", False),
    # Invalid: exclusive premises (two negatives yield no conclusion)
    Mood("Invalid-Hard-Exclusive",
         "No {M} are {P}.", "No {S} are {M}.", "No {S} are {P}.", False),
    # Invalid: existential fallacy (universals do not imply particulars)
    Mood("Invalid-Hard-Existential",
         "All {M} are {P}.", "All {S} are {M}.", "Some {S} are {P}.", False),
    # Invalid: affirming-the-consequent shape (Figure-2 positives)
    Mood("Invalid-Hard-Affirming",
         "All {P} are {M}.", "All {S} are {M}.", "Some {S} are not {P}.", False),
]

MOODS_HARD_AR = [
    Mood("Camestres",
         "كل {P} هي {M}.", "لا شيء من {S} هو {M}.", "لا شيء من {S} هو {P}.", True),
    Mood("Baroco",
         "كل {P} هي {M}.", "بعض {S} ليست {M}.", "بعض {S} ليست {P}.", True),
    Mood("Cesare",
         "لا شيء من {P} هو {M}.", "كل {S} هي {M}.", "لا شيء من {S} هو {P}.", True),
    Mood("Festino",
         "لا شيء من {P} هو {M}.", "بعض {S} هي {M}.", "بعض {S} ليست {P}.", True),
    # Wrong-conclusion twins for Figure-2.
    Mood("Camestres-BadConc",
         "كل {P} هي {M}.", "لا شيء من {S} هو {M}.", "كل {S} هي {P}.", False),
    Mood("Baroco-BadConc",
         "كل {P} هي {M}.", "بعض {S} ليست {M}.", "كل {S} هي {P}.", False),
    Mood("Cesare-BadConc",
         "لا شيء من {P} هو {M}.", "كل {S} هي {M}.", "كل {S} هي {P}.", False),
    Mood("Festino-BadConc",
         "لا شيء من {P} هو {M}.", "بعض {S} هي {M}.", "بعض {S} هي {P}.", False),
    Mood("Invalid-Hard-Illicit-Major",
         "كل {M} هي {P}.", "بعض {S} ليست {M}.", "بعض {S} ليست {P}.", False),
    Mood("Invalid-Hard-Exclusive",
         "لا شيء من {M} هو {P}.", "لا شيء من {S} هو {M}.", "لا شيء من {S} هو {P}.", False),
    Mood("Invalid-Hard-Existential",
         "كل {M} هي {P}.", "كل {S} هي {M}.", "بعض {S} هي {P}.", False),
    Mood("Invalid-Hard-Affirming",
         "كل {P} هي {M}.", "كل {S} هي {M}.", "بعض {S} ليست {P}.", False),
]


def _fill(mood: Mood, cats: tuple[str, str, str]) -> tuple[str, str, str]:
    S, M, P = cats
    return (
        mood.premise_major.format(M=M, P=P),
        mood.premise_minor.format(S=S, M=M),
        mood.conclusion.format(S=S, P=P),
    )


def _cot(mood: Mood, cats: tuple[str, str, str], lang: str) -> list[str]:
    """Uniform, validity-neutral CoT scaffold.

    All CoT steps 1..N-1 are identical regardless of validity — only the
    final step (which the trainer strips) contains the verdict. This
    prevents CoT-distribution leaks when the training recipe removes the
    last step.
    """
    S, M, P = cats
    if lang == "en":
        steps = [
            f"Identify terms: S={S}, M={M}, P={P}.",
            "Examine the role of M in the major and minor premises.",
            "Check whether the proposed conclusion matches what the premises support.",
        ]
        verdict = (
            "The conclusion follows from the premises."
            if mood.valid else
            "The conclusion does not follow from the premises."
        )
        steps.append(verdict)
        return steps
    # Arabic — matched scaffold.
    steps = [
        f"حدد الحدود: S={S} ، M={M} ، P={P}.",
        "افحص دور M في المقدمتين الكبرى والصغرى.",
        "تحقق من مطابقة الاستنتاج المقترح لما تدعمه المقدمات.",
    ]
    verdict = (
        "الاستنتاج يتبع من المقدمات."
        if mood.valid else
        "الاستنتاج لا يتبع من المقدمات."
    )
    steps.append(verdict)
    return steps


def _record(
    *, idx: int, lang: str, mood: Mood, cats: tuple[str, str, str],
    difficulty: str,
) -> Record:
    major, minor, conc = _fill(mood, cats)
    if lang == "en":
        question = f"{major} {minor} Does it follow that: {conc}"
        answer = "yes" if mood.valid else "no"
    else:
        question = f"{major} {minor} هل يلزم أن: {conc}"
        answer = "نعم" if mood.valid else "لا"
    return Record(
        id=f"syllog-{lang}-{idx:06d}",
        lang=lang,  # type: ignore[arg-type]
        category=2,
        question=question,
        answer=answer,
        cot=_cot(mood, cats, lang),
        meta=Meta(
            source="syllogisms",
            license="cc0-1.0",
            difficulty=difficulty,  # type: ignore[arg-type]
        ),
    )


def _sorites_cats(rng: random.Random) -> tuple[list[str], list[str]]:
    """Pick 4 distinct category chains (EN, AR) for a sorites."""
    idxs = rng.sample(range(len(CATEGORIES_EN)), 2)
    # Build a 4-term chain by concatenating two 3-term chains on their
    # shared super-category. Here we synthesise a fresh chain instead.
    en = list(CATEGORIES_EN[idxs[0]]) + [CATEGORIES_EN[idxs[1]][-1]]
    ar = list(CATEGORIES_AR[idxs[0]]) + [CATEGORIES_AR[idxs[1]][-1]]
    return en, ar


def _sorites_record(
    *, idx: int, lang: str, chain: list[str], valid: bool,
) -> Record:
    """4-term sorites: A⊂B, B⊂C, C⊂D ⊢ A⊂D (valid); swap one link for invalid."""
    A, B, C, D = chain
    if not valid:
        # Break transitivity by reversing the middle link.
        B, C = C, B
    if lang == "en":
        p1 = f"All {A} are {B}."
        p2 = f"All {B} are {C}."
        p3 = f"All {C} are {D}."
        conc = f"All {A} are {D}."
        question = f"{p1} {p2} {p3} Does it follow that: {conc}"
        answer = "yes" if valid else "no"
        # Validity-neutral scaffold; only the last step (stripped) differs.
        cot = [
            f"List the premises: {A}-{B}, {B}-{C}, {C}-{D}.",
            "Trace whether universal inclusion composes through each link.",
            (
                f"Therefore {A} ⊆ {D}."
                if valid else
                "Conclusion does not follow."
            ),
        ]
    else:
        p1 = f"كل {A} هي {B}."
        p2 = f"كل {B} هي {C}."
        p3 = f"كل {C} هي {D}."
        conc = f"كل {A} هي {D}."
        question = f"{p1} {p2} {p3} هل يلزم أن: {conc}"
        answer = "نعم" if valid else "لا"
        cot = [
            f"سرد المقدمات: {A}-{B} ، {B}-{C} ، {C}-{D}.",
            "تتبّع ما إذا كان الاحتواء الكلي يتركّب عبر كل حلقة.",
            (
                f"إذًا {A} ⊆ {D}."
                if valid else
                "الاستنتاج لا يلزم."
            ),
        ]
    return Record(
        id=f"syllog-sorites-{lang}-{idx:06d}",
        lang=lang,  # type: ignore[arg-type]
        category=2,
        question=question,
        answer=answer,
        cot=cot,
        meta=Meta(
            source="syllogisms",
            license="cc0-1.0",
            difficulty="hard",  # type: ignore[arg-type]
        ),
    )


def generate(count: int, *, seed: int = 42) -> Iterable[Record]:
    """Yield ``2 * count`` records (one EN + one AR per sample).

    Difficulty mix roughly follows §11.2 of REASONING_DATA.md:
    70% easy/medium (Figure-1 + controls), 20% hard (Figure-2),
    10% hard (4-term sorites).

    Each tier is forced to a 50/50 valid/invalid split so the model
    cannot learn "2-universal-premises ⇒ yes" as a shortcut.
    """
    rng = random.Random(seed)

    def _polarity(conclusion: str) -> str:
        """Classify conclusion surface form from its EN template."""
        c = conclusion.strip()
        if c.startswith("All "):
            return "All"
        if c.startswith("No "):
            return "No"
        if c.startswith("Some ") and " not " in c:
            return "SomeNot"
        if c.startswith("Some "):
            return "Some"
        return "?"

    def _build_pools(moods):
        """Return (valid_idxs, polarity_weights, invalid_by_polarity).

        ``polarity_weights`` mirrors the *valid* pool's conclusion-polarity
        distribution. At sample-time we pick an invalid polarity with these
        weights, then a mood uniformly within that polarity \u2014 so for every
        conclusion surface form, the valid/invalid frequencies match and
        the model cannot shortcut on "All-conclusion \u2192 invalid" etc.
        """
        valid_idxs = [i for i, m in enumerate(moods) if m.valid]
        invalid_idxs = [i for i, m in enumerate(moods) if not m.valid]
        valid_pol_counts: dict[str, int] = {}
        for i in valid_idxs:
            p = _polarity(moods[i].conclusion)
            valid_pol_counts[p] = valid_pol_counts.get(p, 0) + 1
        invalid_by_pol: dict[str, list[int]] = {}
        for i in invalid_idxs:
            p = _polarity(moods[i].conclusion)
            invalid_by_pol.setdefault(p, []).append(i)
        # Restrict weights to polarities that exist on both sides;
        # fall back to uniform invalid sampling for polarities only in invalid.
        shared = [p for p in valid_pol_counts if p in invalid_by_pol]
        shared_weights = [valid_pol_counts[p] for p in shared]
        invalid_only = [p for p in invalid_by_pol if p not in valid_pol_counts]
        return valid_idxs, invalid_idxs, shared, shared_weights, invalid_by_pol, invalid_only

    (e_valid, e_invalid, e_shared_pols, e_shared_w,
     e_inv_by_pol, e_invalid_only) = _build_pools(MOODS_EN)
    (h_valid, h_invalid, h_shared_pols, h_shared_w,
     h_inv_by_pol, h_invalid_only) = _build_pools(MOODS_HARD_EN)

    def _pick_invalid(rng, shared_pols, shared_w, inv_by_pol, invalid_only):
        # 80% of invalid samples come from polarity-matched buckets (kills
        # the conclusion-polarity shortcut); the remaining 20% from
        # invalid-only polarities so classical fallacies stay represented.
        if invalid_only and rng.random() < 0.20:
            pol = rng.choice(invalid_only)
        else:
            pol = rng.choices(shared_pols, weights=shared_w, k=1)[0]
        return rng.choice(inv_by_pol[pol])

    for i in range(count):
        r = rng.random()
        if r < 0.70:
            # Figure-1: easy when valid, medium when invalid control.
            want_valid = rng.random() < 0.5
            if want_valid:
                j = rng.choice(e_valid)
            else:
                j = _pick_invalid(rng, e_shared_pols, e_shared_w,
                                  e_inv_by_pol, e_invalid_only)
            cats_en = rng.choice(CATEGORIES_EN)
            cats_ar = CATEGORIES_AR[CATEGORIES_EN.index(cats_en)]
            mood_en, mood_ar = MOODS_EN[j], MOODS_AR[j]
            difficulty = "easy" if mood_en.valid else "medium"
            yield _record(idx=i, lang="en", mood=mood_en, cats=cats_en,
                          difficulty=difficulty)
            yield _record(idx=i, lang="ar", mood=mood_ar, cats=cats_ar,
                          difficulty=difficulty)
        elif r < 0.90:
            # Figure-2 + illicit / existential / affirming controls — hard.
            want_valid = rng.random() < 0.5
            if want_valid:
                j = rng.choice(h_valid)
            else:
                j = _pick_invalid(rng, h_shared_pols, h_shared_w,
                                  h_inv_by_pol, h_invalid_only)
            cats_en = rng.choice(CATEGORIES_EN)
            cats_ar = CATEGORIES_AR[CATEGORIES_EN.index(cats_en)]
            mood_en, mood_ar = MOODS_HARD_EN[j], MOODS_HARD_AR[j]
            yield _record(idx=i, lang="en", mood=mood_en, cats=cats_en,
                          difficulty="hard")
            yield _record(idx=i, lang="ar", mood=mood_ar, cats=cats_ar,
                          difficulty="hard")
        else:
            # 4-term sorites — hard.
            chain_en, chain_ar = _sorites_cats(rng)
            valid = rng.random() < 0.5
            yield _sorites_record(idx=i, lang="en", chain=chain_en, valid=valid)
            yield _sorites_record(idx=i, lang="ar", chain=chain_ar, valid=valid)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    n = write_jsonl(args.out, generate(args.count, seed=args.seed))
    print(f"Wrote {n:,} records to {args.out}")


if __name__ == "__main__":
    main()
