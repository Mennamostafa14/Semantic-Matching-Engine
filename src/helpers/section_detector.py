# helpers/section_detector.py
from __future__ import annotations

import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Section registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SectionSpec:
    name: str
    weight: float
    patterns: tuple[str, ...]   # regex patterns (case-insensitive)


SECTION_REGISTRY: list[SectionSpec] = [
    SectionSpec(
        name="objectives",
        weight=1.0,
        patterns=(
            r"\bobjective[s]?\b",
            r"\bgoal[s]?\b",
            r"\baim[s]?\b",
            r"\bpurpose\b",
            r"\boutcome[s]?\b",
            r"\bexpected result[s]?\b",
        ),
    ),
    SectionSpec(
        name="problem_definition",
        weight=0.9,
        patterns=(
            r"\bproblem\b",
            r"\bchallenge[s]?\b",
            r"\bissue[s]?\b",
            r"\bpain point[s]?\b",
            r"\bgap[s]?\b",
            r"\blimitation[s]?\b",
            r"\bdifficult\w*\b",
        ),
    ),
    SectionSpec(
        name="solution_approach",
        weight=0.85,
        patterns=(
            r"\bsolution\b",
            r"\bapproach\b",
            r"\bmethodolog\w+\b",
            r"\bframework\b",
            r"\barchitecture\b",
            r"\bdesign\b",
            r"\bproposed\b",
            r"\bimplementation\b",
            r"\bstrateg\w+\b",
        ),
    ),
    SectionSpec(
        name="background_scope",
        weight=0.6,
        patterns=(
            r"\bbackground\b",
            r"\bscope\b",
            r"\bcontext\b",
            r"\bintroduction\b",
            r"\boverview\b",
            r"\bliterature\b",
            r"\brelated work\b",
            r"\bprior work\b",
            r"\bstate of the art\b",
        ),
    ),
]

# Derived constants — computed once at import time
SECTION_WEIGHTS: dict[str, float] = {spec.name: spec.weight for spec in SECTION_REGISTRY}
SECTION_WEIGHTS["general"] = 0.4

DEFAULT_SECTION = "general"

# Pre-compiled patterns — avoids re-compiling on every call
_COMPILED_SECTIONS: list[tuple[SectionSpec, list[re.Pattern]]] = [
    (spec, [re.compile(p, re.IGNORECASE) for p in spec.patterns])
    for spec in SECTION_REGISTRY
]


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def detect_section(text: str) -> str:
    """
    Return the canonical section name for a line of text.

    Scores each section by counting pattern matches. The section with the
    highest positive score wins. Ties go to the first (highest-weight) entry
    in SECTION_REGISTRY. No match → DEFAULT_SECTION ("general").
    """
    best_section = DEFAULT_SECTION
    best_score = 0

    for spec, compiled_patterns in _COMPILED_SECTIONS:
        score = sum(1 for pat in compiled_patterns if pat.search(text))
        if score > best_score:
            best_score = score
            best_section = spec.name

    return best_section


def group_lines_by_section(text: str) -> dict[str, str]:
    """
    Walk lines sequentially; switch the active section bucket when a
    heading-like line is detected (match fires AND line is ≤ 120 chars).

    Returns dict: section_name → concatenated text block.
    """
    buckets: dict[str, list[str]] = {spec.name: [] for spec in SECTION_REGISTRY}
    buckets[DEFAULT_SECTION] = []

    current_section = DEFAULT_SECTION

    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            buckets[current_section].append(line)
            continue

        detected = detect_section(stripped)
        if detected != DEFAULT_SECTION and len(stripped) <= 120:
            current_section = detected

        buckets[current_section].append(line)

    return {
        section: "\n".join(lines).strip()
        for section, lines in buckets.items()
    }