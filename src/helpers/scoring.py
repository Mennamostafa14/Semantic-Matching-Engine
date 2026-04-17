# routes/scoring.py
"""
Pure scoring helpers for the /index/compare endpoint.
No FastAPI, no I/O, fully unit-testable.

Pipeline
--------
1.  extract_keywords(text)          – lightweight TF-style keyword extraction
2.  keyword_overlap(qs, ps)         – Jaccard overlap between two keyword sets
3.  SECTION_WEIGHTS                 – importance multiplier per section
4.  score_block(scores)             – avg / max / blended score dict
5.  build_proposals(raw_hits, ...)  – full grouping + re-ranking pipeline
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import TypedDict

# ---------------------------------------------------------------------------
# 1.  Section importance weights
#     Used both during re-ranking and in the response payload so the caller
#     can explain why a section contributed more.
# ---------------------------------------------------------------------------
SECTION_WEIGHTS: dict[str, float] = {
    "objectives":          1.00,
    "problem_definition":  0.90,
    "solution_approach":   0.85,
    "background_scope":    0.60,
    "general":             0.40,
    "unknown":             0.35,
}
_DEFAULT_SECTION_WEIGHT = 0.35


# ---------------------------------------------------------------------------
# 2.  Keyword extraction
#     No external NLP library required.  Uses a simple TF-IDF-inspired
#     approach:  tokenise → remove stopwords → score by term frequency
#     weighted by inverse document-frequency approximation (log of inverse
#     relative frequency against a small stopword-biased denominator).
#     Returns the top-N tokens as a sorted list.
# ---------------------------------------------------------------------------
_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "this", "that",
    "these", "those", "it", "its", "as", "if", "so", "we", "our", "you",
    "your", "they", "their", "he", "she", "his", "her", "not", "no",
    "also", "which", "who", "what", "how", "when", "where", "while",
    "through", "between", "about", "into", "than", "then", "there",
    "each", "all", "any", "some", "such", "more", "most", "other",
    "after", "before", "use", "used", "using", "provide", "includes",
})


def extract_keywords(text: str, top_n: int = 20) -> list[str]:
    """
    Return up to `top_n` meaningful tokens from `text`, ranked by
    frequency × length (longer domain-specific terms score higher).

    Design decisions
    ----------------
    - Lowercase, alpha-only tokens only (strips numbers/punctuation).
    - Min length 4 chars to remove noise like "obj", "fig".
    - No external library — works anywhere Python runs.
    """
    tokens = re.findall(r"[a-zA-Z]{4,}", text.lower())
    freq: dict[str, int] = defaultdict(int)
    for t in tokens:
        if t not in _STOPWORDS:
            freq[t] += 1

    if not freq:
        return []

    # Score = frequency × log(length)  so "blockchain" beats "data"
    scored = sorted(freq.items(), key=lambda kv: kv[1] * math.log(len(kv[0])), reverse=True)
    return [word for word, _ in scored[:top_n]]


def keyword_overlap(query_kws: list[str], proposal_kws: list[str]) -> tuple[float, list[str]]:
    """
    Jaccard similarity between two keyword lists.

    Returns
    -------
    (jaccard_score: float,  common_keywords: list[str])
    """
    qs = set(query_kws)
    ps = set(proposal_kws)
    if not qs or not ps:
        return 0.0, []
    common = qs & ps
    jaccard = len(common) / len(qs | ps)
    return round(jaccard, 4), sorted(common)


# ---------------------------------------------------------------------------
# 3.  Score block helper
# ---------------------------------------------------------------------------

def score_block(scores: list[float]) -> dict:
    """Compute avg / max / blended score from a list of raw cosine scores."""
    avg   = sum(scores) / len(scores)
    mx    = max(scores)
    final = 0.7 * mx + 0.3 * avg
    return {
        "avg_similarity":  round(avg   * 100, 2),
        "max_similarity":  round(mx    * 100, 2),
        "final_score":     round(final * 100, 2),
        "matched_chunks":  len(scores),
    }


# ---------------------------------------------------------------------------
# 4.  Hit type
# ---------------------------------------------------------------------------

class Hit(TypedDict):
    project_id: str
    section:    str
    text:       str
    score:      float


# ---------------------------------------------------------------------------
# 5.  Main pipeline: build_proposals
# ---------------------------------------------------------------------------

def build_proposals(
    raw_hits:        list[Hit],
    query_keywords:  list[str],
    min_chunks:      int = 2,
    limit:           int = 5,
    kw_weight:       float = 0.15,   # how much keyword overlap contributes
    section_weight:  float = 0.10,   # how much section importance contributes
    vector_weight:   float = 0.75,   # must sum to 1.0 with the above two
) -> list[dict]:
    """
    Full grouping + re-ranking pipeline.

    Steps
    -----
    1.  Group raw_hits  →  proposal_id  →  section  →  [scores]
    2.  Filter proposals with fewer than `min_chunks` total hits (noise).
    3.  Compute section-level score blocks.
    4.  Compute proposal-level score from section-weighted average.
    5.  Extract proposal keywords from matched texts.
    6.  Compute keyword overlap with query.
    7.  Re-rank using:  final = vector × kw_overlap × section_boost
    8.  Sort, cap at `limit`, return.

    Parameters
    ----------
    raw_hits        : list of Hit dicts (project_id, section, text, score)
    query_keywords  : keywords extracted from the uploaded query document
    min_chunks      : discard proposals matched by fewer chunks than this
    limit           : max proposals to return
    kw_weight       : weight of keyword overlap in final score
    section_weight  : weight of section importance bonus in final score
    vector_weight   : weight of pure vector similarity in final score

    Note: kw_weight + section_weight + vector_weight should equal 1.0
    """

    # ── group: proposal → section → scores / texts ──────────────────────────
    # { pid: { section: { "scores": [...], "texts": [...] } } }
    grouped: dict[str, dict[str, dict]] = defaultdict(
        lambda: defaultdict(lambda: {"scores": [], "texts": []})
    )

    for hit in raw_hits:
        pid     = hit["project_id"]
        section = hit["section"]
        grouped[pid][section]["scores"].append(hit["score"])
        grouped[pid][section]["texts"].append(hit["text"])

    proposals = []

    for pid, sections in grouped.items():

        # ── noise filter ────────────────────────────────────────────────────
        total_chunks = sum(len(v["scores"]) for v in sections.values())
        if total_chunks < min_chunks:
            continue

        # ── section-level blocks ─────────────────────────────────────────────
        sections_out: dict[str, dict] = {}
        section_final_scores: list[float] = []
        section_importance_sum = 0.0
        all_proposal_texts: list[str] = []

        for sec, data in sections.items():
            block = score_block(data["scores"])
            sw    = SECTION_WEIGHTS.get(sec, _DEFAULT_SECTION_WEIGHT)
            block["section_weight"] = sw
            sections_out[sec] = block

            # weighted contribution to proposal-level score
            section_final_scores.append(block["final_score"] * sw)
            section_importance_sum += sw
            all_proposal_texts.extend(data["texts"])

        # ── proposal-level vector score (section-weighted mean) ──────────────
        if section_importance_sum > 0:
            vector_score = sum(section_final_scores) / section_importance_sum
        else:
            vector_score = 0.0

        # ── keyword analysis ─────────────────────────────────────────────────
        proposal_text     = " ".join(all_proposal_texts)
        proposal_keywords = extract_keywords(proposal_text, top_n=20)
        kw_jaccard, common_kws = keyword_overlap(query_keywords, proposal_keywords)

        # ── section importance bonus ─────────────────────────────────────────
        # Reward proposals that matched in high-weight sections.
        # Normalise to [0, 1] by dividing by the max possible weight.
        max_possible_weight = max(SECTION_WEIGHTS.values())   # 1.0 for "objectives"
        importance_bonus = (
            section_importance_sum / (len(sections) * max_possible_weight)
        ) if sections else 0.0

        # ── final blended score ──────────────────────────────────────────────
        # Combine all three signals.
        final_score = (
            vector_weight   * vector_score
            + kw_weight     * (kw_jaccard * 100)
            + section_weight * (importance_bonus * 100)
        )

        # ── top passages (proposal level only) ──────────────────────────────
        all_hits_flat = [
            (hit["score"], hit["text"])
            for hit in raw_hits
            if hit["project_id"] == pid
        ]
        top_passages = sorted(all_hits_flat, key=lambda x: x[0], reverse=True)[:3]

        proposals.append({
            "project_id":    pid,
            "overall_score": round(final_score, 2),
            "vector_score":  round(vector_score, 2),
            "matched_chunks": total_chunks,
            "sections":      sections_out,
            "keywords": {
                "query":    query_keywords,
                "proposal": proposal_keywords,
                "overlap":  common_kws,
                "overlap_score": round(kw_jaccard * 100, 2),
            },
            "top_passages": [
                {"score": round(s * 100, 2), "text": t}
                for s, t in top_passages
            ],
        })

    # ── sort by final blended score ──────────────────────────────────────────
    proposals.sort(key=lambda x: x["overall_score"], reverse=True)
    return proposals[:limit]