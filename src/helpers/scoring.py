# routes/scoring.py
"""
Pure scoring helpers for the /index/compare endpoint.
No FastAPI, no I/O, fully unit-testable.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import TypedDict

# ---------------------------------------------------------------------------
# 1.  Section importance weights
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
    # English + Arabic tokens
    tokens = re.findall(r"[a-zA-Z\u0600-\u06FF]{3,}", text.lower())
    freq: dict[str, int] = defaultdict(int)
    for t in tokens:
        if t not in _STOPWORDS:
            freq[t] += 1

    if not freq:
        return []

    scored = sorted(freq.items(), key=lambda kv: kv[1] * math.log(max(len(kv[0]), 2)), reverse=True)
    return [word for word, _ in scored[:top_n]]


def keyword_overlap(query_kws: list[str], proposal_kws: list[str]) -> tuple[float, list[str]]:
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
# 5.  Main pipeline: build_proposals  (chunk-vs-chunk aware)
# ---------------------------------------------------------------------------

def build_proposals(
    raw_hits:        list[Hit],
    query_keywords:  list[str],
    min_chunks:      int = 2,
    limit:           int = 5,
    kw_weight:       float = 0.15,
    section_weight:  float = 0.10,
    vector_weight:   float = 0.75,
) -> list[dict]:
    """
    Chunk-vs-chunk aware grouping + re-ranking pipeline.

    raw_hits now contains hits from MULTIPLE query-chunk searches,
    so the same (proposal_id, section) pair may appear many times.
    We deduplicate by keeping only the BEST score per stored chunk
    (identified by text) to avoid inflating scores through repetition.
    """

    # ── deduplicate: keep best score per (proposal_id, text) ────────────────
    best: dict[tuple[str, str], Hit] = {}
    for hit in raw_hits:
        pid  = hit.get("project_id")
        text = hit.get("text", "")
        if not pid:
            continue
        key = (pid, text)
        if key not in best or hit["score"] > best[key]["score"]:
            best[key] = hit

    deduped_hits = list(best.values())

    # ── group: proposal → section → scores / texts ──────────────────────────
    grouped: dict[str, dict[str, dict]] = defaultdict(
        lambda: defaultdict(lambda: {"scores": [], "texts": []})
    )

    for hit in deduped_hits:
        pid     = hit["project_id"]
        section = hit.get("section", "unknown")
        grouped[pid][section]["scores"].append(hit["score"])
        grouped[pid][section]["texts"].append(hit["text"])

    proposals = []

    for pid, sections in grouped.items():

        # ── noise filter ─────────────────────────────────────────────────────
        total_chunks = sum(len(v["scores"]) for v in sections.values())
        if total_chunks < min_chunks:
            continue

        # ── section-level blocks ──────────────────────────────────────────────
        sections_out: dict[str, dict] = {}
        section_final_scores: list[float] = []
        section_importance_sum = 0.0
        all_proposal_texts: list[str] = []

        for sec, data in sections.items():
            block = score_block(data["scores"])
            sw    = SECTION_WEIGHTS.get(sec, _DEFAULT_SECTION_WEIGHT)
            block["section_weight"] = sw
            sections_out[sec] = block

            section_final_scores.append(block["final_score"] * sw)
            section_importance_sum += sw
            all_proposal_texts.extend(data["texts"])

        # ── proposal-level vector score ───────────────────────────────────────
        if section_importance_sum > 0:
            vector_score = sum(section_final_scores) / section_importance_sum
        else:
            vector_score = 0.0

        # ── keyword analysis ──────────────────────────────────────────────────
        proposal_text     = " ".join(all_proposal_texts)
        proposal_keywords = extract_keywords(proposal_text, top_n=20)
        kw_jaccard, common_kws = keyword_overlap(query_keywords, proposal_keywords)

        # ── section importance bonus ──────────────────────────────────────────
        max_possible_weight = max(SECTION_WEIGHTS.values())
        importance_bonus = (
            section_importance_sum / (len(sections) * max_possible_weight)
        ) if sections else 0.0

        # ── final blended score ───────────────────────────────────────────────
        final_score = (
            vector_weight    * vector_score
            + kw_weight      * (kw_jaccard * 100)
            + section_weight * (importance_bonus * 100)
        )

        # ── top passages ──────────────────────────────────────────────────────
        all_hits_flat = [
            (hit["score"], hit["text"])
            for hit in deduped_hits
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
                "query":         query_keywords,
                "proposal":      proposal_keywords,
                "overlap":       common_kws,
                "overlap_score": round(kw_jaccard * 100, 2),
            },
            "top_passages": [
                {"score": round(s * 100, 2), "text": t}
                for s, t in top_passages
            ],
        })

    proposals.sort(key=lambda x: x["overall_score"], reverse=True)
    return proposals[:limit]