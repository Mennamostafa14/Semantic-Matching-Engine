# helpers/proposal_analysis.py
from __future__ import annotations

import json
import logging
import re
import textwrap

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt builder (unchanged)
# ---------------------------------------------------------------------------

def _build_prompt(context_a: str, context_b: str) -> str:
    return f"""
You are an expert system that explains semantic similarity between two project proposals.

Your task is NOT to summarize.
Your task is to ANALYZE WHY they are similar.

Focus on:
1. Domain similarity (are they in the same field?)
2. Problem similarity (are they solving similar problems?)
3. Objective similarity (do they aim for the same outcomes?)

Use the provided structured signals:
- similarity scores
- section weights
- keyword overlap
- top matched passages

Be precise and analytical.

Return STRICTLY in this format:

Similarity percentage: <number>%

Explanation:
<clear, structured explanation in 4-6 lines>

Key Similarities:
- ...
- ...

Key Differences:
- ...
- ...

--- Proposal A ---
{context_a}

--- Proposal B ---
{context_b}
"""

# ----------------------------------------------------------------------------
def build_llm_context(proposal: dict) -> str:
    sections = proposal.get("sections", {})

    section_summary = "\n".join([
        f"- {name}: {data.get('final_score', 0):.1f}%"
        for name, data in sections.items()
    ])

    keywords = ", ".join(proposal.get("keywords", {}).get("overlap", [])[:10])

    top_passage = ""
    if proposal.get("top_passages"):
        best = max(proposal["top_passages"], key=lambda x: x.get("score", 0))
        top_passage = best.get("text", "")[:250]

    return f"""
Overall similarity: {proposal.get('overall_score')}%

Sections:
{section_summary}

Keywords:
{keywords}

Top evidence:
{top_passage}
"""

def build_llm_context(proposal: dict) -> str:
    sections = proposal.get("sections", {})

    section_summary = "\n".join([
        f"- {name}: {data.get('final_score', 0)}% (weight={data.get('section_weight')})"
        for name, data in sections.items()
    ])

    keywords = ", ".join(proposal.get("keywords", {}).get("overlap", [])[:10])

    top_passages = "\n".join([
        f"- ({p['score']}%) {p['text'][:150]}"
        for p in proposal.get("top_passages", [])[:3]
    ])

    return f"""
Overall similarity: {proposal.get('overall_score')}%

Section similarity:
{section_summary}

Common keywords:
{keywords}

Top matching passages:
{top_passages}
"""

# ----------------------------------------------------------------------------
# helpers/proposal_analysis.py

def safe_parse_llm_output(raw):
    # Guard: LLM returned nothing
    if not raw or not isinstance(raw, str):
        return {"error": "LLM returned no output"}

    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {"raw_text": raw}  # return whatever we got
# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def generate_similarity_analysis(query_proposal, db_proposal, generation_client):
    context_a = build_llm_context(query_proposal)
    context_b = build_llm_context(db_proposal)

    prompt = _build_prompt(context_a, context_b)

    raw = generation_client.generate_text(prompt)
    parsed = safe_parse_llm_output(raw)
    if not parsed:
        return {
            "proposal_id": db_proposal.get("project_id"),
            "reason": "LLM unavailable"
        }

    return {
        "proposal_id": db_proposal.get("project_id"),
        "analysis": parsed
    }