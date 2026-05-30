# helpers/retrieval_utils
import numpy as np
import logging

logger = logging.getLogger('uvicorn.error')

# ── helpers (unchanged) ──────────────────────────────────────────────────

def _mean_pool(vecs: list[list[float]]) -> list[float]:
        arr  = np.array(vecs, dtype=np.float32)
        mean = arr.mean(axis=0)
        norm = np.linalg.norm(mean)
        return (mean / norm).tolist() if norm > 0 else mean.tolist()

def _parse_hit(r) -> dict | None:
        meta = r.metadata or {}
        pid  = meta.get("proposal_id")
        if not pid:
            logger.warning(f"Hit with missing proposal_id skipped: {r.text[:60]!r}")
            return None
        return {
            "project_id": pid,
            "section":    (meta.get("section") or "unknown").strip(),
            "text":       r.text,
            "score":      r.score,
        }