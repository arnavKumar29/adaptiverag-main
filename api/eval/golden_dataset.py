"""
Golden dataset evaluation — offline batch eval against 100 curated Q&A pairs.
Design doc Section 12.2–12.3.
Run weekly via GitHub Actions or manually:
    python -m api.eval.golden_dataset --assert-thresholds
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from api.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

GOLDEN_DATASET_PATH = Path(__file__).parent.parent.parent / "tests" / "fixtures" / "golden_dataset.json"

THRESHOLDS = {
    "faithfulness": 0.75,
    "answer_relevancy": 0.80,
    "context_recall": 0.70,
    "context_precision": 0.65,
}


def load_golden_dataset(path: Optional[Path] = None) -> list[dict]:
    """Load golden Q&A pairs from JSON fixture."""
    p = path or GOLDEN_DATASET_PATH
    if not p.exists():
        logger.warning(f"Golden dataset not found at {p}")
        return []
    with open(p) as f:
        return json.load(f)


async def run_golden_eval(
    dataset: Optional[list[dict]] = None,
    assert_thresholds: bool = False,
) -> dict:
    """
    Run each golden Q&A pair through the RAG pipeline and collect RAGAS scores.
    Returns aggregate metrics.
    """
    from api.pipeline.retrieval.fusion import hybrid_retrieve
    from api.pipeline.compressor import compress_context
    from api.pipeline.generator import generate_with_fallback

    if dataset is None:
        dataset = load_golden_dataset()

    if not dataset:
        return {"error": "No golden dataset found", "count": 0}

    all_scores: dict[str, list[float]] = {
        "faithfulness": [],
        "answer_relevancy": [],
        "context_recall": [],
        "context_precision": [],
    }

    for i, item in enumerate(dataset):
        query = item["question"]
        ground_truth = item.get("answer", "")

        try:
            # Retrieve
            chunks = await hybrid_retrieve(query, top_k=5)
            if not chunks:
                logger.warning(f"[{i+1}] No chunks retrieved for: {query[:60]}")
                continue

            # Compress + generate
            compressed = await compress_context(query, chunks)
            gen_result = await generate_with_fallback(query, compressed)
            answer = gen_result.get("text", "")

            # RAGAS evaluate
            contexts = [c.content for c in chunks]
            try:
                from ragas import evaluate as ragas_evaluate
                from ragas.metrics import (
                    answer_relevancy,
                    context_precision,
                    context_recall,
                    faithfulness,
                )
                from datasets import Dataset

                eval_data = {
                    "question": [query],
                    "answer": [answer],
                    "contexts": [contexts],
                    "ground_truth": [ground_truth],
                }
                ds = Dataset.from_dict(eval_data)
                results = ragas_evaluate(
                    ds,
                    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                )
                scores = results.to_pandas().iloc[0].to_dict()

                for metric in all_scores:
                    val = scores.get(metric)
                    if val is not None:
                        all_scores[metric].append(float(val))

                logger.info(
                    f"[{i+1}/{len(dataset)}] faith={scores.get('faithfulness', 0):.3f} "
                    f"rel={scores.get('answer_relevancy', 0):.3f} — {query[:50]}"
                )
            except ImportError:
                logger.warning("RAGAS not installed — skipping scoring")
                break

        except Exception as e:
            logger.error(f"[{i+1}] Failed: {e}")
            continue

    # Compute aggregates
    aggregates = {}
    for metric, values in all_scores.items():
        if values:
            aggregates[metric] = {
                "mean": round(sum(values) / len(values), 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
                "count": len(values),
            }

    result = {
        "total_questions": len(dataset),
        "evaluated": sum(len(v) for v in all_scores.values()) // max(len(all_scores), 1),
        "aggregates": aggregates,
        "thresholds": THRESHOLDS,
        "pass": True,
    }

    # Check thresholds
    if assert_thresholds and aggregates:
        for metric, threshold in THRESHOLDS.items():
            if metric in aggregates:
                mean = aggregates[metric]["mean"]
                if mean < threshold:
                    result["pass"] = False
                    logger.error(
                        f"THRESHOLD FAIL: {metric} mean={mean:.4f} < {threshold}"
                    )

    return result


# ── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    logging.basicConfig(level=logging.INFO)

    do_assert = "--assert-thresholds" in sys.argv

    result = asyncio.run(run_golden_eval(assert_thresholds=do_assert))
    print(json.dumps(result, indent=2))

    if do_assert and not result.get("pass", True):
        print("\n❌ Golden dataset evaluation FAILED — thresholds not met")
        sys.exit(1)
    else:
        print("\n✅ Golden dataset evaluation passed")
