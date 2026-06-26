"""
RAGAS evaluation — runs asynchronously after each query response.
No latency impact on the user-facing response.
Design doc Section 12.
"""
from __future__ import annotations

import logging
from typing import Optional
from uuid import UUID

from api.core.config import get_settings
from api.core.telemetry import rag_ragas_faithfulness
from api.db.postgres import AsyncSessionLocal, QueryLog, RagasScore

logger = logging.getLogger(__name__)
settings = get_settings()

# Rolling faithfulness buffer (last 100 queries)
_faithfulness_buffer: list[float] = []
MAX_BUFFER = 100


async def evaluate_query(
    query_log_id: UUID,
    query: str,
    answer: str,
    contexts: list[str],
    ground_truth: Optional[str] = None,
) -> Optional[dict]:
    """
    Run RAGAS evaluation on a completed query.
    Writes scores to ragas_scores table.
    Updates Prometheus gauge.
    """
    if not contexts or not answer:
        return None

    try:
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from datasets import Dataset

        # Build evaluation dataset
        data = {
            "question": [query],
            "answer": [answer],
            "contexts": [contexts],
        }
        if ground_truth:
            data["ground_truth"] = [ground_truth]

        dataset = Dataset.from_dict(data)

        metrics = [faithfulness, answer_relevancy, context_precision]
        if ground_truth:
            metrics.append(context_recall)

        results = evaluate(dataset, metrics=metrics)
        scores = results.to_pandas().iloc[0].to_dict()

    except ImportError:
        logger.warning("RAGAS not installed, skipping evaluation")
        return None
    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        return None

    faith = float(scores.get("faithfulness", 0.0) or 0.0)
    relevancy = float(scores.get("answer_relevancy", 0.0) or 0.0)
    recall = float(scores.get("context_recall", 0.0) or 0.0)
    precision = float(scores.get("context_precision", 0.0) or 0.0)

    # Update Prometheus rolling mean
    _faithfulness_buffer.append(faith)
    if len(_faithfulness_buffer) > MAX_BUFFER:
        _faithfulness_buffer.pop(0)
    rag_ragas_faithfulness.set(sum(_faithfulness_buffer) / len(_faithfulness_buffer))

    # Write to Postgres
    async with AsyncSessionLocal() as db:
        score_record = RagasScore(
            query_log_id=query_log_id,
            faithfulness=faith,
            answer_relevancy=relevancy,
            context_recall=recall,
            context_precision=precision,
        )
        db.add(score_record)
        await db.commit()

    score_dict = {
        "faithfulness": faith,
        "answer_relevancy": relevancy,
        "context_recall": recall,
        "context_precision": precision,
    }

    logger.info(
        f"RAGAS scores for {query_log_id}: "
        f"faith={faith:.3f} rel={relevancy:.3f} recall={recall:.3f} prec={precision:.3f}"
    )

    # Check against target — log warning if below threshold
    if faith < settings.ragas_faithfulness_target:
        logger.warning(
            f"Faithfulness {faith:.3f} below target {settings.ragas_faithfulness_target}"
        )

    return score_dict
