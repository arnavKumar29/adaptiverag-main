"""
Feedback learning loop — nightly strategy weight updates.
Design doc Section 13.3.
Runs as a GitHub Actions cron job or standalone script.
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import redis.asyncio as aioredis
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.config import get_settings
from api.db.postgres import AsyncSessionLocal, RouterWeight

logger = logging.getLogger(__name__)
settings = get_settings()


async def update_router_weights(hours: int = 24) -> dict:
    """
    Query last N hours of logs, compute per-class per-strategy mean
    faithfulness, and upsert router_weights table.
    Invalidates Redis cache.
    """
    async with AsyncSessionLocal() as db:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

        # Join query_logs with ragas_scores
        query = text("""
            SELECT
                ql.query_class,
                ql.strategy_used,
                AVG(rs.faithfulness) as mean_faith,
                COUNT(*) as sample_count
            FROM query_logs ql
            JOIN ragas_scores rs ON rs.query_log_id = ql.id
            WHERE ql.created_at > :cutoff
              AND ql.query_class IS NOT NULL
              AND ql.strategy_used IS NOT NULL
              AND rs.faithfulness IS NOT NULL
            GROUP BY ql.query_class, ql.strategy_used
        """)

        result = await db.execute(query, {"cutoff": cutoff})
        rows = result.fetchall()

        if not rows:
            logger.info("No recent query logs with RAGAS scores — skipping weight update")
            return {"updated": 0, "message": "No data in window"}

        updated = 0
        for row in rows:
            query_class = row.query_class
            strategy = row.strategy_used
            mean_faith = float(row.mean_faith)
            count = int(row.sample_count)

            # Upsert router weight
            existing = await db.execute(
                select(RouterWeight).where(
                    RouterWeight.query_class == query_class,
                    RouterWeight.strategy == strategy,
                )
            )
            weight_row = existing.scalar_one_or_none()

            if weight_row is None:
                weight_row = RouterWeight(
                    query_class=query_class,
                    strategy=strategy,
                    weight=mean_faith,
                    sample_count=count,
                )
                db.add(weight_row)
            else:
                weight_row.weight = mean_faith
                weight_row.sample_count = (weight_row.sample_count or 0) + count

            # Update Thompson sampling parameters
            successes = int(count * mean_faith)
            failures = count - successes
            weight_row.alpha = (weight_row.alpha or 1.0) + successes
            weight_row.beta = (weight_row.beta or 1.0) + failures

            updated += 1

        await db.commit()

    # Invalidate Redis cache
    try:
        redis = aioredis.from_url(settings.redis_url, decode_responses=True)
        keys = [k async for k in redis.scan_iter("router_weights:*")]
        if keys:
            await redis.delete(*keys)
            logger.info(f"Invalidated {len(keys)} Redis router weight cache entries")
        await redis.aclose()
    except Exception as e:
        logger.warning(f"Redis cache invalidation failed: {e}")

    logger.info(f"Updated {updated} router weight entries from last {hours}h of data")
    return {"updated": updated, "hours": hours}


# ── CLI entry point (for cron / GitHub Actions) ──────────────────────────────
if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    logging.basicConfig(level=logging.INFO)

    result = asyncio.run(update_router_weights(hours=24))
    print(f"Weight update result: {result}")
