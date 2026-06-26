"""
Router v2 ML classifier training script.
Reads query_logs + ragas_scores from PostgreSQL, extracts features,
trains a LogisticRegression model, and saves to models/router_clf.pkl.

Usage:
    python scripts/train_router_v2.py
    python scripts/train_router_v2.py --min-samples 200

Run weekly via GitHub Actions or manually after accumulating 500+ query logs.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import pickle
import re
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split

logger = logging.getLogger(__name__)


# ── Feature extraction (matches api/pipeline/router.py::_extract_features) ───

def extract_features(query: str) -> list[float]:
    """Extract 6 features from a query for the ML router."""
    words = query.split()

    question_words = {"what", "why", "how", "when", "which", "who", "where"}
    has_question_word = float(any(w.lower() in question_words for w in words[:3]))
    has_quoted = float(bool(re.search(r'"[^"]+"', query)))

    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(query[:500])
        entity_count = float(len(doc.ents))
        sentence_count = float(len(list(doc.sents)))
    except Exception:
        entity_count = 0.0
        sentence_count = 1.0

    return [
        float(len(words)),
        float(sum(len(w) for w in words) / max(len(words), 1)),
        has_question_word,
        has_quoted,
        entity_count,
        sentence_count,
    ]


# ── Data loading ──────────────────────────────────────────────────────────────

async def load_training_data(min_faithfulness: float = 0.5) -> tuple[list, list]:
    """
    Load query logs with RAGAS scores from PostgreSQL.
    Returns (features, labels) where labels are the best strategy per query.
    """
    from sqlalchemy import text
    from api.db.postgres import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        query = text("""
            SELECT
                ql.query,
                ql.strategy_used,
                rs.faithfulness,
                rs.answer_relevancy
            FROM query_logs ql
            JOIN ragas_scores rs ON rs.query_log_id = ql.id
            WHERE ql.strategy_used IS NOT NULL
              AND rs.faithfulness IS NOT NULL
              AND rs.faithfulness >= :min_faith
            ORDER BY ql.created_at DESC
            LIMIT 10000
        """)

        result = await db.execute(query, {"min_faith": min_faithfulness})
        rows = result.fetchall()

    features = []
    labels = []

    for row in rows:
        feats = extract_features(row.query)
        features.append(feats)
        labels.append(row.strategy_used)

    return features, labels


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(
    features: list[list[float]],
    labels: list[str],
    output_path: Path,
) -> dict:
    """Train and save the LogisticRegression model."""
    X = np.array(features)
    y = np.array(labels)

    # Verify minimum class representation
    unique, counts = np.unique(y, return_counts=True)
    class_dist = dict(zip(unique, counts))
    logger.info(f"Class distribution: {class_dist}")

    if len(unique) < 2:
        logger.warning("Need at least 2 classes to train. Skipping.")
        return {"status": "skipped", "reason": "insufficient classes", "classes": class_dist}

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train LogisticRegression
    clf = LogisticRegression(
        max_iter=1000,
        multi_class="multinomial",
        class_weight="balanced",  # handle class imbalance
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Cross-validation
    cv_scores = cross_val_score(clf, X, y, cv=min(5, len(unique)), scoring="accuracy")

    logger.info(f"Test accuracy: {accuracy:.4f}")
    logger.info(f"Cross-val accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    logger.info(f"\n{classification_report(y_test, y_pred)}")

    # Save model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(clf, f)
    logger.info(f"Model saved to {output_path}")

    return {
        "status": "trained",
        "accuracy": round(accuracy, 4),
        "cv_accuracy_mean": round(cv_scores.mean(), 4),
        "cv_accuracy_std": round(cv_scores.std(), 4),
        "samples": len(features),
        "classes": class_dist,
        "classification_report": report,
        "model_path": str(output_path),
    }


# ── CLI entry point ──────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    """Main training pipeline."""
    logger.info("Loading training data from PostgreSQL...")
    features, labels = await load_training_data(
        min_faithfulness=args.min_faithfulness,
    )

    if len(features) < args.min_samples:
        logger.warning(
            f"Only {len(features)} samples available "
            f"(minimum {args.min_samples} required). Skipping training."
        )
        print(f"\n⚠️  Insufficient data: {len(features)}/{args.min_samples} samples")
        print("    Accumulate more query logs with RAGAS scores before training.")
        return

    logger.info(f"Loaded {len(features)} training samples")

    output_path = Path(args.output)
    result = train_model(features, labels, output_path)

    print(f"\n{'='*60}")
    print(f"Router v2 ML Training Results")
    print(f"{'='*60}")
    print(f"Status:       {result['status']}")
    print(f"Samples:      {result.get('samples', 0)}")
    print(f"Accuracy:     {result.get('accuracy', 'N/A')}")
    print(f"CV Accuracy:  {result.get('cv_accuracy_mean', 'N/A')} ± {result.get('cv_accuracy_std', 'N/A')}")
    print(f"Classes:      {result.get('classes', {})}")
    print(f"Model saved:  {result.get('model_path', 'N/A')}")
    print(f"{'='*60}")

    if result["status"] == "trained" and result.get("accuracy", 0) < 0.6:
        print("\n⚠️  Warning: Accuracy is below 0.6 — model may not be reliable.")
        print("    Consider accumulating more diverse query data.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Router v2 ML classifier")
    parser.add_argument(
        "--min-samples", type=int, default=500,
        help="Minimum samples required to train (default: 500)",
    )
    parser.add_argument(
        "--min-faithfulness", type=float, default=0.5,
        help="Minimum faithfulness score for training data (default: 0.5)",
    )
    parser.add_argument(
        "--output", type=str, default="models/router_clf.pkl",
        help="Output model path (default: models/router_clf.pkl)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    asyncio.run(main(args))
