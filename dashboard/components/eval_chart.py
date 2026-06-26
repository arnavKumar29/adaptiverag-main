"""
Evaluation section HTMX partial — RAGAS score display.
Terminal brutalist layout.
"""
from __future__ import annotations


def _score_bar(label: str, value: float, target: float, color: str, bg: str) -> str:
    """Render a single RAGAS metric bar with label, value, and target marker."""
    pct = max(0, min(100, value * 100))
    target_pct = min(100, target * 100)
    is_above = value >= target
    bar_color = color if is_above else "var(--color-red)"
    status_badge = (
        '<span class="badge badge-ok">[OK]</span>'
        if is_above
        else '<span class="badge badge-error">[LO]</span>'
    )

    # Brutalist ASCII progress bar
    filled = int(pct / 10)
    empty = 10 - filled
    target_idx = int(target_pct / 10)
    
    # We'll just use a simple block progress bar with CSS
    return f"""
    <div style="margin-bottom: 22px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 11px; font-weight: 700; color: var(--color-text); text-transform: uppercase;">[{label}]</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                {status_badge}
                <span style="font-size: 14px; font-weight: 700; color: {bar_color}; min-width: 38px; text-align: right;">{value:.2f}</span>
            </div>
        </div>
        <div style="position: relative;">
            <div style="width: 100%; height: 8px; background: rgba(0,0,0,0.3); border: 1px solid var(--color-border);">
                <div style="height: 100%; width: {pct}%; background: {color};"></div>
            </div>
            <div style="position: absolute; left: calc({target_pct}% - 1px); top: -3px; width: 2px; height: 14px;
                         background: var(--color-text); border-radius: 0px;"
                 title="Target: {target:.2f}">
            </div>
        </div>
        <div style="display: flex; justify-content: flex-end; margin-top: 4px;">
            <span style="font-size: 10px; color: var(--color-label);">TARGET: {target:.2f}</span>
        </div>
    </div>
    """


def render_eval_section(scores: dict | None = None) -> str:
    """Render the evaluation metrics section."""

    if scores is None:
        # Placeholder view (no data yet)
        return f"""
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 24px;">

            <div class="card">
                <div class="card-header">
                    <span class="card-title">[ RAGAS METRICS ]</span>
                    <span class="badge badge-warn">NO_DATA</span>
                </div>
                <div class="card-body">
                    {_score_bar("Faithfulness",       0.0, 0.75, "var(--color-blue)",   "var(--color-blue-dim)")}
                    {_score_bar("Answer Relevancy",   0.0, 0.80, "var(--color-green)",  "var(--color-green-dim)")}
                    {_score_bar("Context Recall",     0.0, 0.70, "var(--color-purple)", "var(--color-purple-dim)")}
                    {_score_bar("Context Precision",  0.0, 0.65, "var(--color-yellow)", "var(--color-yellow-dim)")}
                    <div style="margin-top: 8px; padding: 12px; background: var(--color-card-alt); border: 1px solid var(--color-border);">
                        <p style="font-size: 10px; color: var(--color-muted); text-align: center; text-transform: uppercase;">
                            SCORES AUTO-POPULATE AFTER RAGAS EVAL RUN.
                        </p>
                    </div>
                </div>
            </div>

            <div class="card">
                <div class="card-header">
                    <span class="card-title">[ GOLDEN DATASET ]</span>
                    <span class="badge badge-purple">MANUAL</span>
                </div>
                <div class="card-body">
                    <div class="empty-state">
                        <p>[ EXECUTE GOLDEN SUITE ]</p>
                        <div style="padding: 12px; background: var(--color-bg); border: 1px solid var(--color-border); color: var(--color-green); font-size: 11px; margin-top: 12px; text-align: left;">
                            > python -m api.eval.golden_dataset --assert-thresholds<br>
                            <span class="loading"></span>
                        </div>
                        <p style="font-size: 10px; margin-top: 12px; opacity: 0.5;">100 Q&A · 4 RAGAS METRICS · CI/CD READY</p>
                    </div>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">
                <span class="card-title">[ FEEDBACK LOOP ]</span>
            </div>
            <div class="card-body">
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px;">
                    <div style="padding: 16px; border: 1px solid var(--color-border); background: var(--color-card-alt);">
                        <div style="font-size: 20px; font-weight: 700; color: #fff; margin-bottom: 4px;">0</div>
                        <div style="font-size: 11px; color: var(--color-muted); text-transform: uppercase;">THUMBS UP</div>
                    </div>
                    <div style="padding: 16px; border: 1px solid var(--color-border); background: var(--color-card-alt);">
                        <div style="font-size: 20px; font-weight: 700; color: var(--color-red); margin-bottom: 4px;">0</div>
                        <div style="font-size: 11px; color: var(--color-muted); text-transform: uppercase;">THUMBS DOWN</div>
                    </div>
                    <div style="padding: 16px; border: 1px solid var(--color-border); background: var(--color-card-alt);">
                        <div style="font-size: 20px; font-weight: 700; color: #fff; margin-bottom: 4px;">0%</div>
                        <div style="font-size: 11px; color: var(--color-muted); text-transform: uppercase;">APPROVAL RATE</div>
                    </div>
                </div>
            </div>
        </div>
        """

    # If we have real scores, we'd map them here
    return ""
