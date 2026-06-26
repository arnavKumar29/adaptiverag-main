"""
Query log table HTMX partial — paginated query history.
Terminal brutalist text-based layout.
"""
from __future__ import annotations


def render_query_log_table(logs: list[dict], limit: int = 10) -> str:
    """Render the query log table."""

    if not logs:
        return """
        <div class="card">
            <div class="empty-state">
                <p>[NO QUERIES RECORDED]</p>
                <p style="opacity: 0.5;">WAITING FOR INPUT...</p>
            </div>
        </div>
        """

    rows = ""
    for log in logs[:limit]:
        query_text = log.get("query", "")[:90]
        strategy = log.get("strategy_used", "—")
        query_class = log.get("query_class", "—")
        latency = log.get("latency_ms", "—")
        cache_hit = log.get("cache_hit", False)
        model = log.get("model_used", "—")
        created = log.get("created_at", "—")

        strategy_cls = {
            "dense": "badge-blue",
            "sparse": "badge-purple",
            "hybrid": "badge-ok",
        }.get(strategy, "badge-warn")

        qclass_cls = {
            "conceptual": "badge-blue",
            "keyword": "badge-purple",
            "mixed": "badge-warn",
        }.get(query_class, "badge-warn")

        cache_badge = (
            '<span class="badge badge-ok">HIT</span>'
            if cache_hit
            else '<span class="badge" style="color: var(--color-label);">MISS</span>'
        )

        latency_color = "var(--color-green)" if isinstance(latency, (int, float)) and latency < 500 else "var(--color-yellow)"

        if created and created != "—":
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                created = dt.strftime("%H:%M:%S")
            except Exception:
                pass

        rows += f"""
        <tr>
            <td style="color: var(--color-green);">> {query_text}…</td>
            <td><span class="badge {strategy_cls}">{strategy}</span></td>
            <td><span class="badge {qclass_cls}">{query_class}</span></td>
            <td style="color: {latency_color};">{latency}ms</td>
            <td>{cache_badge}</td>
            <td style="color: var(--color-muted);">{model}</td>
            <td style="color: var(--color-label);">{created}</td>
        </tr>
        """

    showing = min(len(logs), limit)
    total = len(logs)

    return f"""
    <div class="card">
        <div class="card-header">
            <span class="card-title">[ QUERY.LOG ]</span>
            <span style="font-size: 10px; color: var(--color-muted);">SHOWING {showing}/{total}</span>
        </div>
        <div style="overflow-x: auto;">
            <table class="data-table">
                <thead>
                    <tr>
                        <th>[QUERY_STRING]</th>
                        <th>[STRAT]</th>
                        <th>[CLASS]</th>
                        <th>[LATENCY]</th>
                        <th>[CACHE]</th>
                        <th>[MODEL]</th>
                        <th>[TIME]</th>
                    </tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
    </div>
    """
