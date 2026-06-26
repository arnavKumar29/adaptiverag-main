"""
Stats cards HTMX partial — overview metrics at a glance.
Terminal brutalist text-based layout.
"""
from __future__ import annotations


def render_stats_cards(health: dict | None) -> str:
    """Render the stats cards grid from health data."""

    if not health:
        return """
        <div class="stats-grid">
            <div class="stat-card" style="border-color: var(--color-red); grid-column: 1/-1;">
                <div class="stat-card-header">
                    <div class="stat-label">[ SYS.ERR ]</div>
                </div>
                <div class="stat-value" style="color: var(--color-red);">OFFLINE</div>
                <div class="stat-sub">Cannot reach API. Connection refused.</div>
            </div>
        </div>
        """

    services = health.get("services", [])
    ok_count = sum(1 for s in services if s.get("status") == "ok")
    total_count = len(services)
    status = health.get("status", "unknown")
    is_ok = status == "ok"

    latencies = [s.get("latency_ms", 0) for s in services if s.get("latency_ms")]
    avg_latency = round(sum(latencies) / len(latencies), 1) if latencies else 0
    version = health.get("version", "1.0.0")

    pct_ok = round(ok_count / max(total_count, 1) * 100)
    status_color = "var(--color-green)" if is_ok else "var(--color-yellow)"
    status_label = "HEALTHY" if is_ok else "DEGRADED"

    return f"""
    <div class="stats-grid">

        <div class="stat-card" style="border-color: {status_color};">
            <div class="stat-card-header">
                <div class="stat-label">[ STATUS ]</div>
            </div>
            <div class="stat-value" style="color: {status_color};">{status_label}</div>
            <div class="stat-sub">{ok_count}/{total_count} SVCS UP ({pct_ok}%)</div>
        </div>

        <div class="stat-card">
            <div class="stat-card-header">
                <div class="stat-label">[ SERVICES ]</div>
            </div>
            <div class="stat-value" style="color: var(--color-blue);">{ok_count}</div>
            <div class="stat-sub">OF {total_count} ONLINE</div>
        </div>

        <div class="stat-card">
            <div class="stat-card-header">
                <div class="stat-label">[ AVG.LATENCY ]</div>
            </div>
            <div class="stat-value" style="color: var(--color-purple);">{avg_latency}ms</div>
            <div class="stat-sub">ACROSS ALL SVCS</div>
        </div>

        <div class="stat-card">
            <div class="stat-card-header">
                <div class="stat-label">[ CORE.VER ]</div>
            </div>
            <div class="stat-value" style="color: var(--color-muted); font-size: 20px;">v{version}</div>
            <div class="stat-sub">BUILD ACTIVE</div>
        </div>

    </div>
    """
