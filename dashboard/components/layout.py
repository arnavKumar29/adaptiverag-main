"""
Base HTML layout shell for the admin dashboard.
Terminal-inspired brutalist dark theme, HTMX live updates.
"""
from __future__ import annotations


def nav_link(label: str, href: str, icon_svg: str, active: str, key: str) -> str:
    """Render a sidebar navigation link."""
    is_active = active == key
    active_cls = "bg-active" if is_active else ""
    prefix = ">" if is_active else " "
    return f"""
    <a href="{href}" class="nav-item {active_cls}">
        <span class="nav-prefix">{prefix}</span>
        <span class="nav-label">[{key.upper()}]</span>
    </a>"""


def page_shell(title: str, content: str, active: str = "overview") -> str:
    """Wrap content in the full HTML shell with nav, styles, and HTMX."""

    nav_items = "".join([
        nav_link("Overview", "/", ICON_OVERVIEW, active, "overview"),
        nav_link("Chat AI", "/chat", ICON_CHAT, active, "chat"),
        nav_link("Documents", "/documents", ICON_DOCS, active, "documents"),
        nav_link("Query Log", "/queries", ICON_QUERY, active, "queries"),
        nav_link("Evaluation", "/eval", ICON_EVAL, active, "eval"),
    ])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} — SYSTEM_DASHBOARD</title>
    <script src="https://unpkg.com/htmx.org@1.9.12"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        /* ── Reset & Base ──────────────────────────────────────────── */
        *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

        :root {{
            --sidebar-w: 240px;
            --topbar-h: 60px;

            /* Brutalist Terminal Palette */
            --color-bg:        #000000;
            --color-sidebar:   #050505;
            --color-card:      #0a0a0a;
            --color-card-alt:  #111111;
            --color-border:    #333333;
            --color-hover:     #1a1a1a;
            --color-active:    #222222;

            --color-text:      #e0e0e0;
            --color-muted:     #888888;
            --color-label:     #555555;

            --color-blue:      #00ffff;
            --color-blue-dim:  rgba(0,255,255,0.1);
            --color-green:     #00ff00;
            --color-green-dim: rgba(0,255,0,0.1);
            --color-yellow:    #ffff00;
            --color-yellow-dim:rgba(255,255,0,0.1);
            --color-red:       #ff0000;
            --color-red-dim:   rgba(255,0,0,0.1);
            --color-purple:    #ff00ff;
            --color-purple-dim:rgba(255,0,255,0.1);

            --radius-sm: 0px;
            --radius-md: 0px;
            --radius-lg: 0px;
            --shadow-card: none;
            --shadow-lg:   none;
            --transition:  none;
        }}

        html, body {{
            height: 100%;
            font-family: 'JetBrains Mono', monospace;
            background: var(--color-bg);
            color: var(--color-text);
            font-size: 13px;
            line-height: 1.5;
            -webkit-font-smoothing: none;
        }}

        /* ── Layout Shell ──────────────────────────────────────────── */
        .shell {{ display: flex; min-height: 100vh; }}

        /* ── Sidebar ───────────────────────────────────────────────── */
        .sidebar {{
            width: var(--sidebar-w);
            background: var(--color-sidebar);
            border-right: 1px solid var(--color-border);
            position: fixed;
            top: 0; left: 0;
            height: 100vh;
            display: flex;
            flex-direction: column;
            z-index: 100;
            overflow-y: auto;
        }}

        .sidebar-logo {{
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 20px;
            border-bottom: 1px solid var(--color-border);
        }}

        .logo-text {{ font-size: 16px; font-weight: 700; color: var(--color-green); letter-spacing: 1px; }}
        .logo-sub  {{ font-size: 10px; color: var(--color-muted); margin-top: 2px; text-transform: uppercase; }}

        .nav-section {{
            padding: 20px 20px 10px;
            font-size: 10px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 2px;
            color: var(--color-label);
        }}

        .nav-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 20px;
            text-decoration: none;
            font-size: 13px;
            color: var(--color-text);
        }}

        .nav-item:hover {{
            background: var(--color-hover);
        }}

        .bg-active {{
            background: var(--color-active);
            color: var(--color-green);
        }}
        
        .nav-prefix {{
            color: var(--color-green);
            font-weight: bold;
            width: 10px;
        }}

        .sidebar-footer {{
            margin-top: auto;
            padding: 16px 20px;
            border-top: 1px solid var(--color-border);
            font-size: 11px;
            color: var(--color-muted);
        }}

        /* ── Topbar ────────────────────────────────────────────────── */
        .topbar {{
            position: fixed;
            top: 0; left: var(--sidebar-w);
            right: 0;
            height: var(--topbar-h);
            background: var(--color-sidebar);
            border-bottom: 1px solid var(--color-border);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 32px;
            z-index: 90;
        }}

        .topbar-title {{
            font-size: 18px;
            font-weight: 700;
            color: #fff;
            text-transform: uppercase;
        }}

        .topbar-right {{
            display: flex;
            align-items: center;
            gap: 16px;
        }}

        .refresh-btn {{
            padding: 6px 12px;
            background: transparent;
            color: var(--color-blue);
            border: 1px solid var(--color-blue);
            font-size: 11px;
            text-transform: uppercase;
            cursor: pointer;
            text-decoration: none;
        }}

        .refresh-btn:hover {{
            background: var(--color-blue-dim);
        }}

        .live-badge {{
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 4px 8px;
            border: 1px solid var(--color-green);
            color: var(--color-green);
            font-size: 10px;
            text-transform: uppercase;
        }}

        /* ── Main Content ──────────────────────────────────────────── */
        .main-content {{
            margin-left: var(--sidebar-w);
            margin-top: var(--topbar-h);
            padding: 32px 36px;
            min-height: calc(100vh - var(--topbar-h));
            max-width: 1400px;
            width: 100%;
        }}

        /* ── Page Header ───────────────────────────────────────────── */
        .page-header {{
            margin-bottom: 28px;
            border-bottom: 1px solid var(--color-border);
            padding-bottom: 16px;
        }}

        .page-header h1 {{
            font-size: 22px;
            font-weight: 700;
            color: var(--color-green);
            text-transform: uppercase;
        }}

        .page-header p {{
            color: var(--color-muted);
            font-size: 12px;
            margin-top: 4px;
        }}

        /* ── Section ───────────────────────────────────────────────── */
        .section {{ margin-bottom: 32px; }}

        .section-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 12px;
            border-bottom: 1px dashed var(--color-border);
            padding-bottom: 4px;
        }}

        .section-title {{
            font-size: 14px;
            font-weight: 700;
            color: #fff;
            text-transform: uppercase;
        }}

        /* ── Stats Grid ────────────────────────────────────────────── */
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 32px;
        }}

        .stat-card {{
            background: var(--color-card);
            border: 1px solid var(--color-border);
            padding: 16px;
        }}

        .stat-card-header {{
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            margin-bottom: 12px;
        }}

        .stat-label {{
            font-size: 11px;
            color: var(--color-muted);
            text-transform: uppercase;
        }}

        .stat-value {{
            font-size: 24px;
            font-weight: 700;
            color: var(--color-green);
            line-height: 1;
            margin-bottom: 4px;
        }}

        .stat-sub {{
            font-size: 10px;
            color: var(--color-label);
        }}

        /* ── Card ──────────────────────────────────────────────────── */
        .card {{
            background: var(--color-card);
            border: 1px solid var(--color-border);
        }}

        .card-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 12px 16px;
            border-bottom: 1px solid var(--color-border);
            background: var(--color-card-alt);
        }}

        .card-title {{
            font-size: 13px;
            font-weight: 700;
            color: #fff;
            text-transform: uppercase;
        }}

        .card-body {{ padding: 16px; }}

        /* ── Table ─────────────────────────────────────────────────── */
        .data-table {{
            width: 100%;
            border-collapse: collapse;
        }}

        .data-table thead {{
            background: var(--color-card-alt);
        }}

        .data-table th {{
            padding: 10px 16px;
            font-size: 10px;
            font-weight: 700;
            text-transform: uppercase;
            color: var(--color-label);
            text-align: left;
            white-space: nowrap;
            border-bottom: 1px solid var(--color-border);
        }}

        .data-table td {{
            padding: 10px 16px;
            font-size: 12px;
            color: var(--color-muted);
            border-bottom: 1px solid var(--color-border);
            vertical-align: middle;
        }}

        .data-table tbody tr:hover td {{
            background: var(--color-hover);
            color: var(--color-text);
        }}

        /* ── Badges ────────────────────────────────────────────────── */
        .badge {{
            display: inline-block;
            padding: 2px 6px;
            font-size: 10px;
            font-weight: 700;
            text-transform: uppercase;
            border: 1px solid currentColor;
        }}

        .badge-ok     {{ color: var(--color-green); }}
        .badge-warn   {{ color: var(--color-yellow); }}
        .badge-error  {{ color: var(--color-red); }}
        .badge-blue   {{ color: var(--color-blue); }}
        .badge-purple {{ color: var(--color-purple); }}

        /* ── Loading / Empty / Error ───────────────────────────────── */
        .loading {{
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 40px;
            color: var(--color-green);
            font-size: 12px;
            text-transform: uppercase;
        }}

        .loading::after {{
            content: '█';
            animation: blink 1s step-end infinite;
            margin-left: 4px;
        }}

        @keyframes blink {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0; }}
        }}

        .empty-state {{
            padding: 40px 20px;
            color: var(--color-muted);
            text-align: center;
            font-size: 12px;
        }}

        .error-box {{
            border: 1px solid var(--color-red);
            padding: 12px;
            color: var(--color-red);
            font-size: 12px;
            margin: 16px 0;
            background: rgba(255,0,0,0.05);
        }}

        /* ── Forms and Inputs ──────────────────────────────────────── */
        input[type="text"], input[type="file"], select {{
            background: var(--color-bg);
            border: 1px solid var(--color-border);
            color: var(--color-text);
            padding: 8px 12px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            outline: none;
        }}

        input[type="text"]:focus, select:focus {{
            border-color: var(--color-green);
        }}

        .btn {{
            background: transparent;
            color: var(--color-text);
            border: 1px solid var(--color-border);
            padding: 8px 16px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            text-transform: uppercase;
            cursor: pointer;
        }}

        .btn:hover {{
            background: var(--color-hover);
        }}

        .btn-primary {{
            color: var(--color-bg);
            background: var(--color-green);
            border-color: var(--color-green);
            font-weight: bold;
        }}

        .btn-primary:hover {{
            background: transparent;
            color: var(--color-green);
        }}

        .btn-delete {{
            background: transparent;
            border: 1px solid var(--color-red);
            color: var(--color-red);
            padding: 4px 8px;
            font-size: 10px;
            text-transform: uppercase;
            cursor: pointer;
        }}

        .btn-delete:hover {{
            background: rgba(255,0,0,0.1);
        }}

        /* ── HTMX ──────────────────────────────────────────────────── */
        .htmx-indicator {{ opacity: 0; transition: none; }}
        .htmx-request .htmx-indicator {{ opacity: 1; }}

        /* ── Mono text ─────────────────────────────────────────────── */
        .mono {{ font-family: 'JetBrains Mono', monospace; }}

    </style>
</head>
<body>
<div class="shell">
    <aside class="sidebar">
        <div class="sidebar-logo">
            <div>
                <div class="logo-text">RAG_ENGINE</div>
                <div class="logo-sub">Adaptive Retrieval</div>
            </div>
        </div>

        <div class="nav-section">system_menu</div>
        {nav_items}

        <div class="sidebar-footer">
            <span>SYS.VER 1.0.0</span>
        </div>
    </aside>

    <header class="topbar">
        <div class="topbar-title">>{title}</div>
        <div class="topbar-right">
            <div class="live-badge">
                [AUTO-REFRESH: ON]
            </div>
            <a href="http://localhost:8000/docs" target="_blank" class="refresh-btn">
                [API DOCS]
            </a>
        </div>
    </header>

    <main class="main-content">
        {content}
    </main>
</div>
</body>
</html>"""

# Empty icons as we are using text-based [BRACKETS]
ICON_OVERVIEW = ""
ICON_DOCS = ""
ICON_QUERY = ""
ICON_EVAL = ""
ICON_CHAT = ""
