"""
Chat UI component for the dashboard.
Hermes-inspired brutalist minimal coding design.
Uses JetBrains Mono everywhere.
"""
from __future__ import annotations

import json

def render_chat_ui() -> str:
    """Render the main chat interface shell."""
    return """
    <style>
        /* Hermes / Brutalist / Coding theme */
        .chat-container {
            font-family: 'JetBrains Mono', monospace !important;
            background-color: #000000;
            border: 1px solid #333;
            height: calc(100vh - 160px);
            display: flex;
            flex-direction: column;
            color: #e5e5e5;
        }
        
        .chat-header {
            border-bottom: 1px solid #333;
            padding: 16px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: #050505;
        }
        
        .chat-header-title {
            font-size: 14px;
            font-weight: 600;
            letter-spacing: 1px;
            text-transform: uppercase;
            color: #fff;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .status-pulse {
            width: 8px;
            height: 8px;
            background-color: #00ff00;
            box-shadow: 0 0 10px #00ff00;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        
        .chat-history {
            flex-grow: 1;
            overflow-y: auto;
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 24px;
            scroll-behavior: smooth;
        }
        
        .msg-block {
            display: flex;
            flex-direction: column;
            max-width: 85%;
        }
        
        .msg-block.user {
            align-self: flex-end;
            align-items: flex-end;
        }
        
        .msg-block.ai {
            align-self: flex-start;
            align-items: flex-flex-start;
        }
        
        .msg-label {
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 6px;
            opacity: 0.6;
        }
        
        .msg-content {
            padding: 16px;
            font-size: 13px;
            line-height: 1.6;
            border: 1px solid #333;
            background: #0a0a0a;
            white-space: pre-wrap;
        }
        
        .msg-block.user .msg-content {
            border-color: #555;
            background: #111;
            color: #fff;
        }
        
        .msg-block.ai .msg-content {
            border-color: #00ff00;
            border-left: 3px solid #00ff00;
        }
        
        .msg-content p {
            margin-bottom: 1em;
        }
        .msg-content p:last-child {
            margin-bottom: 0;
        }
        
        .msg-sources {
            margin-top: 10px;
            padding: 12px;
            border: 1px dashed #333;
            background: #050505;
            font-size: 11px;
            color: #aaa;
        }
        
        .source-item {
            margin-top: 6px;
        }
        .source-item a {
            color: #00ff00;
            text-decoration: none;
        }
        .source-item a:hover {
            text-decoration: underline;
        }
        
        .chat-input-area {
            border-top: 1px solid #333;
            padding: 20px;
            background: #050505;
        }
        
        .chat-form {
            display: flex;
            gap: 12px;
            align-items: flex-end;
        }
        
        .chat-textarea {
            flex-grow: 1;
            background: #000;
            border: 1px solid #444;
            color: #fff;
            padding: 14px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
            resize: none;
            height: 52px;
            transition: all 0.2s;
            outline: none;
        }
        
        .chat-textarea:focus {
            border-color: #00ff00;
            box-shadow: 0 0 0 1px rgba(0, 255, 0, 0.2);
        }
        
        .chat-submit {
            background: #fff;
            color: #000;
            border: none;
            height: 52px;
            padding: 0 24px;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
            font-size: 13px;
            text-transform: uppercase;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .chat-submit:hover {
            background: #00ff00;
        }
        
        /* HTMX indicator */
        .htmx-indicator {
            display: none;
        }
        .htmx-request .htmx-indicator {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            color: #00ff00;
            font-size: 12px;
            text-transform: uppercase;
        }
        .htmx-request .submit-text {
            display: none;
        }
        
        /* Markdown styles within chat */
        .msg-content code {
            background: #1a1a1a;
            padding: 2px 6px;
            color: #00ff00;
        }
        .msg-content pre {
            background: #111;
            padding: 12px;
            border: 1px solid #333;
            overflow-x: auto;
            margin: 10px 0;
        }
        .msg-content pre code {
            background: transparent;
            padding: 0;
            color: #e5e5e5;
        }
    </style>

    <div class="page-header" style="font-family: 'JetBrains Mono', monospace;">
        <h1 style="font-weight: 500; text-transform: uppercase; font-size: 20px;">[ TERMINAL // CHAT ]</h1>
        <p style="font-size: 12px; letter-spacing: 1px; color: #888;">Adaptive RAG Subsystem</p>
    </div>

    <div class="chat-container">
        <div class="chat-header">
            <div class="chat-header-title">
                <div class="status-pulse"></div>
                SYSTEM ONLINE
            </div>
            <div style="font-size: 11px; color: #666;">AWAITING_INPUT</div>
        </div>
        
        <div class="chat-history" id="chat-history">
            <div class="msg-block ai">
                <div class="msg-label">SYSTEM_READY</div>
                <div class="msg-content">Initializing RAG protocol...\nReady to process queries. Enter text below to begin.</div>
            </div>
        </div>
        
        <div class="chat-input-area">
            <form class="chat-form"
                  hx-post="/chat/send" 
                  hx-target="#chat-history" 
                  hx-swap="beforeend"
                  hx-on::after-request="this.reset(); document.getElementById('chat-history').scrollTop = document.getElementById('chat-history').scrollHeight; renderMarkdown();">
                
                <textarea 
                    name="query" 
                    class="chat-textarea" 
                    placeholder="> Enter command or query..." 
                    required
                    onkeydown="if(event.keyCode==13 && !event.shiftKey) { event.preventDefault(); this.form.dispatchEvent(new Event('submit', {cancelable: true})); }"></textarea>
                
                <button type="submit" class="chat-submit">
                    <span class="submit-text">EXECUTE</span>
                    <span class="htmx-indicator">
                        [ PROCESSING ]
                    </span>
                </button>
            </form>
        </div>
    </div>
    
    <script>
        // Simple script to render markdown after each response
        function renderMarkdown() {
            document.querySelectorAll('.markdown-raw').forEach(el => {
                if (!el.classList.contains('rendered')) {
                    el.innerHTML = marked.parse(el.textContent);
                    el.classList.add('rendered');
                }
            });
        }
        
        // Auto-scroll on load
        document.addEventListener('DOMContentLoaded', () => {
            const h = document.getElementById('chat-history');
            h.scrollTop = h.scrollHeight;
        });
        
        // Scroll on any HTMX update
        document.body.addEventListener('htmx:afterSwap', function(evt) {
            if(evt.detail.target.id === 'chat-history') {
                const h = document.getElementById('chat-history');
                h.scrollTop = h.scrollHeight;
                renderMarkdown();
            }
        });
    </script>
    """


def render_chat_message(role: str, content: str, sources: list[dict] | None = None, latency: int | None = None) -> str:
    """Render a single chat message."""
    
    is_user = role == "user"
    css_class = "user" if is_user else "ai"
    label = "USER_INPUT" if is_user else "AI_RESPONSE"
    
    if not is_user and latency:
        label += f" [ {latency}ms ]"
        
    html = f"""
    <div class="msg-block {css_class}">
        <div class="msg-label">{label}</div>
    """
    
    # We add a class `markdown-raw` for the JS to parse, unless it's user input (keep it raw text)
    if is_user:
        import html as html_lib
        safe_content = html_lib.escape(content)
        html += f'<div class="msg-content">{safe_content}</div>'
    else:
        # Hide raw markdown inside a div that marked.js will transform
        import html as html_lib
        safe_content = html_lib.escape(content)
        html += f'<div class="msg-content"><div class="markdown-raw">{safe_content}</div></div>'
        
        # Add sources if available
        if sources:
            html += '<div class="msg-sources"><div style="margin-bottom:8px; border-bottom:1px solid #333; padding-bottom:4px;">[ RETRIEVED_SOURCES ]</div>'
            for idx, src in enumerate(sources):
                doc_name = src.get('source') or 'Unknown Source'
                score = src.get('score', 0)
                html += f'<div class="source-item">[{idx+1}] {doc_name} <span style="opacity:0.5;">(Score: {score:.3f})</span></div>'
            html += '</div>'
            
    html += "</div>"
    return html
