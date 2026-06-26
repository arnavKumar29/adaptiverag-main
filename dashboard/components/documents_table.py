"""
Documents table HTMX partial — ingested document list with status and upload form.
Terminal brutalist text-based layout.
"""
from __future__ import annotations


def render_documents_table(docs: list[dict]) -> str:
    """Render the documents management table and ingestion form."""

    # Ingestion Form
    form_html = """
    <div class="card" style="margin-bottom: 24px; border-color: var(--color-blue);">
        <div class="card-header" style="background: var(--color-blue-dim);">
            <span class="card-title" style="color: var(--color-blue);">[ DATA_INGESTION ]</span>
        </div>
        <div class="card-body">
            <form hx-post="/ingest" hx-encoding="multipart/form-data" hx-target="#documents-table" hx-indicator="#upload-indicator" style="display: flex; gap: 16px; align-items: flex-end; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 250px;">
                    <label style="display: block; font-size: 10px; color: var(--color-label); margin-bottom: 4px;">TARGET FILE (PDF, DOCX, TXT, MD)</label>
                    <input type="file" name="file" accept=".pdf,.docx,.txt,.md" required style="width: 100%;">
                </div>
                <div style="flex: 1; min-width: 150px;">
                    <label style="display: block; font-size: 10px; color: var(--color-label); margin-bottom: 4px;">COLLECTION (OPTIONAL)</label>
                    <input type="text" name="collection" placeholder="default" style="width: 100%;">
                </div>
                <div>
                    <button type="submit" class="btn btn-primary">[ EXECUTE UPLOAD ]</button>
                </div>
            </form>
            <div id="upload-indicator" class="htmx-indicator" style="margin-top: 12px; font-size: 11px; color: var(--color-blue);">
                > UPLOADING DATASTREAM... <span class="loading"></span>
            </div>
            <div id="upload-result"></div>
        </div>
    </div>
    """

    if not docs:
        table_html = """
        <div class="card">
            <div class="empty-state">
                <p>[ NO DOCUMENTS IN DATABASE ]</p>
                <p style="opacity: 0.5;">AWAITING FILE UPLOAD...</p>
            </div>
        </div>
        """
        return form_html + table_html

    rows = ""
    for doc in docs:
        doc_id = str(doc.get("id", ""))[:8]
        filename = doc.get("filename", "—")
        source_type = doc.get("source_type", "—")
        status = doc.get("status", "—")
        chunks = doc.get("chunk_count", "—") or "—"
        collection = doc.get("collection", "default") or "default"
        created = doc.get("created_at", "—")

        # Status styling
        status_cls = {
            "indexed": "badge-ok",
            "processing": "badge-blue",
            "pending": "badge-warn",
            "failed": "badge-error",
        }.get(status, "badge-warn")

        # Format timestamp
        if created and created != "—":
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                created = dt.strftime("%b %d %H:%M")
            except Exception:
                pass

        rows += f"""
        <tr>
            <td><span class="mono" style="color: var(--color-label);">0x{doc_id}</span></td>
            <td style="color: var(--color-text); max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{filename}</td>
            <td><span class="badge" style="color: var(--color-muted); border-color: transparent;">{source_type.upper()}</span></td>
            <td><span class="badge {status_cls}">{status.upper()}</span></td>
            <td style="color: var(--color-green);">{chunks}</td>
            <td style="color: var(--color-muted);">{collection}</td>
            <td style="font-size: 11px; color: var(--color-label);">{created}</td>
            <td>
                <button class="btn-delete"
                    hx-delete="/api/documents/{doc.get('id', '')}"
                    hx-confirm="Delete this document? This cannot be undone."
                    hx-target="#documents-table"
                    hx-swap="innerHTML">
                    [DEL]
                </button>
            </td>
        </tr>
        """

    count = len(docs)
    table_html = f"""
    <div class="card">
        <div class="card-header">
            <span class="card-title">[ DOCUMENT_INDEX ]</span>
            <span style="font-size: 10px; color: var(--color-muted);">TOTAL: {count}</span>
        </div>
        <div style="overflow-x: auto;">
            <table class="data-table">
                <thead>
                    <tr>
                        <th>[ID]</th>
                        <th>[FILENAME]</th>
                        <th>[EXT]</th>
                        <th>[STATUS]</th>
                        <th>[CHUNKS]</th>
                        <th>[COLLECTION]</th>
                        <th>[CREATED]</th>
                        <th>[ACT]</th>
                    </tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
    </div>
    """

    return form_html + table_html
