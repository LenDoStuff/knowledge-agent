# UI

This package owns Streamlit presentation and interaction state. `app.py` only
composes the page; `claims.py` handles claim discovery and uploaded PDFs,
`knowledge_base.py` renders persisted claim evidence, and `research.py` drives
planning, approval, research, report history, and audit views.

UI actions explicitly start ingestion, index rebuilds, agent calls, and history
mutations through the owning package APIs. The UI does not implement provider
clients or persisted formats. It protects active research from index rebuilds,
lets ingestion and rebuild create Custom, LightRAG, or both indexes, and asks
the user to choose an available engine before each new research interaction. It
keeps full-audit warnings visible and surfaces failures instead of selecting a
different retrieval or model path.
