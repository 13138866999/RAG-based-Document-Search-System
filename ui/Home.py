# ui/Home.py
import time
from pathlib import Path
import re
import streamlit as st

import os, sys
from pathlib import Path
import streamlit as st
def ensure_state():
    st.session_state.setdefault("persist_dir", str(Path("./db").resolve()))
    st.session_state.setdefault("k", 5)
    st.session_state.setdefault("threshold", 0.2)
    # 👇 This is the history store; do NOT overwrite it later
    st.session_state.setdefault("chat_history", [])  # [{role, content, meta?}]
    st.session_state.setdefault("pipeline_ready", False)
    st.session_state.setdefault("first_load_error", None)
    st.session_state.setdefault("is_answering", False)


ensure_state()
ROOT = Path(__file__).resolve().parents[1]  # parent of ui/ -> project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # make 'backend' importable

print("cwd:", os.getcwd())
print("project_root:", ROOT)
print("sys.path:", sys.path[:3])
# ---------- page meta ----------
st.set_page_config(page_title="RAG Agent · Chat", page_icon="💬", layout="wide")

# ---------- imports from your backend ----------
# Make sure backend/ has an __init__.py and you're running from project root.
from backend.rag import ChromaData  # your pipeline with load_data(...) and ask(...)

# ---------- helpers ----------
def ensure_state():
    st.session_state.setdefault("persist_dir", str(Path("./db").resolve()))
    st.session_state.setdefault("k", 5)
    st.session_state.setdefault("threshold", 0.2)
    st.session_state.setdefault("chat_history", [])  # list[{"role": "user"/"assistant", "content": str, "meta": dict}]
    st.session_state.setdefault("pipeline_ready", False)
    st.session_state.setdefault("first_load_error", None)

def highlight_snippet(snippet: str, query: str) -> str:
    """Very light keyword highlighter for source snippets."""
    if not query:
        return snippet
    words = [w for w in re.split(r"\W+", query) if len(w) > 2]
    if not words:
        return snippet
    out = snippet
    for w in sorted(set(words), key=len, reverse=True):
        out = re.sub(fr"(?i)({re.escape(w)})", r"**\1**", out)
    return out

ensure_state()

# ---------- SIDEBAR ----------
st.sidebar.header("🔧 Settings")
st.sidebar.caption("These affect retrieval for *new* messages.")
st.session_state.setdefault("k", 5)
st.session_state.setdefault("threshold", 0.2)

st.sidebar.slider("Top-k", 1, 10, key="k")
st.sidebar.slider("Score threshold", 0.0, 1.0, step=0.05, format="%.2f", key="threshold")

st.sidebar.divider()

persist_dir = Path(st.session_state["persist_dir"])
st.sidebar.write("**Persist dir**")
st.sidebar.code(str(persist_dir), language=None)

# Optional: quick actions
col_a, col_b = st.sidebar.columns(2)
with col_a:
    if st.button("Reload DB", use_container_width=True):
        try:
            st.session_state["pipeline"] = ChromaData()
            st.session_state["pipeline"].load_data(str(persist_dir))
            st.session_state["pipeline_ready"] = True
            st.session_state["first_load_error"] = None
            st.toast("Vector DB reloaded.", icon="✅")
        except Exception as e:
            st.session_state["first_load_error"] = str(e)
            st.session_state["pipeline_ready"] = False
            st.toast("Failed to load DB. See main area.", icon="❌")
with col_b:
    # Navigate to Upload/Ingest page if you created it under pages/
    if hasattr(st, "switch_page"):
        if st.button("Upload & Ingest", use_container_width=True):
            st.switch_page("pages/Upload_Ingest.py")

st.sidebar.divider()
st.sidebar.caption("Tip: If DB isn't ready, go to **Upload & Ingest** first.")

# ---------- MAIN HEADER ----------
st.title("💬 RAG Agent")
st.caption("Ask questions about your uploaded documents. Answers include citations and snippets.")

# ---------- PIPELINE INIT (lazy) ----------
if "pipeline" not in st.session_state:
    st.session_state["pipeline"] = ChromaData()

if not st.session_state["pipeline_ready"]:
    # Try to load once automatically
    try:
        st.session_state["pipeline"].load_data(str(persist_dir))
        st.session_state["pipeline_ready"] = True
        st.session_state["first_load_error"] = None
    except Exception as e:
        st.session_state["first_load_error"] = str(e)
        st.session_state["pipeline_ready"] = False

# Show readiness / guidance
ready_col1, ready_col2 = st.columns([1, 2], vertical_alignment="center")
with ready_col1:
    if st.session_state["pipeline_ready"]:
        st.success("Vector DB loaded ✔", icon="✅")
    else:
        st.error("Vector DB not loaded", icon="⚠️")
with ready_col2:
    if not st.session_state["pipeline_ready"]:
        st.info(
            "If this is a fresh project or you changed machines, first go to **Upload & Ingest** and build the index. "
            "Then click **Reload DB** here."
        )
        if st.session_state["first_load_error"]:
            with st.expander("Show load error"):
                st.code(st.session_state["first_load_error"])

st.divider()
user_input = st.chat_input("Ask about your documents…")
has_new = user_input is not None

# ---------- CHAT HISTORY RENDER ----------
# Render existing messages (bubbles)
for i, msg in enumerate(st.session_state["chat_history"]):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        meta = msg.get("meta") or {}
        sources = meta.get("sources") or []
        is_last_msg = (i == len(st.session_state["chat_history"]) - 1)
        skip_sources = has_new and is_last_msg and msg["role"] == "assistant"
        if sources and not skip_sources:
            with st.expander("Sources", expanded=False):
                from pathlib import Path

                for s in sources:
                    head = f"[{s.get('index', '?')}] {Path(s.get('source', '?')).name} · chunk {s.get('chunk', '?')} · score={s.get('score', 0):.3f}"
                    st.markdown(f"**{head}**")
                    st.markdown(s.get("content", ""))
                    st.divider()
# ---------- CHAT INPUT ----------

if user_input:

    # A) show user bubble immediately (this run)
    with st.chat_message("user"):
        st.markdown(user_input)

    # B) persist it to history for future reruns
    st.session_state["chat_history"].append({
        "role": "user",
        "content": user_input
    })
    st.session_state["is_answering"] = True

    # C) create assistant bubble with spinner and typing effect
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("_Thinking…_")

        with st.spinner("Searching docs and composing an answer…"):
            k = int(st.session_state["k"])
            threshold = float(st.session_state["threshold"])
            try:
                result = st.session_state["pipeline"].ask(user_input, k=k, threshold=threshold)
            except Exception as e:
                result = {"answer": f"**Error:** {e}", "sources": [], "latency_s": None}

        # typing effect (optional)
        import time
        answer = (result.get("answer") or "").strip()
        typed = ""
        for ch in answer:
            typed += ch
            placeholder.markdown(typed)
            time.sleep(0.008)

        # latency + sources in same bubble
        if result.get("latency_s") is not None:
            st.caption(f"Answered in {result['latency_s']:.3f}s")

        sources = result.get("sources", [])
        if sources:
            with st.expander("Sources", expanded=False):
                from pathlib import Path

                for s in sources:
                    head = f"[{s.get('index', '?')}] {Path(s.get('source', '?')).name} · chunk {s.get('chunk', '?')} · score={s.get('score', 0):.3f}"
                    st.markdown(f"**{head}**")
                    st.markdown(s.get("content", ""))
                    st.divider()

    # D) persist assistant turn so it re-renders on the next rerun
    st.session_state["chat_history"].append({
        "role": "assistant",
        "content": answer or "_No answer returned._",
        "meta": {
            "sources": sources,
            "latency_s": result.get("latency_s")
        }
    })
    st.session_state["is_answering"] = False

    # IMPORTANT: do NOT st.rerun() here; let this run finish so the bubbles show now
