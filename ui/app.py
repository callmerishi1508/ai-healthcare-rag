import os
import sys
from pathlib import Path

import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

if not os.path.exists(root / "index.faiss"):
    from indexing.build_index import build_index
    build_index()

from pipeline.rag_pipeline import RAGPipeline

st.set_page_config(page_title="AI Healthcare Multi-Agent RAG", layout="wide")

if "history" not in st.session_state:
    st.session_state.history = []

pipeline = RAGPipeline()

st.title("🏥 AI Healthcare Multi-Agent RAG")
st.write("Ask a healthcare research question and see step-by-step evidence, citations, and retrieval sources.")

with st.form(key="query_form"):
    query = st.text_input("Ask your question", key="query_input")
    use_history = st.checkbox("Use conversation context for follow-ups", value=True)
    submit = st.form_submit_button("Run")

if submit and query:
    result = pipeline.run(query, use_history=use_history)
    st.session_state.history.append(result)

if st.session_state.history:
    with st.expander("💬 Conversation history", expanded=False):
        for i, item in enumerate(result["sub_queries"], 1):
            st.markdown(f"**Query {i}:** {item['query']}")
            st.markdown(f"**Answer:** {item['answer']}")
            st.markdown(f"**Critic:** {item['critique']}")
            st.markdown("---")

if st.session_state.history:
    latest = st.session_state.history[-1]
    st.markdown("## ✅ Answer")
    st.markdown(latest["answer"])

    with st.expander("🧩 Query Decomposition", expanded=True):
        st.write(latest["sub_queries"])

    with st.expander("📄 Retrieved Sources", expanded=True):
        for c in latest["context"]:
            st.write(f"**{c['meta']['doc_id']}** — {c['meta'].get('title')} ({c['score']:.3f})")
            if c['meta'].get('source'):
                st.write(f"Source: {c['meta'].get('source')} | {c['meta'].get('date')}")
            if c['meta'].get('url'):
                st.write(c['meta'].get('url'))
            st.markdown("---")

    with st.expander("🧠 Reasoning Trace", expanded=False):
        st.code(latest.get("trace", "No reasoning trace available."))

    with st.expander("🛡️ Critic Review", expanded=False):
        st.write(latest["critique"])
