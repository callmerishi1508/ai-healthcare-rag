import os

if not os.path.exists("index.faiss"):
    from indexing.build_index import build_index
    build_index()

import streamlit as st
from pipeline.rag_pipeline import RAGPipeline

pipeline = RAGPipeline()

st.title("🏥 AI Healthcare Multi-Agent RAG")

query = st.text_input("Ask your question")

if st.button("Run"):
    result = pipeline.run(query)

    st.markdown("## ✅ Answer")
    st.markdown(result["answer"])

    with st.expander("🧩 Query Decomposition"):
        st.write(result["sub_queries"])

    with st.expander("📄 Retrieved Sources"):
        for c in result["context"]:
            st.write(f"{c['meta']['doc_id']} - {c['meta']['title']}")

    with st.expander("🛡️ Critic Review"):
        st.write(result["critique"])