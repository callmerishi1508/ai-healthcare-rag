# AI Healthcare Multi-Agent RAG

## Architecture Overview

This project implements a fully local agentic retrieval-augmented generation (RAG) system over a healthcare knowledge base.

### 1. Knowledge Indexing Pipeline
- Load `knowledge_base_ai_healthcare.json`.
- Chunk documents into 700-800 character segments with 200 character overlap to preserve context across boundaries.
- Create embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
- Store a FAISS index as `index.faiss` and metadata as `metadata.json`.
- Preserve document metadata: `doc_id`, `title`, `source`, `date`, and `chunk_id`.

### 2. Multi-Agent Architecture
- `RetrieverAgent`: hybrid dense + BM25 retrieval, with query routing for complex questions.
- `ReasoningAgent`: explicit, visible reasoning trace and an answer containing strict `doc_id` citations.
- `CriticAgent`: validates citation coverage and warns when retrieved sources are not cited.

### 3. Retrieval Strategy
- Dense retrieval uses FAISS and SentenceTransformer embeddings.
- BM25 retrieval uses `rank_bm25` over tokenized chunks.
- Hybrid scoring blends dense and BM25 scores for better recall.
- Simple query decomposition supports multi-hop reasoning by splitting compound questions.

### 4. UI and Conversation Flow
- Streamlit interface with query input, collapsible reasoning trace, and retrieved sources.
- Conversation history is preserved in session state for follow-up questions.

### 5. Evaluation
- `eval/eval_script.py` runs the 20 eval questions and computes heuristic scores for factual overlap, citation quality, reasoning clarity, and completeness.
- The evaluation saves `eval_results.json` with per-question details and average metrics.

## Improvements and Trade-offs
- The system uses local models only, avoiding external LLM dependencies.
- Citations are explicit but generated from extractive evidence rather than a full natural language generator.
- Evaluation metrics are heuristic and should be supplemented by human review for true factual accuracy.

## How to Run
1. `pip install -r requirements.txt`
2. `python indexing/build_index.py`
3. `streamlit run ui/app.py`
4. `python eval/eval_script.py`
