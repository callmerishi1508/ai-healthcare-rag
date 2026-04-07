import json
from pipeline.rag_pipeline import RAGPipeline

EVAL_PATH = "/mnt/data/eval_set_ai_healthcare[1].json"

pipeline = RAGPipeline()

with open(EVAL_PATH) as f:
    data = json.load(f)

questions = data["questions"]

for q in questions:
    print("="*80)
    print("QUESTION:", q["question"])

    result = pipeline.run(q["question"])

    print("\nANSWER:\n", result["answer"])