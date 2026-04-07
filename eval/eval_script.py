import json
import re
from pathlib import Path
from pipeline.rag_pipeline import RAGPipeline

EVAL_PATH = Path("../eval_set_ai_healthcare.json")
OUTPUT_PATH = Path("eval_results.json")

pipeline = RAGPipeline()

def score_overlap(answer, expected):
    if not expected:
        return 0.0
    answer_tokens = set(re.findall(r"\w+", answer.lower()))
    expected_tokens = set(re.findall(r"\w+", expected.lower()))
    if not expected_tokens:
        return 0.0
    return len(answer_tokens & expected_tokens) / len(expected_tokens)


def score_citations(answer, expected_sources):
    cited = set(re.findall(r"DOC-\d+", answer))
    expected_set = set(expected_sources)
    if not expected_set:
        return 1.0 if cited else 0.0

    true_positives = len(cited & expected_set)
    precision = true_positives / len(cited) if cited else 0.0
    recall = true_positives / len(expected_set)
    return (precision + recall) / 2


def score_reasoning(trace):
    if not trace:
        return 0.0
    steps = [line for line in trace.splitlines() if line.strip()]
    return min(1.0, len(steps) / 4)


with EVAL_PATH.open("r", encoding="utf-8") as f:
    data = json.load(f)

results = []
scores = {"factual_accuracy": [], "citation_quality": [], "reasoning_clarity": [], "completeness": []}

for item in data.get("questions", []):
    question = item["question"]
    expected_answer = item.get("expected_answer", "")
    expected_sources = item.get("source_docs", [])

    result = pipeline.run(question, use_history=False)
    answer = result["answer"]
    trace = result.get("trace", "")

    factual = score_overlap(answer, expected_answer)
    citation = score_citations(answer, expected_sources)
    reasoning = score_reasoning(trace)
    completeness = factual

    scores["factual_accuracy"].append(factual)
    scores["citation_quality"].append(citation)
    scores["reasoning_clarity"].append(reasoning)
    scores["completeness"].append(completeness)

    summary = {
        "eval_id": item.get("eval_id"),
        "question": question,
        "answer": answer,
        "expected_answer": expected_answer,
        "expected_sources": expected_sources,
        "score": {
            "factual_accuracy": factual,
            "citation_quality": citation,
            "reasoning_clarity": reasoning,
            "completeness": completeness,
        },
    }
    results.append(summary)

with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    json.dump({"results": results, "averages": {k: sum(v) / len(v) if v else 0.0 for k, v in scores.items()}}, f, indent=2)

print("Evaluation complete. Results written to", OUTPUT_PATH)
print("Averages:")
for key, values in scores.items():
    print(f"{key}: {sum(values) / len(values) if values else 0.0:.3f}")
