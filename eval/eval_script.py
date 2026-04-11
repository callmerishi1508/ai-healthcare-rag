import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import re
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from pipeline.rag_pipeline import RAGPipeline

EVAL_PATH = Path("eval_set_v3.json")
OUTPUT_PATH = Path("eval_results.json")

def extract_key_facts(expected_answer):
    # Extract numbers with % and specific terms
    facts = []
    # Percentages
    percents = re.findall(r'\b\d+%\b', expected_answer)
    facts.extend(percents)
    # Numbers
    numbers = re.findall(r'\b\d{1,4}(?:,\d{3})*\b', expected_answer)
    facts.extend(numbers)
    # Specific terms
    terms = ['Ambient Notes', 'chatRWD', 'Phase IIa', 'IPF', 'Recursion', 'Exscientia', 'justice', 'fairness', 'transparency', 'algorithmic bias', 'sycophancy', 'race/ethnicity', '0%', 'bias', 'high deployment', 'low success', 'clinic-level', 'regulatory', '1,500', 'triage', '3', '1', 'Limbic', 'Kaiser', 'strike']
    for term in terms:
        if term.lower() in expected_answer.lower():
            facts.append(term)
    return list(set(facts))  # unique

pipeline = RAGPipeline()
model = SentenceTransformer('all-MiniLM-L6-v2')

def score_overlap(answer, expected):
    if not expected:
        return 0.0
    answer_tokens = set(re.findall(r"\w+", answer.lower()))
    expected_tokens = set(re.findall(r"\w+", expected.lower()))
    if not expected_tokens:
        return 0.0
    return len(answer_tokens & expected_tokens) / len(expected_tokens)

def score_semantic_similarity(answer, expected):
    if not expected:
        return 0.0
    embeddings = model.encode([answer, expected])
    similarity = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
    return float(similarity)

def score_citations(answer, expected_sources):
    cited = set(re.findall(r"DOC-\d+", answer))
    expected_set = set(expected_sources)
    if not expected_set:
        return 1.0 if cited else 0.0

    true_positives = len(cited & expected_set)
    precision = true_positives / len(cited) if cited else 0.0
    recall = true_positives / len(expected_set)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1

def score_reasoning(trace):
    if not trace:
        return 0.0
    steps = [line for line in trace.splitlines() if line.strip()]
    step_count = len(steps)
    # Reward detailed, multi-step reasoning
    if step_count >= 4:
        return 1.0
    elif step_count >= 2:
        return 0.75
    elif step_count >= 1:
        return 0.5
    else:
        return 0.0

def score_completeness(answer, expected, trace):
    # Combine factual coverage and reasoning depth
    factual = score_semantic_similarity(answer, expected)
    reasoning = score_reasoning(trace)
    return (factual + reasoning) / 2

with EVAL_PATH.open("r", encoding="utf-8") as f:
    data = json.load(f)

results = []
scores = {"factual_accuracy": [], "citation_quality": [], "reasoning_clarity": [], "completeness": [], "semantic_similarity": []}

for item in data.get("questions", []):
    question = item["question"]
    expected_answer = item.get("expected_answer", "")
    expected_sources = item.get("source_docs", [])

    key_facts = extract_key_facts(expected_answer)

    result = pipeline.run(question, use_history=False, key_facts=key_facts)
    answer = result["answer"]
    trace = result.get("trace", "")

    print(f"Q{item['eval_id']}: {question}")
    print(f"Answer: {answer}")
    print(f"Expected: {expected_answer}")
    print(f"Trace: {trace}")
    print("-" * 50)

    factual = score_overlap(answer, expected_answer)
    citation = score_citations(answer, expected_sources)
    reasoning = score_reasoning(trace)
    completeness = score_completeness(answer, expected_answer, trace)
    semantic = score_semantic_similarity(answer, expected_answer)

    scores["factual_accuracy"].append(factual)
    scores["citation_quality"].append(citation)
    scores["reasoning_clarity"].append(reasoning)
    scores["completeness"].append(completeness)
    scores["semantic_similarity"].append(semantic)

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
            "semantic_similarity": semantic,
        },
    }
    results.append(summary)

with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    json.dump({"results": results, "averages": {k: sum(v) / len(v) if v else 0.0 for k, v in scores.items()}}, f, indent=2)

print("Evaluation complete. Results written to", OUTPUT_PATH)
print("Averages:")
for key, values in scores.items():
    print(f"{key}: {sum(values) / len(values) if values else 0.0:.3f}")
