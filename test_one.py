import sys
sys.path.insert(0, '.')
import json
from pathlib import Path
from pipeline.rag_pipeline import RAGPipeline

pipeline = RAGPipeline()
with open('eval_set_v3.json', 'r') as f:
    data = json.load(f)

item = data['questions'][0]
question = item['question']
expected = item['expected_answer']

result = pipeline.run(question, use_history=False)
answer = result['answer']
trace = result.get('trace', '')
context = result.get('context', [])

print(f'Q{item["eval_id"]}: {question}')
print(f'Answer: {answer}')
print(f'Expected: {expected}')
print(f'Trace: {trace}')
print('Retrieved chunks:')
for i, ctx in enumerate(context):
    print(f'{i+1}: {ctx["text"][:200]}... [Citation: {ctx["meta"]["doc_id"]}]')
print('-' * 50)