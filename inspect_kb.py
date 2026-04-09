import json
with open('knowledge_base_ai_healthcare.json','r',encoding='utf-8') as f:
    kb=json.load(f)
docs={doc['doc_id'] for doc in kb['documents']}
print('num docs',len(docs))
for q in ['DOC-001','DOC-004','DOC-009','DOC-010','DOC-011','DOC-012','DOC-013','DOC-015','DOC-016','DOC-017','DOC-018','DOC-019','DOC-021']:
    print(q, q in docs)
