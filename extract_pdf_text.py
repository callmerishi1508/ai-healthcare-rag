import pdfplumber

pdf_path = "healthcare_ai_corpus_v2.pdf"

with pdfplumber.open(pdf_path) as pdf:
    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        if text:
            print(f"Page {i+1}:\n{text}\n{'-'*50}")