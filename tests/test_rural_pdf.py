# tests/test_rural_pdf.py

from src.ingestion.pdf_loader import load_pdf_text

pdf_path = "data/raw/Rural_Health_Statistics_2021-22.pdf"

text = load_pdf_text(pdf_path)

print("TEXT LENGTH:", len(text))
print("\nFIRST 2000 CHARACTERS:\n")
print(text[:2000])