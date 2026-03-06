
import PyPDF2
import os

pdf_files = [
    "(clean) 2025-A computational approach to evaluate how molecular mechanisms impact large-scale brain activity.pdf",
    "2025-Simulated 5-HT2A receptor activation accounts for the high complexity of brain activity during psychedelic states.pdf"
]

keywords = ["coefficients", "polynomial", "P_e", "P_i", "fitting"]

for pdf_file in pdf_files:
    if os.path.exists(pdf_file):
        print(f"--- Processing {pdf_file} ---")
        try:
            reader = PyPDF2.PdfReader(pdf_file)
            references = []
    
            # Keywords to search for voltage equation
            keywords = ["coefficients", "polynomial", "P_e", "P_i", "fitting"]
            
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                
                for kw in keywords:
                    if kw.lower() in text.lower():
                        idx = text.lower().find(kw.lower())
                        start = max(0, idx - 150)
                        end = min(len(text), idx + len(kw) + 150) 
                        print(f"--- Found '{kw}' on page {page_num + 1} in {pdf_file} ---") 
                        print(text[start:end].replace('\n', ' '))
                        print("-" * 50)
                
            # The original instruction had a syntax error here: 'return referencest(f"--- Page {i+1} ---")'
            # and a leftover 'print(text[:3000])'. These have been removed.


        except Exception as e:
            print(f"Error reading {pdf_file}: {e}")
    else:
        print(f"File not found: {pdf_file}")
