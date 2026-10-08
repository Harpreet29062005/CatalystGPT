import spacy
import re
import os
import glob
import pandas as pd

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

# Define paths
raw_text_folder = "data/raw"
output_file = "data/processed/extracted_data.csv"

# Create output folder if it does not exist
os.makedirs("data/processed", exist_ok=True)

# Find all text files
text_files = glob.glob(os.path.join(raw_text_folder, "*.txt"))

print(f"Found {len(text_files)} text files to process.\n")

# List to store extracted data
results = []

# Regex patterns
overpotential_pattern = re.compile(r"(\d+\.?\d*)\s*mV", re.IGNORECASE)
tafel_pattern = re.compile(r"(\d+\.?\d*)\s*mV\s*/?\s*dec", re.IGNORECASE)

# Common elements found in HEA catalysts
elements = ["Pt", "Mo", "Pd", "Rh", "Ni", "Fe", "Co", "Mn", "Cr", "Cu",
            "Ru", "Ir", "Au", "Ag", "Ti", "V", "Zn", "Sn", "W", "Nb"]

for text_file in text_files:
    filename = os.path.basename(text_file)
    print(f"Processing: {filename}")
    
    with open(text_file, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Process with spaCy (limit to 100k chars for speed)
    doc = nlp(text[:100000])
    
    # Extract sentences that mention overpotential or Tafel
    for sent in doc.sents:
        sent_text = sent.text
        
        if "overpotential" in sent_text.lower() or "tafel" in sent_text.lower():
            overpotential_match = overpotential_pattern.search(sent_text)
            overpotential = float(overpotential_match.group(1)) if overpotential_match else None
            
            tafel_match = tafel_pattern.search(sent_text)
            tafel = float(tafel_match.group(1)) if tafel_match else None
            
            found_elements = []
            for element in elements:
                if element in sent_text:
                    found_elements.append(element)
            
            if found_elements and (overpotential or tafel):
                catalyst = "".join(sorted(set(found_elements)))
                results.append({
                    "paper": filename,
                    "catalyst_elements": catalyst,
                    "overpotential_mV": overpotential,
                    "tafel_mV_per_dec": tafel,
                    "sentence": sent_text[:300]
                })
    
    print(f"  -> Total entries so far: {len(results)}")

# Save results to CSV
if results:
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"\nExtraction complete!")
    print(f"Saved {len(results)} rows to: {output_file}")
    print(f"\nFirst 10 rows:")
    print(df.head(10))
else:
    print("\nNo results found. Check the regex patterns.")