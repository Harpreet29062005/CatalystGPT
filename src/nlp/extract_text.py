import pdfplumber
import os
import glob

# Define paths
pdf_folder = "data/papers"
output_folder = "data/raw"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Find all PDF files
pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))

print(f"Found {len(pdf_files)} PDF files to process.\n")

for pdf_path in pdf_files:
    # Get filename without extension
    filename = os.path.basename(pdf_path)
    text_filename = filename.replace(".pdf", ".txt")
    output_path = os.path.join(output_folder, text_filename)
    
    print(f"Extracting text from: {filename}")
    
    try:
        # Open the PDF
        with pdfplumber.open(pdf_path) as pdf:
            full_text = ""
            # Loop through every page
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n\n"
            
            # Save the extracted text to a .txt file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(full_text)
                
        print(f"  -> Saved to: {output_path}")
        
    except Exception as e:
        print(f"  -> ERROR processing {filename}: {e}")

print("\nExtraction complete!")