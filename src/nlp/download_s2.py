import os
import time
import json
import urllib.request
from semanticscholar import SemanticScholar

# --- Paths ---
output_folder = "data/papers"
metadata_file = "data/papers/metadata.json"
os.makedirs(output_folder, exist_ok=True)

# --- Find next paper number ---
existing = [f for f in os.listdir(output_folder) if f.startswith("paper") and f.endswith(".pdf")]
existing_nums = []
for f in existing:
    try:
        n = int(f.replace("paper", "").replace(".pdf", ""))
        existing_nums.append(n)
    except ValueError:
        continue

start_num = max(existing_nums) + 1 if existing_nums else 1
print(f"Existing papers: {len(existing_nums)}")
print(f"Next paper: paper{start_num}.pdf\n")

# --- Initialize Semantic Scholar ---
sch = SemanticScholar()

# --- Search queries ---
queries = [
    "high entropy alloy hydrogen evolution reaction",
    "high entropy alloy HER electrocatalyst",
    "high entropy alloy water splitting catalyst",
    "high entropy alloy overpotential HER",
    "high entropy alloy electrocatalysis hydrogen",
    "high entropy alloy catalyst hydrogen production",
    "multi-metal alloy hydrogen evolution",
    "nanoporous high entropy alloy HER",
]

TARGET = 100
MAX_PER_QUERY = 30

print("=" * 70)
print("SEMANTIC SCHOLAR PAPER DOWNLOADER")
print("=" * 70)
print(f"Target: {TARGET} new papers\n")

downloaded = 0
seen_dois = set()
metadata_records = []
failed = []

# Load existing metadata if any
if os.path.exists(metadata_file):
    with open(metadata_file, "r") as f:
        metadata_records = json.load(f)
        seen_dois = set(m.get("doi") for m in metadata_records if m.get("doi"))
    print(f"Loaded existing metadata: {len(metadata_records)} records\n")

for q_idx, query in enumerate(queries):
    if downloaded >= TARGET:
        break

    print(f"\n--- Query {q_idx+1}/{len(queries)}: {query} ---")

    try:
        results = sch.search_paper(
            query,
            limit=MAX_PER_QUERY,
            fields=["title", "abstract", "year", "externalIds",
                    "openAccessPdf", "authors", "venue"]
        )

        paper_count = 0
        for paper in results:
            if downloaded >= TARGET:
                break
            paper_count += 1

            # Get PDF URL
            pdf_info = getattr(paper, "openAccessPdf", None)
            if not pdf_info:
                continue

            pdf_url = None
            if isinstance(pdf_info, dict):
                pdf_url = pdf_info.get("url")
            else:
                pdf_url = getattr(pdf_info, "url", None)

            if not pdf_url:
                continue

            # Get DOI to avoid duplicates
            ext_ids = getattr(paper, "externalIds", {}) or {}
            doi = ext_ids.get("DOI", "")
            if doi and doi in seen_dois:
                continue

            title = (getattr(paper, "title", "") or "Unknown").strip()
            year = getattr(paper, "year", "N/A")
            short_title = title[:60]

            filename = f"paper{start_num + downloaded}.pdf"
            filepath = os.path.join(output_folder, filename)

            try:
                print(f"  [{downloaded+1}/{TARGET}] {filename}: {short_title}...")
                req = urllib.request.Request(
                    pdf_url,
                    headers={"User-Agent": "Mozilla/5.0 (research)"}
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = resp.read()

                if len(data) < 10000:
                    print(f"     SKIPPED (too small: {len(data)} bytes)")
                    continue

                with open(filepath, "wb") as f:
                    f.write(data)

                downloaded += 1
                if doi:
                    seen_dois.add(doi)

                metadata_records.append({
                    "file": filename,
                    "title": title,
                    "year": year,
                    "doi": doi,
                    "url": pdf_url,
                    "query": query,
                })

                time.sleep(1.0)

            except Exception as e:
                failed.append(f"{short_title} — {str(e)[:80]}")
                continue

        print(f"  Processed {paper_count} results from this query")

    except Exception as e:
        print(f"  Query failed: {e}")
        continue

# --- Save metadata ---
with open(metadata_file, "w") as f:
    json.dump(metadata_records, f, indent=2)

# --- Summary ---
print("\n" + "=" * 70)
print("DOWNLOAD COMPLETE")
print("=" * 70)
print(f"Successfully downloaded: {downloaded} new papers")
print(f"Failed: {len(failed)}")
print(f"Total papers now: {len(existing_nums) + downloaded}")
print(f"Metadata saved to: {metadata_file}")

if failed:
    print(f"\nFirst 5 failures:")
    for f in failed[:5]:
        print(f"  - {f}")