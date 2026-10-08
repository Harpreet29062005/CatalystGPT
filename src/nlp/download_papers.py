import arxiv
import os
import time
import urllib.request

# --- Paths ---
output_folder = "data/papers"
os.makedirs(output_folder, exist_ok=True)

# --- Find the next available paper number ---
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
print(f"Next paper will be: paper{start_num}.pdf\n")


# --- Search queries ---
# Multiple queries to find diverse papers
queries = [
    'all:"high entropy alloy" AND all:"hydrogen evolution"',
    'all:"high entropy alloy" AND all:"HER catalyst"',
    'all:"high entropy alloy" AND all:"electrocatalyst" AND all:"hydrogen"',
    'all:"high entropy alloy" AND all:"water splitting"',
    'all:"high entropy alloy" AND all:"overpotential"',
]

TARGET = 100   # Total new papers to download
MAX_PER_QUERY = 40

print("=" * 70)
print("ARXIV AUTOMATED PAPER DOWNLOADER")
print("=" * 70)
print(f"Target: {TARGET} new papers\n")

downloaded = 0
seen_titles = set()
failed = []

for q_idx, query in enumerate(queries):
    if downloaded >= TARGET:
        break

    print(f"\n--- Query {q_idx+1}/{len(queries)}: {query} ---")

    try:
        client = arxiv.Client(
            page_size=50,
            delay_seconds=3,
            num_retries=3,
        )
        search = arxiv.Search(
            query=query,
            max_results=MAX_PER_QUERY,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        results = list(client.results(search))
        print(f"  Found {len(results)} results")

        for result in results:
            if downloaded >= TARGET:
                break

            title = result.title.strip().lower()
            if title in seen_titles:
                continue
            seen_titles.add(title)

            # Sanitize title for printing
            short_title = result.title[:60].replace("\n", " ")

            try:
                pdf_url = result.pdf_url
                filename = f"paper{start_num + downloaded}.pdf"
                filepath = os.path.join(output_folder, filename)

                # Download with a 30-second timeout
                print(f"  [{downloaded+1}/{TARGET}] {filename}: {short_title}...")
                urllib.request.urlretrieve(pdf_url, filepath)

                # Check the file is not empty
                size = os.path.getsize(filepath)
                if size < 10000:  # < 10 KB means it's probably broken
                    os.remove(filepath)
                    print(f"     SKIPPED (file too small: {size} bytes)")
                    continue

                downloaded += 1
                time.sleep(1.5)  # Be polite to arXiv servers

            except Exception as e:
                failed.append(f"{short_title} — {e}")
                print(f"     FAILED: {e}")
                continue

    except Exception as e:
        print(f"  Query failed: {e}")
        continue

# --- Summary ---
print("\n" + "=" * 70)
print("DOWNLOAD COMPLETE")
print("=" * 70)
print(f"Successfully downloaded: {downloaded} papers")
print(f"Failed: {len(failed)}")
print(f"Total papers now in data/papers/: {len(existing_nums) + downloaded}")

if failed:
    print(f"\nFailures (first 5):")
    for f in failed[:5]:
        print(f"  - {f}")

print(f"\nAll files saved to: {output_folder}")