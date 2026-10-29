import requests
import numpy as np
import random
from collections import Counter
import time
import pandas as pd

# =========================
# CONFIG
# =========================
START_YEAR = 2000
END_YEAR = 2023

K_PER_YEAR = 10000
TOP_K_CONCEPTS = 300

EMAIL = "alessandro.spelta@unipv.it"
BASE_URL = "https://api.openalex.org/works"

# =========================
# FILTERS (IMPORTANT)
# =========================
def build_filters(year):
    return {
        "publication_year": f"{year}",
        "has_references": "true",
        "type": "article",
        "cited_by_count": ">50"
    }

# =========================
# FETCH FUNCTION
# =========================
def fetch_year(year, max_items):
    cursor = "*"
    results = []

    while len(results) < max_items:
        params = {
            "filter": ",".join([f"{k}:{v}" for k, v in build_filters(year).items()]),
            "per_page": 200,
            "cursor": cursor,
            "select": "id,publication_year,concepts,referenced_works,cited_by_count"
        }

        headers = {"User-Agent": f"mailto:{EMAIL}"}

        r = requests.get(BASE_URL, params=params, headers=headers)
        if r.status_code != 200:
            print("API error:", r.text)
            break

        data = r.json()
        batch = data.get("results", [])

        if not batch:
            break

        results.extend(batch)

        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break

        time.sleep(0.1)

    # deterministic truncation
    results = sorted(results, key=lambda x: x.get("cited_by_count", 0), reverse=True)

    return results[:max_items]

# =========================
# CONCEPT NORMALIZATION
# =========================
def norm_concepts(concepts):
    if not concepts:
        return []
    s = sum(c.get("score", 0) for c in concepts)
    if s == 0:
        return []
    return [(c["display_name"], c["score"]/s) for c in concepts]

# =========================
# DOWNLOAD DATA
# =========================
all_data = {}

print("Downloading filtered OpenAlex dataset...")

for year in range(START_YEAR, END_YEAR + 1):
    print(f"Year {year}")

    data = fetch_year(year, K_PER_YEAR)
    all_data[year] = data

    print(f"  Collected {len(data)} papers")

# =========================
# BUILD GLOBAL CONCEPT SPACE
# =========================
print("\nBuilding concept space...")

global_counts = Counter()

paper_concepts = {}
paper_year = {}

for year in all_data:
    for p in all_data[year]:

        pid = p["id"]
        paper_year[pid] = year
        paper_concepts[pid] = norm_concepts(p.get("concepts", []))

        for c, w in paper_concepts[pid]:
            global_counts[c] += w

top_concepts = [c for c, _ in global_counts.most_common(TOP_K_CONCEPTS)]
index = {c:i for i,c in enumerate(top_concepts)}

N = len(top_concepts)
T = END_YEAR - START_YEAR + 1

print(f"Concepts selected: {N}")

# =========================
# BUILD OT TENSOR
# =========================
T_tensor = np.zeros((N, N, T))

print("\nBuilding OT tensor...")

missing_refs = 0
valid_refs = 0

for year in all_data:
    t = year - START_YEAR

    for p in all_data[year]:

        src_id = p["id"]
        src = paper_concepts.get(src_id, [])

        if not src:
            continue

        refs = p.get("referenced_works", [])
        if not refs:
            continue

        for ref in refs:

            tgt = paper_concepts.get(ref, [])
            if not tgt:
                missing_refs += 1
                continue

            valid_refs += 1

            # OT-safe normalization
            src_norm = sum(w for _, w in src) + 1e-12
            tgt_norm = sum(w for _, w in tgt) + 1e-12

            for i, wi in src:
                if i not in index:
                    continue
                ii = index[i]

                for j, wj in tgt:
                    if j not in index:
                        continue
                    jj = index[j]

                    T_tensor[ii, jj, t] += (wi/src_norm) * (wj/tgt_norm)


# =========================
# SAVE OUTPUTS
# =========================
np.save("ot_filtered_tensor.npy", T_tensor)

pd.DataFrame(top_concepts, columns=["concept"]).to_csv(
    "concept_list.csv",
    index=False
)

# =========================
# DIAGNOSTICS
# =========================
print("\n===== SUMMARY =====")
print(f"Tensor shape: {T_tensor.shape}")
print(f"Missing references: {missing_refs}")
print(f"Valid references: {valid_refs}")
print(f"Sparsity: {1 - np.count_nonzero(T_tensor)/T_tensor.size:.4f}")

print("\nSaved:")
print("- ot_filtered_tensor.npy")
print("- concept_list.csv")
