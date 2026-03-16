"""Manual demo for workspace-level search (DATAMAN-163).

Usage:
    python tests/manual/demo_workspace_search.py

Uses staging credentials from CLAUDE.md.
"""

import os

import roboflow

thisdir = os.path.dirname(os.path.abspath(__file__))
os.environ["ROBOFLOW_CONFIG_DIR"] = f"{thisdir}/data/.config"

WORKSPACE = "model-evaluation-workspace"

rf = roboflow.Roboflow()
ws = rf.workspace(WORKSPACE)

# --- Single page search ---
print("=== Single page search ===")
page = ws.search("project:false", page_size=5)
print(f"Total results: {page['total']}")
print(f"Results in this page: {len(page['results'])}")
print(f"Continuation token: {page.get('continuationToken')}")
for img in page["results"]:
    print(f"  - {img.get('filename', 'N/A')}")

# --- Paginated search_all ---
print("\n=== Paginated search_all (page_size=3, max 2 pages) ===")
count = 0
for page_results in ws.search_all("*", page_size=3):
    count += 1
    print(f"Page {count}: {len(page_results)} results")
    for img in page_results:
        print(f"  - {img.get('filename', 'N/A')}")
    if count >= 2:
        print("(stopping after 2 pages for demo)")
        break

print("\nDone.")
