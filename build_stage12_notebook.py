# -*- coding: utf-8 -*-
"""Generate KernelKG_GPT_Stage12_API.ipynb — a CPU-only notebook that runs ONLY
Stage 1+2 (KG-GPT-style LLM retrieval) and saves the triple-pool cache.

It reuses the already-tested cells from KernelKG_GPT_Kaggle.ipynb (client,
data, type_dict, prompts, helpers, stage12 adapter, run_stage12) and replaces
the intro / install / config with trimmed, API-only versions. No GPU needed.

Run:  python build_stage12_notebook.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FULL_NB = os.path.join(HERE, "KernelKG_GPT_Kaggle.ipynb")

with open(FULL_NB, "r", encoding="utf-8") as f:
    full = json.load(f)


def cell_src(marker):
    """Return the source list of the first code cell containing `marker`."""
    for c in full["cells"]:
        if c["cell_type"] == "code" and marker in "".join(c["source"]):
            return c["source"]
    raise KeyError(marker)


cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.splitlines(keepends=True)})


def code_reuse(marker):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": cell_src(marker)})


# ===========================================================================
md(r"""# KernelKG-GPT — Stage 1+2 only (API, no GPU)

Runs **just the KG-GPT-style retrieval** (claim segmentation → candidate
relations → top-K relation ranking → triple pool) via the **NVIDIA API**, and
saves the triple-pool cache to `/kaggle/working/cache/`.

This step needs **only the API — no GPU**. Run it on a CPU session to save GPU
quota; then feed the produced `stage12_*.pkl` files into the Stage-3 training
notebook (`KernelKG_GPT_Kaggle.ipynb`).

## Before you run
1. **Add data**: attach the KG-GPT data as a Kaggle dataset, containing
   `factkg_train.pickle`, `factkg_dev.pickle`, `factkg_test.pickle`,
   `dbpedia_2015_undirected_light.pickle`, `relations_for_final.pickle`.
2. **Accelerator**: None (CPU is fine).
3. **Internet**: On (to reach the NVIDIA API).
4. **Paste your NVIDIA API key** in the Config cell (`NVIDIA_API_KEY`).

## Output
`/kaggle/working/cache/stage12_{train,dev,test}.pkl` — list of records, each
`{qid, claim, entity_set, sub_sentences, triple_pool, label, reasoning_types}`.
Download these (or "Save Version") and attach to the Stage-3 notebook.
""")

# --- install (only openai; no torch needed for this stage) ---
md("## 1. Install dependencies (only the OpenAI SDK)")
code('# %pip works on both Kaggle and local VSCode (Windows-safe)\n'
     '%pip install -q -U "openai>=1.30"\nprint("done")')

# --- trimmed, API-only config ---
md("## 2. Configuration (API + paths + limits only)")
code(r'''import os, glob, pickle, time, re, json, random
import numpy as np

# ---------------- NVIDIA API (Stage 1+2 LLM) ----------------
# Two (or more) keys for round-robin failover: when one key errors (rate limit,
# quota, transient failure), call_llm rotates to the next key and retries.
NVIDIA_API_KEYS = [
    "nvapi-PASTE_YOUR_KEY_1_HERE",
    "nvapi-PASTE_YOUR_KEY_2_HERE",
]
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
# >>> Set this to the EXACT model id from build.nvidia.com you want to use.
LLM_MODEL = "qwen/qwen3-235b-a22b"
DISABLE_THINKING = True      # append "/no_think" so Qwen3 returns the answer directly
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 2048
LLM_MAX_WORKERS = 2          # concurrent API calls. Free NVIDIA tiers rate-limit
                             # hard (HTTP 429) — keep this LOW (1-2). Raise only
                             # if you have a high-throughput/paid endpoint.
TOP_K = 5
MAX_TRIPLES = 30

# ---------------- How many claims to process per split ----------------
# >0 = first N claims ; 0 = skip ; -1 = the ENTIRE split.
# Full train + test (as requested); dev = validation subset (set -1 for full,
# 0 to skip). The run is RESUMABLE — safe to stop and re-run this notebook.
TRAIN_LIMIT = -1
DEV_LIMIT   = 2000
TEST_LIMIT  = -1

# ---------------- Paths (auto-detect Kaggle vs local/VSCode) ----------------
# os.name == "nt" => Windows (local). Kaggle is Linux with /kaggle/working.
if os.name != "nt" and os.path.isdir("/kaggle/working"):
    WORK = "/kaggle/working"                      # Kaggle
else:
    WORK = os.path.abspath("kkg_work")            # local (VSCode)
# Folders to search for the KG-GPT data pickles (first match wins).
DATA_DIRS = [
    "/kaggle/input",                               # Kaggle dataset mount
    "d:/Data mining/KernelGAT/kg-gpt/data",        # local repo (this machine)
    os.path.abspath(os.path.join("..", "kg-gpt", "data")),
    os.path.abspath("."),
]
CACHE_DIR = os.path.join(WORK, "cache"); os.makedirs(CACHE_DIR, exist_ok=True)
print("Config loaded. LLM_MODEL =", LLM_MODEL, "| WORK =", WORK)''')

# --- reused, already-tested cells ---
md("## 3. NVIDIA API client(s) + robust LLM call with key rotation")
code_reuse("def call_llm")

md("## 4. Locate & load data (FactKG + DBpedia)")
code_reuse("def find_file")

md("## 5. Build `type_dict` (once, cached)")
code_reuse("def build_type_dict")

md("## 6. KG-GPT prompts + helper functions")
code_reuse("SENTENCE_DIVIDE_PROMPT = ")
code_reuse("def graph_extractor")

md("## 7. Stage 1+2 — adapter (LLM retrieval) + triple pool builder")
code_reuse("def build_triple_pool")

md("## 8. Run Stage 1+2 over each split (cached, concurrent)\n"
   "Set a split's limit to 0 in the Config cell to skip it.")
code_reuse("def run_stage12")

md("## 8b. (Optional) Re-process only the empty-pool claims\n"
   "Keeps the cache; re-runs `process_claim` only on empty records (e.g. after "
   "improving entity handling). Marks tried ones so a re-run won't redo genuine "
   "empties; keeps the resume log in sync.")
code_reuse("def fix_empty_pools")

# --- new: preview + summary of the produced cache ---
md("## 9. Preview the cached output")
code(r'''def preview(cache_path, k=3):
    if not cache_path or not os.path.exists(cache_path):
        print(cache_path, "-> (skipped / not found)"); return
    with open(cache_path, "rb") as f:
        recs = pickle.load(f)
    sizes = [len(r["triple_pool"]) for r in recs]
    empty = sum(1 for s in sizes if s == 0)
    avg = (sum(sizes)/len(sizes)) if sizes else 0
    print(f"\n=== {os.path.basename(cache_path)} ===")
    print(f"records={len(recs)} | avg pool={avg:.2f} | empty={empty} "
          f"({100*empty/max(len(recs),1):.1f}%)")
    for r in recs[:k]:
        print("-"*60)
        print("claim :", r["claim"][:90])
        print("label :", r["label"], "| types:", r.get("reasoning_types"))
        print("pool  :", r["triple_pool"][:5], ("..." if len(r["triple_pool"])>5 else ""))

for p in [train_cache, dev_cache, test_cache]:
    preview(p)''')

md(r"""## 10. Next step

The cache files are now in `/kaggle/working/cache/`. To use them for Stage 3:

1. **Save Version** (commit) this notebook so the outputs persist, **or**
   download `cache/stage12_*.pkl`.
2. Open the Stage-3 notebook (`KernelKG_GPT_Kaggle.ipynb`) on a **GPU** session,
   attach these cache files, and point its `train_cache` / `dev_cache` /
   `test_cache` at them (or skip its Stage 1+2 cells since the cache exists).

> Re-running this notebook skips any split whose `stage12_{split}.pkl` already
> exists (delete the file to recompute).
""")

# ===========================================================================
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}

out_path = os.path.join(HERE, "KernelKG_GPT_Stage12_API.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("Wrote", out_path, "with", len(cells), "cells")
