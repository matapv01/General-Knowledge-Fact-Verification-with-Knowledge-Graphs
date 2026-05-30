# -*- coding: utf-8 -*-
"""Generate KernelKG_GPT_Stage3_Colab.ipynb — trains the KernelGAT verifier
(Stage 3) and evaluates it, on Google Colab (GPU), consuming the Stage 1+2
cache produced earlier (stage12_{train,dev,test}.pkl on Google Drive).

Reuses the tested Stage-3 cells (dataset / model / train / eval) from
KernelKG_GPT_Kaggle.ipynb; only the intro, install, mount, and config cells
are Colab-specific. Stage 3 uses plain torch + transformers (BertModel), so no
version pinning is needed.

Run:  python build_stage3_colab_notebook.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FULL_NB = os.path.join(HERE, "KernelKG_GPT_Kaggle.ipynb")

with open(FULL_NB, "r", encoding="utf-8") as f:
    full = json.load(f)


def cell_src(marker):
    for c in full["cells"]:
        if c["cell_type"] == "code" and marker in "".join(c["source"]):
            return c["source"]
    raise KeyError(marker)


cells = []


def md(t):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)})


def code(t):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": t.splitlines(keepends=True)})


def reuse(marker):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": cell_src(marker)})


# ===========================================================================
md(r"""# KernelKG-GPT — Stage 3 (train + evaluate) on Colab

Trains the **KernelGAT** verifier on the Stage 1+2 triple-pool cache and
evaluates it on the test split.

- Input: `MyDrive/kernelkg_gpt/cache/stage12_{train,dev,test}.pkl`
  (produced by the Stage 1+2 notebook).
- Output: best checkpoint → `MyDrive/kernelkg_gpt/outputs/best.pt`.

Each `(claim, triple)` becomes a node `[CLS] claim [SEP] head rel tail [SEP]`;
the model scores nodes with 21 Gaussian kernels + a sentence-level GAT and
aggregates to a True/False decision (NLL loss). This is plain
torch+transformers — **no vLLM, no version pinning**.

## Before you run
1. **Runtime → GPU** (T4 is enough; A100 faster).
2. Make sure the Stage 1+2 caches exist on Drive (run the Stage 1+2 notebook
   first). `dev` cache is used for validation / early stopping.
3. Run top-to-bottom.

## Fair-comparison tip
Set `MODEL_TYPE = "concat_baseline"` to train the supervised baseline on the
*same* triples (no kernel/GAT). Compare with `MODEL_TYPE = "kernel"` to isolate
the KernelGAT architecture's contribution.
""")

# --- GPU check ---
md("## 0. Check GPU")
code("!nvidia-smi")

# --- install (light; Colab already has torch + transformers) ---
md("## 1. Dependencies\nColab already ships torch + transformers (which include `BertModel` and `get_linear_schedule_with_warmup`). We only ensure they're importable.")
code(r'''import torch, transformers
print("torch", torch.__version__, "| transformers", transformers.__version__,
      "| GPU", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")''')

# --- mount drive ---
md("## 2. Mount Google Drive")
code(r'''from google.colab import drive
drive.mount("/content/drive")
print("Drive mounted.")''')

# --- config ---
md("## 3. Configuration")
code(r'''import os, pickle, random
import numpy as np
from tqdm.auto import tqdm

# ---- Model ----
MODEL_TYPE   = "kernel"            # "kernel" | "concat_baseline"
BERT_MODEL   = "bert-base-uncased"
NUM_KERNELS  = 21
NUM_LABELS   = 2
DROPOUT      = 0.1
MAX_SEQ_LEN  = 96                  # triples are short
MAX_NODES    = 10                  # KG-GPT returns up to ~30 triples; cap to 10 nodes
TRIPLE_FORMAT = "plain"            # "plain" | "separators"

# ---- Training ----
BATCH_SIZE   = 8
EVAL_BATCH   = 16
GRAD_ACCUM   = 4
LR           = 5e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS   = 5
WARMUP_RATIO = 0.1
EARLY_STOP_PATIENCE = 3
SEED         = 42

# ---- Paths (Google Drive) ----
WORK        = "/content/drive/MyDrive/kernelkg_gpt"
CACHE_DIR   = os.path.join(WORK, "cache")
OUTPUT_DIR  = os.path.join(WORK, "outputs"); os.makedirs(OUTPUT_DIR, exist_ok=True)
train_cache = os.path.join(CACHE_DIR, "stage12_train.pkl")
dev_cache   = os.path.join(CACHE_DIR, "stage12_dev.pkl")
test_cache  = os.path.join(CACHE_DIR, "stage12_test.pkl")

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    import torch; torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
set_seed()

for _p in (train_cache, dev_cache, test_cache):
    print(("OK   " if os.path.exists(_p) else "MISSING"), _p)
print("model_type:", MODEL_TYPE, "| output:", OUTPUT_DIR)''')

# --- reused, already-tested Stage 3 cells ---
md("## 4. Dataset (graph builder + tokenization)")
reuse("class FactKGGraphDataset")

md("## 5. Model — KernelGAT verifier (+ fair baseline)")
reuse("class KernelKGGPT")

md("## 6. Train (with dev early-stopping) + per-reasoning-type eval")
reuse("def train():")

md("## 7. Final evaluation on the test split")
reuse("TEST accuracy")

# --- extra: save metrics + predictions ---
md("## 8. (Optional) Save test predictions + metrics to Drive")
code(r'''import json
model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best.pt"), map_location=device))
model.eval()

rows, correct, total = [], 0, 0
from collections import defaultdict
type_stats = defaultdict(lambda: {"c": 0, "t": 0})
with torch.no_grad():
    for batch in test_loader:
        out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                    batch["token_type_ids"].to(device), batch["is_null"].to(device))
        probs = out["logits"].exp().cpu()           # logits are log-probs
        preds = probs.argmax(-1)
        for i in range(len(batch["claims"])):
            p = int(preds[i]); y = int(batch["labels"][i])
            correct += int(p == y); total += 1
            for t in batch["reasoning_types"][i]:
                type_stats[t]["t"] += 1; type_stats[t]["c"] += int(p == y)
            rows.append({"claim": batch["claims"][i], "pred": p, "label": y,
                         "p_true": float(probs[i, 1]),
                         "reasoning_types": batch["reasoning_types"][i]})

metrics = {"model_type": MODEL_TYPE, "test_accuracy": correct / max(total, 1),
           "n": total,
           "per_type": {t: s["c"] / s["t"] for t, s in type_stats.items() if s["t"]}}
with open(os.path.join(OUTPUT_DIR, f"test_metrics_{MODEL_TYPE}.json"), "w") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)
with open(os.path.join(OUTPUT_DIR, f"test_predictions_{MODEL_TYPE}.jsonl"), "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(json.dumps(metrics, indent=2, ensure_ascii=False))
print("Saved metrics + predictions to", OUTPUT_DIR)''')

md(r"""## 9. Notes

- **Memory**: effective batch = `BATCH_SIZE × GRAD_ACCUM`. Each item is
  `MAX_NODES` BERT passes, so a batch is `BATCH_SIZE × MAX_NODES` sequences.
  If you OOM, lower `BATCH_SIZE` or `MAX_NODES`, or raise `GRAD_ACCUM`.
- **Fair baseline**: rerun with `MODEL_TYPE = "concat_baseline"`; the metrics
  file is suffixed by model type so both are kept. Report
  `kernel` vs `concat_baseline` to show the architecture's contribution.
- **Tokenization is cached in-memory** (uint16) per dataset at load time; the
  first epoch's start spends a minute tokenizing the train split.
- **Resume**: best checkpoint is saved to Drive each time dev accuracy improves.
""")

# ===========================================================================
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "accelerator": "GPU", "colab": {"provenance": []},
    },
    "nbformat": 4, "nbformat_minor": 5,
}

out_path = os.path.join(HERE, "KernelKG_GPT_Stage3_Colab.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("Wrote", out_path, "with", len(cells), "cells")
