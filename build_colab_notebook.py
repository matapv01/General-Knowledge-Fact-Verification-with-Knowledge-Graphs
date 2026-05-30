# -*- coding: utf-8 -*-
"""Generate KernelKG_GPT_Stage12_Colab.ipynb — runs Stage 1+2 on Google Colab
using a LOCALLY hosted Qwen2.5-7B, instead of a remote API.

Backends (config flag BACKEND):
  - "vllm" (DEFAULT): vLLM **offline** `LLM` engine using the user's proven
    recipe (vllm==0.7.3, transformers==4.48.3, AWQ model, enforce_eager,
    enable_prefix_caching). A dynamic micro-batcher feeds concurrent claims to
    `llm.generate([...])`. The shared few-shot prefix is prefix-cached → fast.
  - "transformers" (fallback): HF model + micro-batcher (no vLLM/version
    juggling). Robust but slower.

Cache → Google Drive (resumable). Reuses the tested data / type_dict /
helpers / prompts / stage12 / run_stage12 cells from KernelKG_GPT_Kaggle.ipynb.

Run:  python build_colab_notebook.py
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
md(r"""# KernelKG-GPT — Stage 1+2 on Colab with a **local Qwen2.5-7B (vLLM offline)**

KG-GPT-style retrieval (claim segmentation → candidate relations → top-K
ranking → triple pool) with Qwen hosted **locally** — no API, no rate limits.

Uses the **vLLM offline engine** (proven recipe: `vllm==0.7.3`,
`transformers==4.48.3`, full-precision fp16 weights, `enforce_eager`,
`enable_prefix_caching`).
A dynamic **micro-batcher** turns the concurrent per-claim calls into batched
`llm.generate([...])` calls; because every prompt shares the same long
few-shot prefix, **prefix caching** makes this very fast.

| BACKEND | Notes |
|---|---|
| **`vllm`** (default) | Fast. Pinned versions known to work (incl. T4 / A100). |
| `transformers` | Fallback using Colab's native stack + a HF micro-batcher. Slower but no version pinning. |

## Before you run
1. **Runtime → GPU** (A100 recommended — full-precision 7B fits its 80 GB).
2. **Upload KG-GPT data to Drive**, e.g. `MyDrive/kggpt_data/` with
   `factkg_train.pickle`, `factkg_dev.pickle`, `factkg_test.pickle`,
   `dbpedia_2015_undirected_light.pickle`, `relations_for_final.pickle`.
3. Run top-to-bottom. If the install asks to **Restart runtime**, do it, then
   re-run from the *Config* cell. Cache → `MyDrive/kernelkg_gpt/cache/`.
""")

# --- GPU check ---
md("## 0. Check GPU")
code("!nvidia-smi")

# --- config (first: defines BACKEND + sets vLLM env vars before any vllm import) ---
md("## 1. Configuration")
code(r'''import os, glob, pickle, time, re, json, random
import numpy as np

# vLLM engine env (must be set before importing vllm) -- from the proven recipe
os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------- Backend ----------------
BACKEND = "vllm"              # "vllm" (default) | "transformers" (fallback)
LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"   # 3B: faster + leaves lots of KV-cache room
DISABLE_THINKING = False      # Qwen2.5-Instruct is not a thinking model
LLM_TEMPERATURE = 0.0         # greedy -> deterministic, easiest to parse
LLM_MAX_TOKENS = 256          # outputs are short (a list / a few lines)
LLM_MAX_WORKERS = 256         # concurrent claims -> fills the batch queue
TOP_K = 5
MAX_TRIPLES = 30
MAX_MODEL_LEN = 4096          # vLLM context window (input + output)

# micro-batcher (both backends)
TFM_BATCH = 128               # claims per generate() call
TFM_COALESCE_WAIT = 0.05      # seconds to collect a batch

# vLLM engine knobs — tuned for 3B on a 40GB GPU with spare system RAM
VLLM_GPU_UTIL = 0.90          # ~36GB reserved; 3B weights ~6GB -> ~30GB KV cache
VLLM_MAX_NUM_SEQS = 200       # many concurrent seqs fit with the small model

# ---------------- How many claims per split ----------------
# >0 = first N ; 0 = skip ; -1 = ENTIRE split.
TRAIN_LIMIT = -1
DEV_LIMIT   = 2000
TEST_LIMIT  = -1

# ---------------- Paths (Google Drive => persistent). Dirs made after mount. --
WORK = "/content/drive/MyDrive/kernelkg_gpt"
CACHE_DIR = os.path.join(WORK, "cache")
DATA_DIRS = [
    "/content/drive/MyDrive/kggpt_data",
    "/content/drive/MyDrive",
    "/content",
]
print("Config loaded. BACKEND =", BACKEND, "| Model =", LLM_MODEL)''')

# --- install (conditional + idempotent) ---
md(r"""## 2. Install dependencies

`vllm` backend installs the **pinned, proven** combo (`vllm==0.7.3`,
`transformers==4.48.3`, `tokenizers==0.21.0`) and removes a broken `torchcodec`.
Because this changes `torch`/`transformers`, the kernel **must be restarted**
before importing them — otherwise you get errors like
`cannot import name 'is_offline_mode'` (a half-old / half-new transformers).

**This cell auto-restarts the runtime after installing.** When it restarts,
just run all cells again (Runtime → Run all): the install is idempotent, sees
the correct versions, skips, and continues without restarting.
""")
code(r'''import subprocess, sys

def _pip(*args):
    subprocess.run([sys.executable, "-m", "pip", *args], check=False)

if BACKEND == "vllm":
    _pip("uninstall", "-y", "-q", "torchcodec")     # broken on Colab, breaks imports
    need_restart = False

    # Pin transformers + tokenizers FIRST (so vllm's deps don't override them),
    # then vllm. Detect whether anything changed -> restart needed.
    try:
        import transformers
        tf_ok = transformers.__version__.startswith("4.48")
    except Exception:
        tf_ok = False
    if not tf_ok:
        _pip("install", "-q", "transformers==4.48.3", "tokenizers==0.21.0"); need_restart = True
    else:
        print("transformers", transformers.__version__, "OK")

    try:
        import vllm
        vllm_ok = vllm.__version__.startswith("0.7")
    except Exception:
        vllm_ok = False
    if not vllm_ok:
        # --no-deps so vllm doesn't pull a newer transformers over our pin
        _pip("install", "-q", "vllm==0.7.3")
        _pip("install", "-q", "transformers==4.48.3", "tokenizers==0.21.0")  # re-pin after vllm
        need_restart = True
    else:
        print("vllm", vllm.__version__, "OK")

    if need_restart:
        print("\n>>> Installed pinned versions. RESTARTING runtime to apply them...")
        print(">>> After it restarts, run all cells again (Runtime -> Run all).")
        import os, time
        time.sleep(1)
        os.kill(os.getpid(), 9)     # Colab auto-restarts the kernel
    else:
        print("Dependencies ready (no restart needed).")
else:
    _pip("install", "-q", "-U", "accelerate")
    print("transformers backend — using Colab's native torch/transformers.")''')

# --- mount drive (+ make cache dir) ---
md("## 3. Mount Google Drive")
code(r'''from google.colab import drive
drive.mount("/content/drive")
os.makedirs(CACHE_DIR, exist_ok=True)
print("Drive mounted. CACHE_DIR =", CACHE_DIR)''')

# --- load model + build call_llm (micro-batched, branches on backend) ---
md(r"""## 4. Load the model + build `call_llm` (dynamic micro-batching)

A background batcher coalesces the concurrent per-claim requests into one
`generate([...])` call. With `vllm`, prefix caching reuses the shared few-shot
prompt prefix across all calls → big speedup. Keep `LLM_MAX_WORKERS ≥ TFM_BATCH`
so batches stay full.
""")
code(r'''import threading, queue

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
def _clean(text):
    if not text: return ""
    text = _THINK_RE.sub("", text)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()

def _wrap_chat(user_msg, system_msg="You are a helpful assistant."):
    return (f"<|im_start|>system\n{system_msg}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n")

_q = queue.Queue()

if BACKEND == "vllm":
    import transformers
    assert transformers.__version__.startswith("4.48"), (
        f"transformers is {transformers.__version__}, but vLLM 0.7.3 needs 4.48.x. "
        "Re-run the install cell and let it RESTART the runtime, then run all again."
    )
    from vllm import LLM, SamplingParams
    print(f"Loading {LLM_MODEL} with vLLM ...")
    _llm = LLM(
        model=LLM_MODEL,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=VLLM_GPU_UTIL,
        dtype="float16",
        enforce_eager=True,
        enable_prefix_caching=True,
        swap_space=16,                 # use the abundant free system RAM for KV overflow
        max_num_seqs=VLLM_MAX_NUM_SEQS,
        disable_log_stats=True,
    )
    _SP = SamplingParams(temperature=LLM_TEMPERATURE, max_tokens=LLM_MAX_TOKENS,
                         stop=["<|im_end|>", "<|im_start|>"])

    def _generate_batch(prompts):
        outs = _llm.generate([_wrap_chat(p) for p in prompts], _SP, use_tqdm=False)
        return [o.outputs[0].text for o in outs]

else:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {LLM_MODEL} with transformers (bf16) ...")
    _tok = AutoTokenizer.from_pretrained(LLM_MODEL)
    _tok.padding_side = "left"
    if _tok.pad_token is None:
        _tok.pad_token = _tok.eos_token
    _model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL, torch_dtype=torch.bfloat16).to("cuda").eval()

    def _generate_batch(prompts):
        texts = [_tok.apply_chat_template(
                    [{"role": "system", "content": "You are a helpful assistant."},
                     {"role": "user", "content": p}],
                    tokenize=False, add_generation_prompt=True) for p in prompts]
        enc = _tok(texts, return_tensors="pt", padding=True, truncation=True,
                   max_length=MAX_MODEL_LEN).to(_model.device)
        with torch.no_grad():
            out = _model.generate(**enc, max_new_tokens=LLM_MAX_TOKENS,
                                   do_sample=LLM_TEMPERATURE > 0,
                                   temperature=max(LLM_TEMPERATURE, 1e-2), top_p=0.9,
                                   pad_token_id=_tok.eos_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        return _tok.batch_decode(gen, skip_special_tokens=True)

def _batch_worker():
    while True:
        first = _q.get()
        if first is None: return
        batch = [first]
        deadline = time.time() + TFM_COALESCE_WAIT
        while len(batch) < TFM_BATCH:
            try:
                nxt = _q.get(timeout=max(0.0, deadline - time.time()))
            except queue.Empty:
                break
            if nxt is None:
                _q.put(None); break
            batch.append(nxt)
        try:
            texts = _generate_batch([b["text"] for b in batch])
            for b, t in zip(batch, texts):
                b["result"] = _clean(t); b["event"].set()
        except Exception as e:
            print("[batcher] generate error:", e)
            for b in batch:
                b["result"] = ""; b["event"].set()

_batcher = threading.Thread(target=_batch_worker, daemon=True)
_batcher.start()

def call_llm(prompt, sweeps=1):
    item = {"text": prompt, "event": threading.Event(), "result": None}
    _q.put(item)
    item["event"].wait()
    return item["result"]

print(f"call_llm ready (backend={BACKEND}, batch={TFM_BATCH}, workers={LLM_MAX_WORKERS}).")
print("LLM test:", repr(call_llm("Reply with the single word: ok")[:50]))''')

# --- reused, already-tested cells ---
md("## 5. Locate & load data (FactKG + DBpedia)")
reuse("def find_file")

md("## 6. Build `type_dict` (once, cached to Drive)")
reuse("def build_type_dict")

md("## 7. KG-GPT prompts + helper functions")
reuse("SENTENCE_DIVIDE_PROMPT = ")
reuse("def graph_extractor")

md("## 8. Stage 1+2 — adapter + triple pool builder")
reuse("def build_triple_pool")

md("## 9. Run Stage 1+2 over each split (concurrent + resumable)")
reuse("def run_stage12")

md("## 9b. (Optional) Re-process only the empty-pool claims\n"
   "Keeps the cache; re-runs `process_claim` only on records with an empty "
   "`triple_pool` (e.g. after the gold-entity fix). Marks tried ones so a re-run "
   "won't redo genuine empties; keeps the resume log in sync.")
reuse("def fix_empty_pools")

# --- preview ---
md("## 10. Preview the cached output")
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

md(r"""## 11. Notes

- **Speed**: raise `TFM_BATCH` / `VLLM_MAX_NUM_SEQS` (A100 has room) and keep
  `LLM_MAX_WORKERS ≥ TFM_BATCH`. Prefix caching makes the shared few-shot prompt
  nearly free after the first call.
- **Resume**: cache lives in `MyDrive/kernelkg_gpt/cache/`. If Colab
  disconnects, re-run — it continues from `stage12_{split}.partial.jsonl`.
- **type_dict** is built once over DBpedia and cached to Drive.
- **Next**: feed `stage12_{train,dev,test}.pkl` into the Stage-3 training
  notebook (`KernelKG_GPT_Kaggle.ipynb`).
- If `vllm` ever fails to install/load, set `BACKEND="transformers"` in Config
  and re-run from there.
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

out_path = os.path.join(HERE, "KernelKG_GPT_Stage12_Colab.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("Wrote", out_path, "with", len(cells), "cells")
