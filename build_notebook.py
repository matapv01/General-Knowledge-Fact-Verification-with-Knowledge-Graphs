# -*- coding: utf-8 -*-
"""Generate KernelKG_GPT_Kaggle.ipynb — a self-contained Kaggle notebook
running the full KernelKG-GPT pipeline (Stage 1+2 via NVIDIA API + Stage 3
KernelGAT training/eval on GPU).

Run:  python build_notebook.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
KGGPT_PROMPTS = os.path.join(HERE, "..", "kg-gpt", "prompts")


def read(p):
    with open(p, "r", encoding="utf-8") as f:
        return f.read()


SENTENCE_DIVIDE_PROMPT = read(os.path.join(KGGPT_PROMPTS, "sentence_divide_prompt.txt"))
RELATION_RETRIEVAL_PROMPT = read(os.path.join(KGGPT_PROMPTS, "relation_retrieval_prompt.txt"))

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})


def code(text):
    cells.append({
        "cell_type": "code", "metadata": {}, "execution_count": None,
        "outputs": [], "source": text.splitlines(keepends=True),
    })


# ===========================================================================
md(r"""# KernelKG-GPT — End-to-End on Kaggle (GPU)

Fact verification on **FactKG** combining:

- **Stage 1+2 — KG-GPT** (LLM retrieval): claim segmentation → candidate
  relations (Algorithm 1) → top-K relation ranking. Runs via the **NVIDIA
  API** (OpenAI-compatible) with a Qwen model.
- **Stage 3 — KernelGAT** (kernel reasoning): a trainable BERT + Gaussian
  kernel graph-attention verifier over `(claim, triple)` nodes.

The notebook runs the whole flow: build `type_dict` → Stage 1+2 (cached) →
train Stage 3 → evaluate.

## Before you run
1. **Add data**: attach the KG-GPT data as a Kaggle dataset. It must contain:
   `factkg_train.pickle`, `factkg_dev.pickle`, `factkg_test.pickle`,
   `dbpedia_2015_undirected_light.pickle`, `relations_for_final.pickle`.
2. **Enable GPU**: Settings → Accelerator → GPU.
3. **Enable Internet**: Settings → Internet → On (needed to download
   `bert-base-uncased` and to reach the NVIDIA API).
4. **Add your NVIDIA API key** as a Kaggle Secret named `NVIDIA_API_KEY`
   (Add-ons → Secrets), or paste it in the Config cell.

> **Note on scale**: Stage 1+2 calls the API ~2-3× per claim. The full FactKG
> train split (~86k) is impractical in one session, so the Config cell uses
> subset limits. Increase them as your time/quota allow.
""")

# ---------------------------------------------------------------------------
md("## 1. Install dependencies")
code(r"""# %pip works both on Kaggle and locally (Windows/VSCode). Installs into the
# running kernel. Kaggle ships torch+transformers; locally install them too if
# missing:  %pip install -q -U "openai>=1.30" torch transformers
%pip install -q -U "openai>=1.30"
print("done")""")

# ---------------------------------------------------------------------------
md("## 2. Configuration\nAll knobs in one place.")
code(r'''import os, glob, pickle, time, re, json, math, random
import numpy as np

# ---------------- NVIDIA API (Stage 1+2 LLM) ----------------
# Two (or more) keys for round-robin failover: when one key errors (rate limit,
# quota, transient failure), call_llm rotates to the next key and retries.
NVIDIA_API_KEYS = [
    "nvapi-PASTE_YOUR_KEY_1_HERE",
    "nvapi-PASTE_YOUR_KEY_2_HERE",
]
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
# >>> Set this to the EXACT model id from build.nvidia.com that you want to use.
#     (You mentioned a Qwen ~122B-A10B model; if its NIM id differs, change it here.)
LLM_MODEL = "qwen/qwen3-235b-a22b"
DISABLE_THINKING = True      # append "/no_think" so Qwen3 returns the answer directly
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 2048
LLM_MAX_WORKERS = 2          # concurrent API calls. Free NVIDIA tiers rate-limit
                             # hard (HTTP 429) — keep this LOW (1-2). Raise only
                             # if you have a high-throughput/paid endpoint.
TOP_K = 5
MAX_TRIPLES = 30

# ---------------- How many claims to process per split (API stage) ----------
# >0 = first N claims ; 0 = skip ; -1 = the ENTIRE split.
# Full train + test (as requested). Dev kept as a validation subset for early
# stopping (set -1 for full dev, or 0 to skip). The run is resumable.
TRAIN_LIMIT = -1
DEV_LIMIT   = 2000
TEST_LIMIT  = -1

# ---------------- Stage 3 (KernelGAT) ----------------
MODEL_TYPE   = "kernel"        # "kernel" | "concat_baseline"
BERT_MODEL   = "bert-base-uncased"
NUM_KERNELS  = 21
NUM_LABELS   = 2
DROPOUT      = 0.1
MAX_SEQ_LEN  = 96
MAX_NODES    = 10
TRIPLE_FORMAT = "plain"        # "plain" | "separators"

BATCH_SIZE   = 8
EVAL_BATCH   = 16
GRAD_ACCUM   = 4
LR           = 5e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS   = 5
WARMUP_RATIO = 0.1
EARLY_STOP_PATIENCE = 3
SEED         = 42

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
CACHE_DIR  = os.path.join(WORK, "cache");   os.makedirs(CACHE_DIR, exist_ok=True)
OUTPUT_DIR = os.path.join(WORK, "outputs"); os.makedirs(OUTPUT_DIR, exist_ok=True)
print("WORK dir:", WORK)

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch; torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    except Exception: pass
set_seed()
print("Config loaded. LLM_MODEL =", LLM_MODEL)''')

# ---------------------------------------------------------------------------
md(r"""## 3. NVIDIA API client(s) + robust LLM call with smart key rotation

One client per key in `NVIDIA_API_KEYS`. The rotation now classifies errors:

- **401/403 (auth/forbidden)** → that key is **permanently disabled** (a 403
  means the key is invalid or not authorized — no point retrying it).
- **429 (rate limit)** → **exponential backoff** before retrying (free NVIDIA
  tiers rate-limit aggressively, so keep `LLM_MAX_WORKERS` small).
- other errors → short sleep + rotate.

On total failure `call_llm` **raises** (instead of returning a degraded
answer), so the claim is left un-cached and retried on the next run rather than
saved with an empty pool.
""")
code(r'''import threading
from openai import OpenAI

clients = [OpenAI(base_url=NVIDIA_BASE_URL, api_key=k) for k in NVIDIA_API_KEYS]
_N = len(clients)
_client_lock = threading.Lock()
_cur = {"i": 0}
_disabled = [False] * _N            # keys killed by auth errors (401/403)

class LLMUnavailable(Exception):
    pass

def _status_code(e):
    c = getattr(e, "status_code", None)
    if c: return int(c)
    m = re.search(r"(?:Error code|'status'):\s*(\d+)", str(e))
    return int(m.group(1)) if m else None

def _advance():
    _cur["i"] = (_cur["i"] + 1) % _N

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

def _clean(text: str) -> str:
    if not text: return ""
    text = _THINK_RE.sub("", text)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()

def call_llm(prompt: str, sweeps: int = 4) -> str:
    """Return the model's answer (thinking stripped). Rotate/backoff on errors;
    raise LLMUnavailable if every key fails (so the caller can retry later)."""
    sys_msg = "You are a helpful assistant."
    user_msg = prompt + ("\n/no_think" if DISABLE_THINKING else "")
    backoff = 2.0
    last_err = None
    for _ in range(max(1, sweeps) * _N):
        with _client_lock:
            # skip disabled keys
            tries = 0
            while _disabled[_cur["i"]] and tries < _N:
                _advance(); tries += 1
            if _disabled[_cur["i"]]:
                raise LLMUnavailable("all API keys disabled (auth failed)")
            idx = _cur["i"]
        try:
            resp = clients[idx].chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "system", "content": sys_msg},
                          {"role": "user", "content": user_msg}],
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )
            return _clean(resp.choices[0].message.content)
        except Exception as e:
            last_err = e
            code = _status_code(e)
            with _client_lock:
                if code in (401, 403):
                    _disabled[idx] = True
                    print(f"[call_llm] key#{idx} DISABLED (HTTP {code} auth/forbidden)")
                    _advance()
                elif code == 429:
                    print(f"[call_llm] key#{idx} 429 rate-limited; backoff {backoff:.0f}s")
                    _advance()
                else:
                    print(f"[call_llm] key#{idx} error: {e}")
                    _advance()
            if code == 429:
                time.sleep(backoff); backoff = min(backoff * 2, 60)
            else:
                time.sleep(1)
    raise LLMUnavailable(f"all keys failed; last error: {last_err}")

print(f"{_N} API client(s) ready.")
# Quick connectivity smoke test (comment out to save a call)
try:
    print("LLM test:", repr(call_llm("Reply with the single word: ok")[:50]))
except Exception as e:
    print("LLM test failed:", e)''')

# ---------------------------------------------------------------------------
md("## 4. Locate & load data (FactKG + DBpedia)")
code(r'''def find_file(name):
    """Find a file by exact name under any of DATA_DIRS (first match wins)."""
    for root in DATA_DIRS:
        if not os.path.isdir(root):
            continue
        hits = glob.glob(os.path.join(root, "**", name), recursive=True)
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"Could not find {name}. Searched: {DATA_DIRS}. "
        f"Adjust DATA_DIRS in the Config cell to point at your KG-GPT data."
    )

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

FACTKG_PATHS = {
    "train": find_file("factkg_train.pickle"),
    "dev":   find_file("factkg_dev.pickle"),
    "test":  find_file("factkg_test.pickle"),
}
DBPEDIA_PATH   = find_file("dbpedia_2015_undirected_light.pickle")
RELATIONS_PATH = find_file("relations_for_final.pickle")
print("FactKG:", FACTKG_PATHS)
print("DBpedia:", DBPEDIA_PATH)
print("Relations:", RELATIONS_PATH)

print("\nLoading DBpedia (~1.5GB, may take a minute)...")
KG = load_pickle(DBPEDIA_PATH)
print("DBpedia entities:", len(KG))''')

# ---------------------------------------------------------------------------
md("## 5. Build `type_dict` (once, cached)\nThe KG-GPT data has no prebuilt `type_dict.pickle`; we build it from DBpedia + the relations list (porting `kg-gpt/data/make_type_dict.py`).")
code(r'''from tqdm.auto import tqdm

TYPE_DICT_PATH = os.path.join(CACHE_DIR, "type_dict.pickle")

def build_type_dict(dbp, relations_path):
    relations = load_pickle(relations_path)
    relation_set = {}
    for rel in relations:
        if "~" not in rel[0]:
            relation_set[rel] = 0

    total = {}
    for ent in tqdm(list(dbp), desc="build type_dict"):
        try:
            bpl = dbp[ent]
            rels = list(bpl)
            head_types = bpl["22-rdf-syntax-ns#type"]
        except Exception:
            continue
        if len(rels) == 1 and "rdf-schema#label" in rels:
            continue
        for rel in rels:
            for tail in bpl[rel]:
                if '"' in tail:
                    continue
                try:
                    tail_types = dbp[tail]["22-rdf-syntax-ns#type"]
                    _ = relation_set[rel]
                except Exception:
                    continue
                for ht in head_types:
                    for tt in tail_types:
                        try:
                            total[ht][tt][rel] = 0
                        except Exception:
                            try:
                                total[ht][tt] = {}; total[ht][tt][rel] = 0
                            except Exception:
                                total[ht] = {}; total[ht][tt] = {}; total[ht][tt][rel] = 0
    return total

if os.path.exists(TYPE_DICT_PATH):
    print("Loading cached type_dict...")
    type_dict = load_pickle(TYPE_DICT_PATH)
else:
    type_dict = build_type_dict(KG, RELATIONS_PATH)
    with open(TYPE_DICT_PATH, "wb") as f:
        pickle.dump(type_dict, f)
print("type_dict head-types:", len(type_dict))''')

# ---------------------------------------------------------------------------
md("## 6. KG-GPT helper functions + prompts\nCopied verbatim from `kg-gpt/factkg_test.py` (parsing, Algorithm 1, `graph_extractor`). The retrieval parser is made robust to extra brackets from chatty LLMs.")
code('SENTENCE_DIVIDE_PROMPT = ' + json.dumps(SENTENCE_DIVIDE_PROMPT) + '\n\n'
     'RELATION_RETRIEVAL_PROMPT = ' + json.dumps(RELATION_RETRIEVAL_PROMPT) + '\n\n'
     'print("prompts embedded:", len(SENTENCE_DIVIDE_PROMPT), len(RELATION_RETRIEVAL_PROMPT))')

code(r'''def claim_divider_parse_answer(answer, gt_entities):
    processed_answer_set = {}
    answer = (answer or "").strip()
    splitted_answers = answer.split("\n")
    all_entities = []
    try:
        for nth_answer in splitted_answers:
            nth_answer = nth_answer.strip()
            for i in range(3):
                if str(i + 1) + ". " in nth_answer[:5]:
                    temp_ans = nth_answer.split(str(i + 1) + ". ")[1]
                    temp_split = temp_ans.split(", Entity set: ")
                    sentence = temp_split[0]
                    entity_set = temp_split[1]
                    entity_set = entity_set.split("[")[1]
                    entity_set = entity_set.split("]")[0]
                    entity_set = entity_set.split(" ## ")
                    new_entity_set = []
                    for ent in entity_set:
                        new_entity_set.append(ent[1:-1]); all_entities.append(ent[1:-1])
                    break
            processed_answer_set[sentence] = new_entity_set
    except Exception:
        return None
    return processed_answer_set


def relation_candidates(KG, type_dict, entity_set):
    final_type_set = []; final_entity_set = []; new_entity_set = []
    for ent in entity_set:
        if '"' in ent:
            final_entity_set.append(ent); new_entity_set.append(ent); continue
        total = []; ent = ent.strip()
        splitted_ent = ent.split(" ")
        if len(splitted_ent) == 1:
            splitted_ent = ent.split("_")
        for spl_ent in splitted_ent:
            total.append([spl_ent.strip(), spl_ent.strip()[0].upper() + spl_ent.strip()[1:]])
        temp_list = []
        for chunk in total:
            if len(temp_list) == 0:
                temp_list = [chunk[0], chunk[1]]; continue
            new_list = []
            for temp in temp_list:
                new_list.append(temp + chunk[0]); new_list.append(temp + chunk[1])
            temp_list = new_list.copy()
        is_type = 0
        for type_ in temp_list:
            try:
                _ = type_dict[type_]; final_type_set.append([type_])
                new_entity_set.append(type_); is_type = 1; break
            except Exception:
                continue
        if is_type == 0:
            final_entity_set.append(ent); new_entity_set.append(ent)

    all_type_list = []
    if len(final_type_set) == 1:
        for temp_type in final_type_set[0]:
            for k in list(type_dict[temp_type]):
                all_type_list += type_dict[temp_type][k]
        all_type_list = list(set(all_type_list))
    else:
        for fs in final_type_set:
            tmp = []
            for ofs in final_type_set:
                if fs != ofs:
                    for fe in fs:
                        for oe in ofs:
                            if len(tmp) == 0:
                                try: tmp = type_dict[fe][oe]
                                except Exception: tmp = []
                            else:
                                try: tmp = [t for t in tmp if t in type_dict[fe][oe]]
                                except Exception: tmp = []
            all_type_list += tmp.copy()
        all_type_list = list(set(all_type_list))

    all_entity_list = []
    if len(final_entity_set) == 1:
        try: all_entity_list = list(KG[final_entity_set[0]])
        except Exception: all_entity_list = []
    else:
        for fe in final_entity_set:
            for ofe in final_entity_set:
                if fe != ofe:
                    try:
                        for temp_rel in list(KG[fe]):
                            try:
                                other = list(KG[ofe])
                                if "~" in temp_rel[0]:
                                    if temp_rel.split("~")[1] in other:
                                        all_entity_list.append(temp_rel.split("~")[1])
                                    elif "~" + temp_rel in other:
                                        all_entity_list.append(temp_rel)
                            except Exception: pass
                    except Exception: pass
        all_entity_list = list(set(all_entity_list))

    final_relation_list = []
    if len(all_type_list) == 0:
        for rel in all_entity_list:
            final_relation_list.append(rel.split("~")[1] if "~" in rel[0] else rel)
    elif len(all_entity_list) == 0:
        for rel in all_type_list:
            final_relation_list.append(rel.split("~")[1] if "~" in rel[0] else rel)
    else:
        for rel in all_entity_list:
            if "~" in rel[0]:
                if rel.split("~")[1] in all_type_list:
                    final_relation_list.append(rel.split("~")[1])
            elif rel in all_type_list or "~" + rel in all_type_list or len(all_type_list) == 0:
                final_relation_list.append(rel)
    return final_relation_list, new_entity_set


def retrieval_relation_parse_answer(answer):
    # Robust to chatty output: take the LAST bracketed list found.
    matches = re.findall(r"\[[^\]]+\]", answer or "")
    if not matches:
        return None
    match = matches[-1]
    comps = [c.strip() for c in match.strip("[]").split(",")]
    comps = [c.strip("''") if "'" in c else c for c in comps]
    comps = [c.strip().strip('"') for c in comps if c.strip()]
    return comps or None


def graph_extractor(target_list):
    if len(target_list) == 0:
        return target_list
    return_list = []; filter_dict = {"head": {}, "tail": {}}
    return_list.append(target_list[0])
    used_heads = []; used_tails = []
    filter_dict["head"][target_list[0][0]] = [target_list[0][1]]
    filter_dict["tail"][target_list[0][2]] = [target_list[0][1]]
    used_heads.append(target_list[0][0]); used_tails.append(target_list[0][2])
    for tar in target_list:
        if tar in return_list:
            continue
        try:
            if tar[1] in filter_dict["head"][tar[0]]:
                continue
        except Exception:
            try:
                if tar[1] in filter_dict["tail"][tar[2]]:
                    continue
            except Exception:
                pass
        try:
            if tar[1] not in list(filter_dict["head"][tar[0]]):
                filter_dict["head"][tar[0]] = [tar[1]]
                try: filter_dict["tail"][tar[2]].append(tar[1])
                except Exception: filter_dict["tail"][tar[2]] = [tar[1]]
                return_list.append(tar); continue
        except Exception:
            pass
        try:
            if tar[1] not in list(filter_dict["tail"][tar[2]]):
                filter_dict["tail"][tar[2]] = [tar[1]]
                try: filter_dict["head"][tar[0]].append(tar[1])
                except Exception: filter_dict["head"][tar[0]] = [tar[1]]
                return_list.append(tar); continue
        except Exception:
            pass
        if tar[2] in used_heads or tar[0] in used_tails:
            return_list.append(tar)
            try: filter_dict["head"][tar[0]].append(tar[1])
            except Exception: filter_dict["head"][tar[0]] = [tar[1]]
            try: filter_dict["tail"][tar[2]].append(tar[1])
            except Exception: filter_dict["tail"][tar[2]] = [tar[1]]
    return return_list

print("KG-GPT helpers ready.")''')

# ---------------------------------------------------------------------------
md("## 7. Stage 1+2 — adapter (LLM retrieval) + triple pool builder\nFaithful to KG-GPT, including multi-hop / type-entity / cross-sub-claim bridging.")
code(r'''def sentence_divide(claim, gt_entities):
    q = (SENTENCE_DIVIDE_PROMPT
         .replace("<<<<CLAIM>>>>", claim)
         .replace("<<<<ENTITY_SET>>>>", str(gt_entities)))
    for _ in range(3):
        out = call_llm(q)
        parsed = claim_divider_parse_answer(out, gt_entities)
        if parsed:
            return parsed
    return {claim: list(gt_entities)}   # fallback: whole claim as one sub-sentence


def top_k_relations(sub_sentence, candidates):
    if len(candidates) <= TOP_K:
        return list(candidates)
    q = (RELATION_RETRIEVAL_PROMPT
         .replace("<<<<TOP_K>>>>", str(TOP_K))
         .replace("<<<<SENTENCE>>>>", sub_sentence)
         .replace("<<<<RELATION_SET>>>>", str(candidates)))
    for _ in range(3):
        out = call_llm(q)
        picked = retrieval_relation_parse_answer(out)
        if picked:
            kept = [r for r in picked if r in candidates]
            return kept or list(candidates[:TOP_K])
    return list(candidates[:TOP_K])


def build_subclaim_triples(relations, entity_set):
    out = []
    for rel in relations:
        if len(entity_set) == 1:
            out.append([entity_set[0], rel]); out.append([entity_set[0], "~" + rel])
        for a in range(len(entity_set)):
            for b in range(len(entity_set)):
                if a != b:
                    out.append([entity_set[a], rel, entity_set[b]])
                    out.append([entity_set[a], "~" + rel, entity_set[b]])
    return out


def build_triple_pool(total_evidence, KG, type_dict, gt_entities, max_triples=MAX_TRIPLES):
    additional = []
    for evi in total_evidence:
        try:
            _ = type_dict[evi[0][0]]; _ = type_dict[evi[0][2]]
            for trip in evi: additional.append(trip[1])
        except Exception:
            continue

    before_final = []
    for evi in total_evidence:
        cur = []
        for trip in evi:
            try:
                _ = type_dict[trip[0]]; continue            # head is a type -> skip
            except Exception:
                try:
                    _ = type_dict[trip[2]]                    # tail is a type -> expand
                    try:
                        for tail in KG[trip[0]][trip[1]]:
                            cur.append([trip[0], trip[1], tail])
                    except Exception:
                        continue
                except Exception:
                    try:
                        if len(trip) == 2:
                            cur.append(trip)
                        elif trip[2] in KG[trip[0]][trip[1]]:
                            cur.append(trip)
                    except Exception:
                        pass
        if cur:
            before_final.append(cur)

    final_evidence = []
    for chunk in before_final:
        find_gt = 0
        for trip in chunk:
            if len(trip) != 2 and trip[0] in gt_entities and trip[2] in gt_entities:
                final_evidence.append(trip); find_gt = 1
        if find_gt == 1:
            continue
        if len(before_final) == 1:
            for trip in chunk:
                if len(trip) == 2:
                    try:
                        for tail in KG[trip[0]][trip[1]]:
                            final_evidence.append([trip[0], trip[1], tail])
                    except Exception:
                        continue
            break
        additional = list(set(additional))
        if len(additional) != 0:
            for sec in before_final:
                if chunk == sec: continue
                for trip in chunk:
                    if len(trip) == 2: continue
                    for st in sec:
                        if len(st) == 2: continue
                        for rel_ in additional:
                            for ti in [0, 2]:
                                for si in [0, 2]:
                                    for add in ["", "~"]:
                                        try:
                                            if add == "" and "~" in rel_:
                                                if trip[ti] in KG[st[si]][rel_.split("~")[1]]:
                                                    final_evidence += [trip, st, [st[si], rel_.split("~")[1], trip[ti]]]
                                        except Exception: pass
                                        try:
                                            if trip[ti] in KG[st[si]][add + rel_]:
                                                final_evidence += [trip, st, [st[si], add + rel_, trip[ti]]]
                                        except Exception: pass
        else:
            for sec in before_final:
                if chunk == sec: continue
                for trip in chunk:
                    for st in sec:
                        if len(trip) == 2 or len(st) == 2: continue
                        if (trip[0] in st and trip[0] not in gt_entities) or (trip[2] in st and trip[2] not in gt_entities):
                            final_evidence += [trip, st]

    new_final = []
    for trip in final_evidence:
        if "~" in trip[1]:
            flipped = [trip[2], trip[1].split("~")[1], trip[0]]
            if flipped not in new_final and flipped not in final_evidence:
                new_final.append(flipped); continue
        else:
            if trip not in new_final:
                new_final.append(trip)

    pruned = graph_extractor(new_final)
    pool, seen = [], set()
    for trip in pruned:
        if len(trip) >= 3:
            t = (trip[0], trip[1], trip[2])
            if t not in seen:
                seen.add(t); pool.append(t)
        if len(pool) >= max_triples:
            break
    return pool


def process_claim(claim, entity_set):
    divided = sentence_divide(claim, entity_set)

    # Leverage FactKG's gold Entity_set: the prompt already tells the LLM to use
    # only the given entities, but smaller models add junk (e.g. the pronoun
    # "he"). Keep only entities that are in entity_set (case-insensitive, mapped
    # back to the canonical KG key form). If a sub-sentence keeps nothing, drop
    # it; if everything is dropped, fall back to the whole claim + gold entities.
    _gold_lc = {e.lower(): e for e in entity_set}
    _cleaned = {}
    for _sub, _ents in divided.items():
        _kept = []
        for _e in _ents:
            if _e in entity_set:
                _kept.append(_e)
            elif _e.lower() in _gold_lc:
                _kept.append(_gold_lc[_e.lower()])   # canonicalize minor case diffs
        if _kept:
            _cleaned[_sub] = _kept
    divided = _cleaned if _cleaned else {claim: list(entity_set)}

    sub_data, total_evidence = [], []
    for sub_text, sub_entities in divided.items():
        try:
            cand, norm_ents = relation_candidates(KG, type_dict, sub_entities)
        except Exception:
            cand, norm_ents = [], list(sub_entities)
        if len(cand) == 0:
            chosen = []
        elif len(cand) < TOP_K:
            chosen = list(cand)
        else:
            chosen = top_k_relations(sub_text, cand)
        sub_data.append({"text": sub_text, "entities": list(norm_ents), "top_k_relations": list(chosen)})
        st = build_subclaim_triples(chosen, list(norm_ents))
        if st: total_evidence.append(st)
    pool = build_triple_pool(total_evidence, KG, type_dict, list(entity_set))
    return {"claim": claim, "entity_set": list(entity_set),
            "sub_sentences": sub_data, "triple_pool": pool}

print("Stage 1+2 ready.")''')

# ---------------------------------------------------------------------------
md(r"""## 8. Run Stage 1+2 over each split (concurrent + **RESUMABLE**)

Each completed claim is appended immediately to a `stage12_{split}.partial.jsonl`
log, so the run **survives interruptions**: re-running this cell loads what was
already processed and continues only the remaining claims. When a split
finishes it is consolidated into `stage12_{split}.pkl` (the file Stage 3 reads).

`limit`: `>0` = first N claims · `0` = skip · `<0` = the **entire** split.

> Full FactKG train (~86k claims × 2-3 API calls) is a long, paid run. Raise
> `LLM_MAX_WORKERS` (if your NVIDIA quota allows) to speed it up; it is safe to
> stop and re-run this cell to resume.
""")
code(r'''import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

def _partial_path(split): return os.path.join(CACHE_DIR, f"stage12_{split}.partial.jsonl")
def _final_path(split):   return os.path.join(CACHE_DIR, f"stage12_{split}.pkl")

def _load_done(split):
    """qid -> record, seeded from a finished .pkl and/or the partial JSONL log."""
    done = {}
    fp = _final_path(split)
    if os.path.exists(fp):
        try:
            for r in load_pickle(fp):
                done[r["qid"]] = r
        except Exception:
            pass
    pp = _partial_path(split)
    if os.path.exists(pp):
        with open(pp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line); done[r["qid"]] = r
                except Exception:
                    continue   # skip a half-written final line
    return done

def run_stage12(split, limit):
    if limit == 0:
        print(f"[{split}] skipped (limit=0)")
        return None

    factkg = load_pickle(FACTKG_PATHS[split])
    items = list(factkg.items())
    if limit and limit > 0:
        items = items[:limit]

    done = _load_done(split)
    todo = [(i, it) for i, it in enumerate(items) if i not in done]
    print(f"[{split}] total={len(items)} done={len(done)} todo={len(todo)}")

    def work(args):
        qid, (claim_text, cdata) = args
        try:
            out = process_claim(claim_text, cdata.get("Entity_set", []))
            lab = cdata.get("Label", [None])
            out["qid"] = qid
            out["label"] = lab[0] if isinstance(lab, (list, tuple)) and lab else lab
            out["reasoning_types"] = cdata.get("types", [])
            return out
        except Exception as e:
            print(f"[{split}] qid={qid} error: {e}")
            return None

    if todo:
        lock = threading.Lock()
        with open(_partial_path(split), "a", encoding="utf-8") as jf, \
             ThreadPoolExecutor(max_workers=LLM_MAX_WORKERS) as ex:
            futs = [ex.submit(work, t) for t in todo]
            for fut in tqdm(as_completed(futs), total=len(futs), desc=f"Stage12::{split}"):
                r = fut.result()
                if r is None:
                    continue
                with lock:                              # durable per-claim checkpoint
                    jf.write(json.dumps(r, ensure_ascii=False) + "\n"); jf.flush()
                    done[r["qid"]] = r

    records = [done[q] for q in sorted(done)]
    with open(_final_path(split), "wb") as f:
        pickle.dump(records, f)

    sizes = [len(r["triple_pool"]) for r in records]
    empty = sum(1 for s in sizes if s == 0)
    avg = (sum(sizes) / len(sizes)) if sizes else 0
    print(f"[{split}] saved {len(records)} -> {_final_path(split)} | "
          f"avg pool={avg:.2f}, empty={empty} ({100*empty/max(len(records),1):.1f}%)")
    return _final_path(split)

train_cache = run_stage12("train", TRAIN_LIMIT)
dev_cache   = run_stage12("dev",   DEV_LIMIT)
test_cache  = run_stage12("test",  TEST_LIMIT)''')

# ---------------------------------------------------------------------------
md(r"""## 8b. (Optional) Re-process only the empty-pool claims

Keeps the cache and re-runs `process_claim` **only** on records whose
`triple_pool` is empty (e.g. after improving entity handling). Records that
fill up are updated in place; those still empty are marked `_tried_fix` so a
re-run won't redo them. The resume log is kept in sync so a later
`run_stage12` won't revert the fixes.

Run this AFTER the helper/adapter cells are loaded (so the current
`process_claim` is in scope). To force-retry genuine empties later (e.g. with a
bigger model), clear the flag: `for r in records: r.pop("_tried_fix", None)`.
""")
code(r'''import json
from concurrent.futures import ThreadPoolExecutor, as_completed

def fix_empty_pools(cache_path, save_every=2000):
    if not os.path.exists(cache_path):
        print(cache_path, "-> not found, skip"); return
    with open(cache_path, "rb") as f:
        records = pickle.load(f)
    todo = [i for i, r in enumerate(records)
            if len(r["triple_pool"]) == 0 and not r.get("_tried_fix")]
    print(f"{os.path.basename(cache_path)}: empty&untried={len(todo)} / total={len(records)}")
    if not todo:
        print("  nothing to do."); return

    def work(i):
        r = records[i]
        try:
            out = process_claim(r["claim"], r["entity_set"])
            return i, out["triple_pool"], out["sub_sentences"]
        except Exception as e:
            print(f"  qid={r.get('qid')} error: {e}")
            return i, [], None

    def _save():
        with open(cache_path, "wb") as f:
            pickle.dump(records, f)
        pp = cache_path.replace(".pkl", ".partial.jsonl")     # keep resume log in sync
        with open(pp, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    done = filled = 0
    with ThreadPoolExecutor(max_workers=LLM_MAX_WORKERS) as ex:
        futs = [ex.submit(work, i) for i in todo]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="fix empties"):
            i, pool, subs = fut.result()
            records[i]["triple_pool"] = pool
            if subs is not None:
                records[i]["sub_sentences"] = subs
            records[i]["_tried_fix"] = True
            done += 1
            if len(pool) > 0: filled += 1
            if done % save_every == 0: _save()
    _save()
    still = sum(1 for r in records if len(r["triple_pool"]) == 0)
    print(f"  newly filled={filled}, still empty={still} ({100*still/len(records):.1f}%)")

# Run on whichever caches exist (test first to gauge, then train).
for _split in ["test", "dev", "train"]:
    fix_empty_pools(os.path.join(CACHE_DIR, f"stage12_{_split}.pkl"))''')

# ---------------------------------------------------------------------------
md("## 9. Stage 3 — graph builder + dataset")
code(r'''import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup

NULL_TRIPLE_TEXT = "no_triple no_relation no_triple"

def format_triple(h, r, t, mode=TRIPLE_FORMAT):
    if mode == "separators":
        return f"[H] {h} [R] {r} [T] {t}"
    return f"{h} {r} {t}"

def build_graph(claim, triple_pool, max_nodes=MAX_NODES, mode=TRIPLE_FORMAT):
    nodes = []
    for trip in triple_pool[:max_nodes]:
        h, r, t = trip
        nodes.append({"triple_text": format_triple(h, r, t, mode), "is_null": False})
    while len(nodes) < max_nodes:
        nodes.append({"triple_text": NULL_TRIPLE_TEXT, "is_null": True})
    return nodes

def _label_to_int(lab):
    if isinstance(lab, (list, tuple)): lab = lab[0] if lab else None
    if isinstance(lab, str): return 1 if lab.strip().lower() in ("true","1","supported") else 0
    return 1 if bool(lab) else 0

class FactKGGraphDataset(Dataset):
    def __init__(self, cache_path, tokenizer, max_seq_len=MAX_SEQ_LEN,
                 max_nodes=MAX_NODES, mode=TRIPLE_FORMAT):
        with open(cache_path, "rb") as f:
            records = pickle.load(f)
        if isinstance(records, dict):
            records = [{"claim": k, **v} for k, v in records.items()]
        n = len(records)
        vocab = getattr(tokenizer, "vocab_size", 0) or 0
        id_dtype = np.uint16 if vocab and vocab <= 65535 else np.int32
        self._ids  = np.zeros((n, max_nodes, max_seq_len), dtype=id_dtype)
        self._mask = np.zeros((n, max_nodes, max_seq_len), dtype=np.uint8)
        self._seg  = np.zeros((n, max_nodes, max_seq_len), dtype=np.uint8)
        self._null = np.zeros((n, max_nodes), dtype=np.bool_)
        self._labels = np.zeros((n,), dtype=np.int64)
        self.claims, self.reasoning_types = [], []
        for i, item in enumerate(tqdm(records, desc=f"tokenize {os.path.basename(cache_path)}")):
            claim_text = item["claim"]
            for k, node in enumerate(build_graph(claim_text, item["triple_pool"], max_nodes, mode)):
                enc = tokenizer(claim_text, node["triple_text"], max_length=max_seq_len,
                                padding="max_length", truncation=True, return_token_type_ids=True)
                self._ids[i, k]  = enc["input_ids"]
                self._mask[i, k] = enc["attention_mask"]
                self._seg[i, k]  = enc["token_type_ids"]
                self._null[i, k] = node["is_null"]
            self._labels[i] = _label_to_int(item.get("label"))
            self.claims.append(claim_text)
            self.reasoning_types.append(item.get("reasoning_types", []))
    def __len__(self): return len(self.claims)
    def __getitem__(self, idx):
        return {
            "input_ids": torch.from_numpy(self._ids[idx].astype(np.int64)),
            "attention_mask": torch.from_numpy(self._mask[idx].astype(np.int64)),
            "token_type_ids": torch.from_numpy(self._seg[idx].astype(np.int64)),
            "is_null": torch.from_numpy(self._null[idx].copy()),
            "label": torch.tensor(int(self._labels[idx]), dtype=torch.long),
            "claim": self.claims[idx], "reasoning_types": self.reasoning_types[idx],
        }

def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "token_type_ids": torch.stack([b["token_type_ids"] for b in batch]),
        "is_null": torch.stack([b["is_null"] for b in batch]),
        "labels": torch.stack([b["label"] for b in batch]),
        "claims": [b["claim"] for b in batch],
        "reasoning_types": [b["reasoning_types"] for b in batch],
    }
print("Stage 3 data ready.")''')

# ---------------------------------------------------------------------------
md("## 10. Stage 3 — model (KernelGAT) + fair baseline")
code(r'''import torch.nn as nn
import torch.nn.functional as F

def kernel_mus(n):
    l = [1.0]
    if n == 1: return l
    b = 2.0/(n-1); l.append(1-b/2)
    for i in range(1, n-1): l.append(l[i]-b)
    return l

def kernel_sigmas(n):
    l = [0.001]
    if n == 1: return l
    return l + [0.1]*(n-1)

class KernelKGGPT(nn.Module):
    def __init__(self, bert_model=BERT_MODEL, num_kernels=NUM_KERNELS, num_labels=NUM_LABELS,
                 max_nodes=MAX_NODES, max_seq_len=MAX_SEQ_LEN, dropout=DROPOUT):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.H = self.bert.config.hidden_size
        self.K = num_kernels; self.dropout = nn.Dropout(dropout)
        mu = torch.FloatTensor(kernel_mus(num_kernels)).view(1,1,1,num_kernels)
        sg = torch.FloatTensor(kernel_sigmas(num_kernels)).view(1,1,1,num_kernels)
        self.register_buffer("mu", mu); self.register_buffer("sigma", sg)
        self.proj_select = nn.Linear(num_kernels, 1)
        self.proj_att = nn.Linear(num_kernels, 1)
        self.proj_inference_de = nn.Linear(self.H*2, num_labels)
        self.proj_gat = nn.Sequential(nn.Linear(self.H*2, 128), nn.ReLU(True), nn.Linear(128, 1))

    def _pool_node(self, q, d, aq, ad):
        aq = aq.unsqueeze(-1); ad = ad.unsqueeze(1).unsqueeze(-1)
        sim = torch.bmm(q, d.transpose(1,2)).unsqueeze(-1)
        pv = torch.exp(-((sim-self.mu)**2)/(self.sigma**2)/2)*ad
        ps = pv.sum(2)
        ls = torch.log(torch.clamp(ps, min=1e-10))*aq
        ls = ls.sum(1)/(aq.sum(1)+1e-10)
        return self.proj_select(ls)

    def _pool_token(self, q, d, aq, ad):
        ad = ad.unsqueeze(1).unsqueeze(-1)
        sim = torch.bmm(q, d.transpose(1,2)).unsqueeze(-1)
        pv = torch.exp(-((sim-self.mu)**2)/(self.sigma**2)/2)*ad
        ls = torch.log(torch.clamp(pv.sum(2), min=1e-10))
        ls = self.proj_att(ls).squeeze(-1)
        ls = ls.masked_fill((1-aq).bool(), -1e4)
        return F.softmax(ls, dim=1)

    def _self_attention(self, inputs, hiddens, mask_text, idx, is_null):
        B,N,L,H = hiddens.shape
        own_h = hiddens[:, idx:idx+1].expand(-1,N,-1,-1)
        own_m = mask_text[:, idx:idx+1].expand(-1,N,-1)
        own_i = inputs[:, idx:idx+1].expand(-1,N,-1)
        on = F.normalize(own_h, p=2, dim=-1); en = F.normalize(hiddens, p=2, dim=-1)
        att = self._pool_token(en.reshape(-1,L,H), on.reshape(-1,L,H),
                               mask_text.reshape(-1,L), own_m.reshape(-1,L))
        att = att.view(B,N,L,1)
        denoise = (att*hiddens).sum(dim=2)
        s = self.proj_gat(torch.cat([own_i, denoise], dim=-1))
        if is_null is not None:
            s = s.masked_fill(is_null.unsqueeze(-1), -1e4)
        w = F.softmax(s, dim=1)
        return (denoise*w).sum(dim=1)

    def forward(self, input_ids, attention_mask, token_type_ids, is_null):
        B,N,L = input_ids.shape
        out = self.bert(input_ids=input_ids.view(-1,L),
                        attention_mask=attention_mask.view(-1,L),
                        token_type_ids=token_type_ids.view(-1,L))
        hidden = self.dropout(out.last_hidden_state); pooled = out.pooler_output; H = hidden.size(-1)
        mt = attention_mask.view(-1,L).float().clone(); mt[:,0] = 0.0
        mc = (1.0 - token_type_ids.view(-1,L).float())*mt
        me = token_type_ids.view(-1,L).float()*mt
        hn = F.normalize(hidden, p=2, dim=2)
        ns = self._pool_node(hn, hn, mc, me).view(B,N,1).masked_fill(is_null.unsqueeze(-1), -1e4)
        select_prob = F.softmax(ns, dim=1)
        inputs = pooled.view(B,N,H); hiddens = hidden.view(B,N,L,H); mt3 = mt.view(B,N,L)
        de = [self._self_attention(inputs, hiddens, mt3, i, is_null) for i in range(N)]
        de = torch.stack(de, dim=1)
        feat = torch.cat([inputs, de], dim=-1)
        per_node = F.softmax(self.proj_inference_de(feat), dim=-1)
        agg = torch.clamp((select_prob*per_node).sum(dim=1), min=1e-10)
        return {"logits": torch.log(agg), "node_probs": select_prob.squeeze(-1)}

class BertConcatBaseline(nn.Module):
    def __init__(self, bert_model=BERT_MODEL, num_labels=NUM_LABELS, dropout=DROPOUT, **kw):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.H = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.H, num_labels)
    def forward(self, input_ids, attention_mask, token_type_ids, is_null):
        B,N,L = input_ids.shape
        out = self.bert(input_ids=input_ids.view(-1,L),
                        attention_mask=attention_mask.view(-1,L),
                        token_type_ids=token_type_ids.view(-1,L))
        cls = out.pooler_output.view(B,N,self.H)
        valid = (~is_null).float().unsqueeze(-1)
        pooled = (cls*valid).sum(1)/valid.sum(1).clamp(min=1.0)
        return {"logits": F.log_softmax(self.classifier(self.dropout(pooled)), dim=-1), "node_probs": None}

def build_model():
    if MODEL_TYPE == "kernel": return KernelKGGPT()
    if MODEL_TYPE == "concat_baseline": return BertConcatBaseline()
    raise ValueError(MODEL_TYPE)

print("Stage 3 model ready.")''')

# ---------------------------------------------------------------------------
md("## 11. Train + evaluate")
code(r'''from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

train_ds = FactKGGraphDataset(train_cache, tokenizer)
dev_ds   = FactKGGraphDataset(dev_cache,   tokenizer)
test_ds  = FactKGGraphDataset(test_cache,  tokenizer)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
dev_loader   = DataLoader(dev_ds,   batch_size=EVAL_BATCH, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(test_ds,  batch_size=EVAL_BATCH, shuffle=False, collate_fn=collate_fn)

@torch.no_grad()
def evaluate(model, loader):
    model.eval(); correct=total=0; stats=defaultdict(lambda:{"c":0,"t":0})
    for batch in loader:
        out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                    batch["token_type_ids"].to(device), batch["is_null"].to(device))
        preds = out["logits"].argmax(-1).cpu(); labels = batch["labels"]
        correct += (preds==labels).sum().item(); total += labels.size(0)
        for i, types in enumerate(batch["reasoning_types"]):
            hit = int(preds[i].item()==labels[i].item())
            for t in types: stats[t]["t"]+=1; stats[t]["c"]+=hit
    return correct/max(total,1), stats

def train():
    set_seed()
    model = build_model().to(device)
    print("model_type:", MODEL_TYPE)
    criterion = nn.NLLLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = (len(train_loader)*NUM_EPOCHS)//max(1,GRAD_ACCUM)
    sched = get_linear_schedule_with_warmup(optim, int(total_steps*WARMUP_RATIO), total_steps)
    best, patience = 0.0, 0
    for epoch in range(NUM_EPOCHS):
        model.train(); optim.zero_grad(); running=0.0
        for step, batch in enumerate(tqdm(train_loader, desc=f"epoch {epoch+1}")):
            out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                        batch["token_type_ids"].to(device), batch["is_null"].to(device))
            loss = criterion(out["logits"], batch["labels"].to(device))
            (loss/GRAD_ACCUM).backward(); running += loss.item()
            if (step+1)%GRAD_ACCUM==0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step(); sched.step(); optim.zero_grad()
        acc, stats = evaluate(model, dev_loader)
        print(f"[epoch {epoch+1}] train_loss={running/len(train_loader):.4f}  dev_acc={acc:.4f}")
        for t,s in sorted(stats.items()):
            if s["t"]: print(f"    {t:22s}: {s['c']/s['t']:.4f} ({s['t']})")
        if acc>best:
            best=acc; patience=0
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best.pt"))
            print("    * saved best")
        else:
            patience+=1
            if patience>=EARLY_STOP_PATIENCE:
                print("Early stopping."); break
    print(f"Best dev acc: {best:.4f}")
    return model

model = train()''')

code(r'''# ---- Final evaluation on the test split ----
model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best.pt"), map_location=device))
acc, stats = evaluate(model, test_loader)
print("="*50)
print(f"TEST accuracy: {acc:.4f}")
print("Per reasoning type:")
for t,s in sorted(stats.items()):
    if s["t"]: print(f"  {t:22s}: {s['c']/s['t']:.4f} ({s['t']})")''')

# ---------------------------------------------------------------------------
md(r"""## 12. Notes & scaling

- **Bottleneck = Stage 1+2 (API)**. To scale up, raise `TRAIN_LIMIT` /
  `DEV_LIMIT` / `TEST_LIMIT` and `LLM_MAX_WORKERS` (watch your NVIDIA quota
  and rate limits). Stage-1+2 caches persist in `/kaggle/working/cache`, so
  you can re-train Stage 3 without re-calling the API.
- **Fair baseline**: set `MODEL_TYPE = "concat_baseline"` and re-run cells
  10-11 to get the supervised baseline on the *same* triples. Compare against
  `MODEL_TYPE = "kernel"` to isolate the KernelGAT architecture's contribution.
- **Empty triple pools** print a coverage warning at Stage 1+2. Those claims
  fall back to an all-NULL graph (the model predicts from prior).
- **Persisting results**: everything under `/kaggle/working` is saved as the
  notebook's output. Download `outputs/best.pt` and `cache/stage12_*.pkl` to
  reuse later.
- **Exact model id**: confirm `LLM_MODEL` matches the NVIDIA NIM id you intend
  to use (build.nvidia.com → the model's API code).
""")

# ===========================================================================
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "accelerator": "GPU",
    },
    "nbformat": 4, "nbformat_minor": 5,
}

out_path = os.path.join(HERE, "KernelKG_GPT_Kaggle.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("Wrote", out_path, "with", len(cells), "cells")
