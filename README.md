# KernelKG-GPT

Fair combination of two fact-verification pipelines on **FactKG**:

- **Stage 1 + 2** — reuses [KG-GPT](https://github.com/jiho283/KG-GPT) for
  claim segmentation, candidate relation extraction (Algorithm 1) and top-K
  relation ranking via GPT.
- **Stage 3** — replaces KG-GPT's rule-based `graph_extractor` + LLM verifier
  with the [KernelGAT](https://github.com/thunlp/KernelGAT) kernel reasoning
  architecture (Gaussian kernel pooling + sentence-level GAT) operating on
  `(claim, triple)` nodes.

**No extensions beyond the two papers** — no auxiliary supervision, no
edge-type embedding, no extra signals. The only deltas from a verbatim port
are mechanical:

1. Input nodes are `[CLS] claim [SEP] head rel tail [SEP]` instead of
   `[CLS] claim [SEP] wiki_title evidence [SEP]`.
2. Kernel mu/sigma are registered as torch buffers (the reference impl
   hard-codes `.cuda()` calls).
3. NULL padding nodes are masked in every softmax so a variable number of
   retrieved triples can be padded up to `max_nodes`.
4. `num_labels = 2` (FactKG True / False) instead of 3 (FEVER S/R/NEI).

## Layout

```
kernelkg-gpt/
├── configs/default.yaml
├── stage12/                  # KG-GPT adapter — Stage 1+2 wrapper
│   ├── adapter.py
│   └── triple_pool_builder.py
├── stage3/                   # KernelGAT-style verifier
│   ├── triple_formatter.py
│   ├── graph_builder.py
│   ├── data.py
│   ├── model.py
│   └── losses.py
├── scripts/
│   ├── 01_run_stage12.py
│   ├── 02_train_stage3.py
│   ├── 03_evaluate.py
│   └── 04_ablation.py
├── utils/io_utils.py
├── data/                     # FactKG + DBpedia (gitignored)
└── cache/                    # Stage 1+2 cached outputs (gitignored)
```

The KG-GPT repo is expected at `../kg-gpt/` (sibling folder).

## Setup

```bash
pip install -r requirements.txt
echo "sk-..." > openai_api_key.txt

# Required data under data/:
#   factkg_train.pickle  factkg_dev.pickle  factkg_test.pickle
#   dbpedia_2015_undirected_light.pickle
#   type_dict.pickle           # built by kg-gpt/data/make_type_dict.py
```

## Run

```bash
# 1) Cache Stage 1+2 outputs (one-time, hits the OpenAI API)
python scripts/01_run_stage12.py --split dev --limit 50    # smoke test
python scripts/01_run_stage12.py --split dev
python scripts/01_run_stage12.py --split test
python scripts/01_run_stage12.py --split train

# 2) Train Stage 3
python scripts/02_train_stage3.py --config configs/default.yaml

# 3) Evaluate
python scripts/03_evaluate.py --model_path outputs/exp1/best.pt

# 4) Ablation suite
python scripts/04_ablation.py
```

## What's reused vs adapted

| Component | Source | Status |
|---|---|---|
| Sentence-division prompt | `kg-gpt/prompts/sentence_divide_prompt.txt` | unchanged |
| Candidate-relation extraction (Algorithm 1) | `kg-gpt/factkg_test.py::relation_candidates` | imported as-is |
| Top-K relation prompt | `kg-gpt/prompts/relation_retrieval_prompt.txt` | unchanged |
| Triple pool builder | KG-GPT evidence construction (`get_answer` lines ~294-465) + `graph_extractor` | replicated faithfully: type-entity expansion, 3-hop `additional`, cross-sub-claim bridging, dedup. Returns a deterministic ordered triple list. |
| Graph builder | new | NULL padding for fixed batch shape |
| Kernel pooling, node kernel, GAT attention | KernelGAT `kgat/models.py` | reimplemented identical equations with buffers + NULL mask (self-attention uses full-text mask for both args, matching the reference) |
| Loss | KernelGAT NLL on log-probabilities | unchanged |

## Models

`model_type` in the config selects the Stage-3 model:

- `kernel` (default) — the KernelGAT-style verifier.
- `concat_baseline` — a fair supervised baseline that sees the *same* triple
  pool but mean-pools the per-node `[CLS]` vectors through a linear head (no
  kernel/GAT). Report this alongside the kernel model so any gain is
  attributable to the architecture rather than to supervised fine-tuning.

## Stage 1+2 cache format

`scripts/01_run_stage12.py` writes a **list** of records (keyed internally by
`qid`, not by claim text, so duplicate claim strings don't collide). It also
prints triple-pool coverage (average pool size and the fraction of empty
pools — empty pools fall back to an all-NULL graph at Stage 3).

## Ablation variants

| Variant | What it measures |
|---|---|
| `default` | Reference setting (max_nodes=10, 21 kernels) |
| `max_nodes_5` / `max_nodes_15` / `max_nodes_20` | Sensitivity to graph size |
| `format_separators` | `[H] {h} [R] {r} [T] {t}` instead of plain |
| `kernels_11` | Fewer kernels |

Run all: `python scripts/04_ablation.py`
