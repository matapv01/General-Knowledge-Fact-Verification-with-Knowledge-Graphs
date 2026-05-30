"""Run Stage 1 + 2 (LLM-based retrieval) over a FactKG split and cache the
triple pools to disk so Stage 3 can train/evaluate without re-calling GPT.

Usage:
    python scripts/01_run_stage12.py --split dev
    python scripts/01_run_stage12.py --split train
    python scripts/01_run_stage12.py --split test
"""

import argparse
import os
import sys

import openai
import torch
from tqdm import tqdm

# Make project root importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stage12 import Stage12Adapter  # noqa: E402
from utils import load_pickle, set_seed  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "dev", "test"], required=True)
    parser.add_argument("--factkg_path", default=None,
                        help="Path to factkg_{split}.pickle (default: data/factkg_{split}.pickle)")
    parser.add_argument("--dbpedia_path",
                        default="data/dbpedia_2015_undirected_light.pickle")
    parser.add_argument("--type_dict_path", default="data/type_dict.pickle")
    parser.add_argument("--openai_key_file", default="openai_api_key.txt")
    parser.add_argument("--model", default="gpt-3.5-turbo-0613")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_triples", type=int, default=30)
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N claims (debug)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)

    factkg_path = args.factkg_path or f"data/factkg_{args.split}.pickle"
    output_path = args.output or f"cache/stage12_{args.split}.pt"

    print(f"[01_run_stage12] Loading FactKG split from {factkg_path}")
    factkg = load_pickle(factkg_path)

    print(f"[01_run_stage12] Loading DBpedia KG from {args.dbpedia_path}")
    dbpedia = load_pickle(args.dbpedia_path)

    print(f"[01_run_stage12] Loading type dict from {args.type_dict_path}")
    type_dict = load_pickle(args.type_dict_path)

    with open(args.openai_key_file, "r") as f:
        openai.api_key = f.read().strip()

    adapter = Stage12Adapter(
        dbpedia=dbpedia,
        type_dict=type_dict,
        model_name=args.model,
        top_k=args.top_k,
        max_triples=args.max_triples,
    )

    # FactKG entries are keyed by claim text; values are dicts with keys
    # 'Entity_set', 'Label', 'types'. We key our cache by an integer qid so
    # duplicate claim strings (same text, different label) don't collide.
    claim_items = list(factkg.items())
    if args.limit:
        claim_items = claim_items[: args.limit]

    # Cache is a LIST of records (not a dict keyed by claim) to avoid collisions.
    results = []
    empty_pools = 0
    pool_sizes = []
    for qid, (claim_text, claim_data) in enumerate(
        tqdm(claim_items, desc=f"Stage12::{args.split}")
    ):
        try:
            out = adapter.process(claim_text, claim_data.get("Entity_set", []))
            label = claim_data.get("Label", [None])
            out["qid"] = qid
            out["label"] = label[0] if isinstance(label, (list, tuple)) and label else label
            out["reasoning_types"] = claim_data.get("types", [])
            results.append(out)

            n = len(out["triple_pool"])
            pool_sizes.append(n)
            if n == 0:
                empty_pools += 1
        except Exception as e:
            print(f"[error] qid={qid} {claim_text[:60]}... :: {e}")
            continue

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(results, output_path)

    n_saved = len(results)
    avg_pool = (sum(pool_sizes) / n_saved) if n_saved else 0.0
    print(f"[01_run_stage12] Saved {n_saved}/{len(claim_items)} claims → {output_path}")
    print(
        f"[01_run_stage12] Triple-pool coverage: "
        f"avg={avg_pool:.2f} triples/claim, "
        f"empty={empty_pools} ({100*empty_pools/max(n_saved,1):.1f}%)"
    )
    if empty_pools and empty_pools / max(n_saved, 1) > 0.15:
        print(
            "[01_run_stage12] WARNING: high empty-pool rate — these claims will "
            "fall back to a degenerate (all-NULL) graph at Stage 3."
        )


if __name__ == "__main__":
    main()
