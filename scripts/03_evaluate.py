"""Evaluate a trained KernelKG-GPT checkpoint on a cached split.

Reports overall accuracy and per-reasoning-type breakdown (FactKG ``types``).
"""

import argparse
import os
import sys
from collections import defaultdict

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stage3 import FactKGGraphDataset, build_model, collate_fn  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split_cache", default=None,
                        help="Override config['test_cache']")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(config["bert_model"])

    model = build_model(config).to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    cache_path = args.split_cache or config["test_cache"]
    test_ds = FactKGGraphDataset(
        cache_path, tokenizer,
        max_seq_len=config["max_seq_len"],
        max_nodes=config["max_nodes"],
        formatter_name=config.get("triple_format", "plain"),
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn,
    )

    correct = 0
    total = 0
    type_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="evaluate"):
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                token_type_ids=batch["token_type_ids"].to(device),
                is_null=batch["is_null"].to(device),
            )
            preds = outputs["logits"].argmax(dim=-1).cpu()
            labels = batch["labels"]
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for i, types in enumerate(batch["reasoning_types"]):
                hit = int(preds[i].item() == labels[i].item())
                for t in types:
                    type_stats[t]["total"] += 1
                    type_stats[t]["correct"] += hit

    print("\n=== FINAL RESULTS ===")
    print(f"Cache: {cache_path}")
    print(f"Checkpoint: {args.model_path}")
    print(f"Overall accuracy: {correct/max(total,1):.4f}  ({correct}/{total})")
    print("\nPer reasoning type:")
    for t, s in sorted(type_stats.items()):
        if s["total"] > 0:
            print(f"  {t:25s}: {s['correct']/s['total']:.4f}  ({s['total']})")


if __name__ == "__main__":
    main()
