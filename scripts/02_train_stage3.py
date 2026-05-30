"""Train Stage 3 — KernelGAT-style verifier on Stage 1+2 cached triple pools."""

import argparse
import os
import sys
from collections import defaultdict

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stage3 import FactKGGraphDataset, KernelKGGPTLoss, build_model, collate_fn  # noqa: E402
from utils import set_seed  # noqa: E402


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    type_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    with torch.no_grad():
        for batch in tqdm(loader, desc="eval"):
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

    acc = correct / max(total, 1)
    return acc, type_stats


def train(config):
    set_seed(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config["output_dir"], exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(config["bert_model"])

    train_ds = FactKGGraphDataset(
        config["train_cache"], tokenizer,
        max_seq_len=config["max_seq_len"],
        max_nodes=config["max_nodes"],
        formatter_name=config.get("triple_format", "plain"),
    )
    dev_ds = FactKGGraphDataset(
        config["dev_cache"], tokenizer,
        max_seq_len=config["max_seq_len"],
        max_nodes=config["max_nodes"],
        formatter_name=config.get("triple_format", "plain"),
    )

    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"],
        shuffle=True, collate_fn=collate_fn,
        num_workers=config.get("num_workers", 0),
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=config["eval_batch_size"],
        shuffle=False, collate_fn=collate_fn,
        num_workers=config.get("num_workers", 0),
    )

    model = build_model(config).to(device)
    print(f"[train] model_type = {config.get('model_type', 'kernel')}")

    criterion = KernelKGGPTLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    accum = max(1, config.get("gradient_accumulation", 1))
    total_steps = (len(train_loader) * config["num_epochs"]) // accum
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config["warmup_ratio"]),
        num_training_steps=total_steps,
    )

    best_acc = 0.0
    patience = 0

    for epoch in range(config["num_epochs"]):
        model.train()
        optimizer.zero_grad()

        running = 0.0
        for step, batch in enumerate(tqdm(train_loader, desc=f"epoch {epoch+1}")):
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                token_type_ids=batch["token_type_ids"].to(device),
                is_null=batch["is_null"].to(device),
            )
            losses = criterion(outputs, batch["labels"].to(device))
            (losses["total"] / accum).backward()
            running += losses["total"].item()

            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        n = len(train_loader)
        print(f"[epoch {epoch+1}] train loss = {running/n:.4f}")

        acc, type_stats = evaluate(model, dev_loader, device)
        print(f"[epoch {epoch+1}] dev acc = {acc:.4f}")
        for t, s in sorted(type_stats.items()):
            if s["total"] > 0:
                print(f"  {t:25s}: {s['correct']/s['total']:.4f} ({s['total']})")

        if acc > best_acc:
            best_acc = acc
            patience = 0
            ckpt_path = os.path.join(config["output_dir"], "best.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ↳ new best saved to {ckpt_path}")
        else:
            patience += 1
            if patience >= config["early_stopping_patience"]:
                print("Early stopping triggered.")
                break

    print(f"Best dev accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Optional key=value overrides, e.g. --override max_nodes=15")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    for kv in args.override:
        k, v = kv.split("=", 1)
        if k in config:
            t = type(config[k])
            config[k] = t(v) if t is not bool else (v.lower() == "true")
        else:
            try:
                config[k] = int(v)
            except ValueError:
                try:
                    config[k] = float(v)
                except ValueError:
                    config[k] = v

    train(config)
