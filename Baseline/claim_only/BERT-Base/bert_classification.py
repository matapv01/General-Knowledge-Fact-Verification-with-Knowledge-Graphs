import argparse
import logging
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import logging as hf_logging

SHARED_DIR = Path(__file__).resolve().parents[3] / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from factkg_metrics import get_type_id, print_results

hf_logging.set_verbosity_error()
torch.manual_seed(555)


class WarmupLinearScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps, scheduler_steps, last_epoch=-1):
        self.warmup_steps    = warmup_steps
        self.scheduler_steps = scheduler_steps
        super().__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(
            0.0,
            float(self.scheduler_steps - step)
            / float(max(1, self.scheduler_steps - self.warmup_steps)),
        )


def define_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name",        required=True,   type=str)
    parser.add_argument("--train_data_path", required=True,   type=str)
    parser.add_argument("--valid_data_path", required=True,   type=str)
    parser.add_argument("--model_name",      default="bert-base-uncased", type=str)
    parser.add_argument("--load_model_path", default=None,    type=str)
    parser.add_argument("--batch_size",      default=4,       type=int)
    parser.add_argument("--eval_batch_size", default=8,       type=int)
    parser.add_argument("--total_step",      default=1000000, type=int)
    parser.add_argument("--total_epoch",     default=None,    type=int)
    parser.add_argument("--warmup_steps",    default=0,       type=int)
    parser.add_argument("--scheduler_steps", default=None,    type=int)
    parser.add_argument("--accumulation_steps", default=1,    type=int)
    parser.add_argument("--lr",              default=1e-4,    type=float)
    parser.add_argument("--clip",            default=None,    type=float)
    parser.add_argument("--optim",           default="adam",  type=str)
    parser.add_argument("--scheduler",       default="fixed", type=str)
    parser.add_argument("--weight_decay",    default=0.1,     type=float)
    parser.add_argument("--report_every_step", default=10,    type=int)
    parser.add_argument("--save_every_step",   default=500,   type=int)
    parser.add_argument("--eval_every_step",   default=100,   type=int)
    parser.add_argument("--patience",          default=10,    type=int)
    return parser.parse_args()


class FactKGDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data      = data
        self.tokenizer = tokenizer
        self.claims    = list(data.keys())

    def __len__(self):
        return len(self.claims)

    def __getitem__(self, index):
        claim = self.claims[index]
        entry = self.data[claim]

        label   = int(entry["Label"][0])
        type_id = get_type_id(entry.get("types", []))

        encoded = self.tokenizer.encode_plus(
            claim,
            add_special_tokens=True,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return (
            encoded["input_ids"][0],
            encoded["attention_mask"][0],
            label,
            type_id,
        )


def set_optim(args, model):
    if args.optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    scheduler = None
    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=args.patience
        )
    elif args.scheduler == "linear":
        n_steps   = args.scheduler_steps or args.total_step
        scheduler = WarmupLinearScheduler(
            optimizer, warmup_steps=args.warmup_steps, scheduler_steps=n_steps
        )
    return optimizer, scheduler


def run_eval(model, dataloader, device, loss_fn):
    model.eval()
    torch.set_grad_enabled(False)

    labels   = []
    preds    = []
    type_ids = []
    total_loss = 0.0

    for batch in dataloader:
        input_ids    = batch[0].to(device)
        attn_mask    = batch[1].to(device)
        true_labels  = batch[2].to(device)
        batch_types  = batch[3]

        outputs = model(input_ids, attention_mask=attn_mask)
        loss    = loss_fn(outputs[0], true_labels)
        total_loss += loss.item()

        preds.append(outputs[0].detach().cpu().numpy())
        labels.extend(true_labels.cpu().numpy().tolist())
        type_ids.extend(batch_types.tolist())

    model.train()
    torch.set_grad_enabled(True)

    labels   = np.array(labels)
    preds    = np.argmax(np.vstack(preds), axis=1)
    type_ids = np.array(type_ids)
    avg_loss = total_loss / len(dataloader)

    return labels, preds, type_ids, avg_loss


def main(args):
    exp_dir = f"exp_{args.exp_name}"
    os.makedirs(exp_dir, exist_ok=True)

    model_prefix = args.model_name.split("/")[0].split("-")[0]
    os.makedirs(model_prefix, exist_ok=True)

    log_path = os.path.join(
        exp_dir,
        args.model_name + "_" + datetime.now().strftime("%Y-%m-%d %H.%M.%S") + ".log",
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path)],
    )
    logger = logging.getLogger(__name__)
    logger.info("exp=%s  model=%s", args.exp_name, args.model_name)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, do_lower_case=True)
    model     = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    ).to(device)

    if args.load_model_path:
        model.load_state_dict(torch.load(args.load_model_path))
        logger.info("Loaded checkpoint: %s", args.load_model_path)

    optimizer, scheduler = set_optim(args, model)

    with open(args.train_data_path, "rb") as f:
        train_data = pickle.load(f)
    with open(args.valid_data_path, "rb") as f:
        valid_data = pickle.load(f)

    train_dataset = FactKGDataset(train_data, tokenizer)
    val_dataset   = FactKGDataset(valid_data, tokenizer)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=8
    )

    loss_fn = nn.CrossEntropyLoss()
    step    = 0

    for epoch in range(args.total_epoch):
        model.train()
        torch.set_grad_enabled(True)

        for i, batch in enumerate(train_loader):
            step += 1
            print(f"Epoch {epoch} | Batch {i}/{len(train_loader)}", end="\r")

            input_ids   = batch[0].to(device)
            attn_mask   = batch[1].to(device)
            true_labels = batch[2].to(device)

            model.zero_grad()
            outputs    = model(input_ids, attention_mask=attn_mask)
            loss       = loss_fn(outputs[0], true_labels)
            train_loss = round(loss.item() * args.accumulation_steps, 8)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % args.report_every_step == 0:
                lr = optimizer.param_groups[0]["lr"]
                logger.info("epoch=%d  step=%d  loss=%.6f  lr=%g", epoch, step, train_loss, lr)

            if step % 100 == 0:
                labels, preds, type_ids, val_loss = run_eval(model, val_loader, device, loss_fn)
                print_results(args.model_name, labels, preds, type_ids)
                logger.info(
                    "step=%d  val_loss=%.4f  val_acc=%.4f",
                    step, val_loss, accuracy_score(labels, preds)
                )

        ckpt_dir = f"./{model_prefix}/checkpoint-{epoch}/"
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(ckpt_dir, "pytorch_model.bin"))


if __name__ == "__main__":
    args = define_argparser()
    main(args)
