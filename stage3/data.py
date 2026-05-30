"""Dataset for Stage 3 — turns Stage 1+2 cache into BERT-tokenized graphs.

The Stage 1+2 cache is a LIST of records (see scripts/01_run_stage12.py), each
``{qid, claim, triple_pool, label, reasoning_types}``. Tokenization is done
once in ``__init__`` and stored in compact numpy arrays (uint16 ids, uint8
masks) so it is not repeated every epoch and stays memory-friendly even for
the ~86k-claim FactKG train split (~0.4 GB).
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .graph_builder import build_graph


def _label_to_int(label_raw):
    if isinstance(label_raw, (list, tuple)):
        label_raw = label_raw[0] if label_raw else None
    if isinstance(label_raw, str):
        return 1 if label_raw.strip().lower() in ("true", "1", "supported") else 0
    return 1 if bool(label_raw) else 0


class FactKGGraphDataset(Dataset):
    def __init__(
        self,
        cache_path: str,
        tokenizer,
        max_seq_len: int = 96,
        max_nodes: int = 10,
        formatter_name: str = "plain",
        show_progress: bool = True,
    ):
        records = torch.load(cache_path)
        # Backward-compat: accept the old dict-keyed-by-claim format too.
        if isinstance(records, dict):
            records = [
                {"claim": k, **v} for k, v in records.items()
            ]

        self.max_seq_len = max_seq_len
        self.max_nodes = max_nodes

        n = len(records)
        # uint16 is enough for vocabs <= 65535 (bert-base = 30522); fall back
        # to int32 for larger vocabularies.
        vocab_size = getattr(tokenizer, "vocab_size", 0) or 0
        id_dtype = np.uint16 if vocab_size and vocab_size <= 65535 else np.int32
        self._ids = np.zeros((n, max_nodes, max_seq_len), dtype=id_dtype)
        self._mask = np.zeros((n, max_nodes, max_seq_len), dtype=np.uint8)
        self._seg = np.zeros((n, max_nodes, max_seq_len), dtype=np.uint8)
        self._null = np.zeros((n, max_nodes), dtype=np.bool_)
        self._labels = np.zeros((n,), dtype=np.int64)
        self.claims = []
        self.reasoning_types = []

        it = tqdm(records, desc=f"tokenize {cache_path}") if show_progress else records
        for i, item in enumerate(it):
            claim_text = item["claim"]
            graph = build_graph(
                claim=claim_text,
                triple_pool=item["triple_pool"],
                max_nodes=max_nodes,
                formatter_name=formatter_name,
            )
            for k, node in enumerate(graph["nodes"]):
                enc = tokenizer(
                    claim_text,
                    node["triple_text"],
                    max_length=max_seq_len,
                    padding="max_length",
                    truncation=True,
                    return_token_type_ids=True,
                )
                self._ids[i, k] = enc["input_ids"]
                self._mask[i, k] = enc["attention_mask"]
                self._seg[i, k] = enc["token_type_ids"]
                self._null[i, k] = node["is_null"]

            self._labels[i] = _label_to_int(item.get("label"))
            self.claims.append(claim_text)
            self.reasoning_types.append(item.get("reasoning_types", []))

    def __len__(self):
        return len(self.claims)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.from_numpy(self._ids[idx].astype(np.int64)),
            "attention_mask": torch.from_numpy(self._mask[idx].astype(np.int64)),
            "token_type_ids": torch.from_numpy(self._seg[idx].astype(np.int64)),
            "is_null": torch.from_numpy(self._null[idx].copy()),
            "label": torch.tensor(int(self._labels[idx]), dtype=torch.long),
            "claim": self.claims[idx],
            "reasoning_types": self.reasoning_types[idx],
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
