# BERT-Base — Claim Only Baseline

This module implements the **Claim Only** baseline for the FactKG fact-verification task using BERT-based sequence classifiers. The model takes only the **claim text** as input (no external evidence from the Knowledge Graph) and predicts whether the claim is `True` or `False`.

Supported models via Hugging Face `transformers`:
- `bert-base-uncased` (BERT)
- `bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12` (BlueBERT)

> **Paper**: [FactKG: Fact Verification via Reasoning on Knowledge Graphs](https://arxiv.org/abs/2305.06590) (ACL 2023)

---

## Directory Structure

```
BERT-Base/
├── bert_classification.py   # Main training & evaluation script
├── flan_xl_zeroshot.py      # Zero-shot inference with Flan-T5 (optional)
├── requirements.txt         # Python dependencies
└── README.md
```

---

## Dataset

Download the FactKG dataset from [Google Drive](https://drive.google.com/drive/folders/1q0_MqBeGAp5_cBJCBf_1alYaYm14OeTk?usp=share_link).

| File | Description |
|---|---|
| `factkg_train.pickle` | Training set (~108k claims) |
| `factkg_dev.pickle` | Development / validation set |
| `factkg_test.pickle` | Test set (no `Evidence` field) |

Each claim entry in the pickle dictionary contains:
- `Label` — `[True]` or `[False]`
- `Entity_set` — named entities in the claim
- `Evidence` — ground-truth KG paths *(training/dev only)*
- `types` — reasoning type tags: `num1` (One-hop), `multi hop`, `multi claim`, `existence`, `negation`

**Note**: This Claim Only baseline does **not** use `Evidence` or the KG at inference time.

---

## Installation

```bash
pip install -r requirements.txt
```

`requirements.txt` includes: `pandas`, `pytorch-lightning`, `scikit-learn`, `transformers`.

> A CUDA-capable GPU is strongly recommended.

---

## Training & Evaluation

All training and evaluation is handled by a single script: `bert_classification.py`.

### BERT (`bert-base-uncased`)

```bash
python bert_classification.py \
  --model_name bert-base-uncased \
  --exp_name bert_log \
  --train_data_path /path/to/factkg_train.pickle \
  --valid_data_path /path/to/factkg_test.pickle \
  --scheduler linear \
  --batch_size 64 \
  --eval_batch_size 64 \
  --total_epoch 3
```

### BlueBERT

```bash
python bert_classification.py \
  --model_name bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12 \
  --exp_name bluebert_log \
  --train_data_path /path/to/factkg_train.pickle \
  --valid_data_path /path/to/factkg_test.pickle \
  --scheduler linear \
  --batch_size 64 \
  --eval_batch_size 64 \
  --total_epoch 3
```

### Flan-T5 XL (Zero-shot, optional)

```bash
python flan_xl_zeroshot.py \
  --valid_data_path /path/to/factkg_test.pickle \
  --model_name google/flan-t5-xl
```

---

## Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--model_name` | `bert-base-uncased` | Hugging Face model identifier |
| `--exp_name` | *(required)* | Experiment name used for log directory |
| `--train_data_path` | *(required)* | Path to `factkg_train.pickle` |
| `--valid_data_path` | *(required)* | Path to `factkg_test.pickle` (or dev) |
| `--batch_size` | `4` | Training batch size |
| `--eval_batch_size` | `8` | Evaluation batch size |
| `--total_epoch` | `None` | Number of training epochs |
| `--lr` | `1e-4` | Learning rate |
| `--scheduler` | `fixed` | LR scheduler: `fixed`, `linear`, `plateau` |
| `--warmup_steps` | `0` | Warmup steps for linear scheduler |
| `--optim` | `adam` | Optimizer: `adam`, `adamw` |
| `--weight_decay` | `0.1` | Weight decay (used with `adamw`) |
| `--load_model_path` | `None` | Path to a pre-trained checkpoint to resume |
| `--accumulation_steps` | `1` | Gradient accumulation steps |
| `--report_every_step` | `10` | Log training loss every N steps |
| `--eval_every_step` | `100` | Run evaluation every N steps |
| `--save_every_step` | `500` | Save checkpoint every N steps |

---

## Output

- **Logs**: saved in `exp_<exp_name>/` as `.log` files with timestamps.
- **Checkpoints**: saved under `./<model_prefix>/checkpoint-<epoch>/pytorch_model.bin` after each epoch.
- **Evaluation**: per-epoch accuracy is printed to console and logged, broken down by reasoning type (One-hop, Multi-hop, Conjunction, Existence, Negation).

---

## Reasoning Types

The model evaluates accuracy broken down by the following reasoning categories:

| Code | Type |
|---|---|
| `num1` | One-hop |
| `multi hop` | Multi-hop |
| `multi claim` | Conjunction |
| `existence` | Existence |
| `negation` | Negation |

---

## License

CC BY-NC-SA 4.0
