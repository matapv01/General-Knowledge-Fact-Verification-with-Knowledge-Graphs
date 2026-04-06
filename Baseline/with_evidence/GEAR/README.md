# GEAR — Graph-Evidence Augmented Reasoning Baseline

This module implements the **With Evidence** baseline for the FactKG fact-verification task. Unlike the Claim Only approach, **GEAR** first retrieves supporting evidence paths from the DBpedia Knowledge Graph, then uses that evidence together with the claim to predict `True` / `False`.

The pipeline consists of two major stages:

```
Stage 1: Graph Retriever  →  Stage 2: Classifier
  (retrieve/)                  (classifier/)
```

> **Paper**: [FactKG: Fact Verification via Reasoning on Knowledge Graphs](https://arxiv.org/abs/2305.06590) (ACL 2023)

---

## Directory Structure

```
GEAR/
├── retrieve/
│   └── model/
│       ├── config/
│       │   ├── relation_predict_top3.yaml
│       │   ├── relation_predict_top5.yaml
│       │   ├── relation_predict_top10.yaml
│       │   └── hop_predict.yaml
│       ├── relation_predict/
│       │   ├── main.py        # Train / eval Relation Predictor
│       │   ├── model.py       # FactKGRelationClassifier (Lightning)
│       │   └── data.py        # DataModule for relation prediction
│       └── hop_predict/
│           ├── main.py        # Train / eval Hop Predictor
│           ├── model.py       # HopPredictorManager
│           └── data.py        # Dataset builder for hop prediction
├── classifier/
│   ├── baseline.py            # Main classifier training & evaluation script
│   └── preprocess.py          # KG path search & candidate generation
├── requirements.txt
└── README.md
```

---

## Dataset

Download the FactKG dataset and DBpedia KG from [Google Drive](https://drive.google.com/drive/folders/1q0_MqBeGAp5_cBJCBf_1alYaYm14OeTk?usp=share_link).

| File | Description |
|---|---|
| `factkg_train.pickle` | Training set (~108k claims) |
| `factkg_dev.pickle` | Development / validation set |
| `factkg_test.pickle` | Test set (no ground-truth `Evidence`) |
| `dbpedia_2015_undirected.pickle` | Full DBpedia KG (undirected) |
| `dbpedia_2015_undirected_light.pickle` | Subset of DBpedia used in FactKG |

Each claim in the pickle has: `Label`, `Entity_set`, `Evidence` (train/dev only), `types`.

---

## Installation

```bash
pip install -r requirements.txt
```

`requirements.txt` includes: `pandas`, `pytorch-lightning`, `scikit-learn`, `transformers`.

> A CUDA-capable GPU is required for both stages.

---

## Pipeline Overview

```
factkg_{train,dev,test}.pickle
        │
        ▼
┌─────────────────────────────────────────┐
│  Stage 1a — Relation Predictor          │
│  retrieve/model/relation_predict/       │
│  Predicts top-K KG relations per claim  │
└───────────────┬─────────────────────────┘
                │  test_relations_top3.json
                ▼
┌─────────────────────────────────────────┐
│  Stage 1b — Hop Predictor               │
│  retrieve/model/hop_predict/            │
│  Predicts number of hops per claim      │
└───────────────┬─────────────────────────┘
                │  predictions_hop.json
                ▼
┌─────────────────────────────────────────┐
│  Stage 2 — Classifier                   │
│  classifier/baseline.py                 │
│  Uses KG paths + claim → True / False   │
└─────────────────────────────────────────┘
```

---

## Stage 1: Graph Retriever

Navigate into the `retrieve/` directory before running any retriever commands.

```bash
cd retrieve
```

### Step 1 — Train the Relation Predictor

The relation predictor is a multi-label BERT classifier that predicts the most relevant KG relations for a claim.

```bash
cd model/relation_predict

# Train
python main.py --mode train --config ../config/relation_predict_top3.yaml

# Evaluate & generate test_relations_top3.json
python main.py --mode eval \
  --config ../config/relation_predict_top3.yaml \
  --model_path <path/to/checkpoint.ckpt>
```

**Config options** (`config/relation_predict_top3.yaml`):

| Key | Value | Description |
|---|---|---|
| `data_path` | `../total_data.pkl` | Preprocessed combined dataset |
| `relation_path` | `./relations_for_final.pickle` | Set of all KG relations |
| `model_name` | `bert-base-uncased` | Backbone model |
| `train_batch_size` | `125` | Training batch size |
| `eval_batch_size` | `125` | Evaluation batch size |
| `max_input_length` | `512` | Max token length |
| `max_epoch` | `10` | Training epochs |
| `top_k` | `3` | Number of top relations to predict |

Output: `model/relation_predict/test_relations_top3.json`

---

### Step 2 — Train the Hop Predictor

The hop predictor is a BERT classifier that predicts the number of reasoning hops (1, 2, or 3) required for a claim.

```bash
cd model/hop_predict

# Train
python main.py --mode train --config ../config/hop_predict.yaml

# Evaluate & generate predictions_hop.json
python main.py --mode eval \
  --config ../config/hop_predict.yaml \
  --model_path ./model.pth
```

**Config options** (`config/hop_predict.yaml`):

| Key | Value | Description |
|---|---|---|
| `train_data_path` | `../train.json` | Training data |
| `dev_data_path` | `../dev.json` | Validation data |
| `test_data_path` | `../test.json` | Test data |
| `model_name` | `bert-base-uncased` | Backbone model |
| `NUM_EPOCHS` | `1` | Training epochs |
| `TRAIN_BATCH_SIZE` | `64` | Training batch size |
| `VAL_BATCH_SIZE` | `64` | Validation batch size |
| `TEST_BATCH_SIZE` | `1` | Test batch size |
| `LEARNING_RATE` | `1e-5` | Learning rate |

Output: `model/hop_predict/predictions_hop.json`

---

## Stage 2: Classifier

The classifier reads the KG evidence paths produced by Stage 1 and trains a BERT-based model (`bert-base-cased`) to verify each claim.

Internally, `preprocess.py` is called automatically by `baseline.py` to:
1. Read `test_relations_top3.json` and `predictions_hop.json` from Stage 1.
2. Walk the KG and collect `connected` and `walkable` candidate paths per claim.
3. Save candidate paths as `train_candid_paths.bin`, `dev_candid_paths.bin`, `test_candid_paths_top3.bin`.

```bash
cd classifier

python baseline.py \
  --data_path /path/to/factkg_data_directory \
  --kg_path /path/to/dbpedia_2015_undirected_light.pickle
```

**Arguments**:

| Argument | Default | Description |
|---|---|---|
| `--data_path` | *(required)* | Directory containing `factkg_{train,dev,test}.pickle` |
| `--kg_path` | *(required)* | Path to DBpedia KG pickle file |
| `--lr` | `5e-5` | Learning rate (Adam optimizer) |
| `--model_cls` | `cat` | Classifier type: `cat` (claim + evidence concat) or `sent` (claim only) |
| `--epoch` | `10` | Max training epochs (early stopping after 3 bad epochs) |
| `--n_candid` | `3` | Number of candidate path sets to load (`test_candid_paths_top3.bin`) |
| `--scratch` | *(flag)* | If set, initializes BERT with random weights (no pre-training) |

**Classifier architecture** (`--model_cls cat`):
- Tokenizes claim and evidence separately using `bert-base-cased`
- Concatenates token sequences: `[CLS] claim [SEP] evidence [SEP]`
- Passes through BERT encoder → takes `[CLS]` representation → 2-layer MLP → binary output

**Output files**:
- `valid_pred.bin` — validation prediction results
- `test_pred.bin` — final test prediction results with per-type accuracy breakdown

---

## Evaluation

Both the Relation Predictor and the Classifier automatically report accuracy broken down by reasoning type:

| Type Code | Reasoning Type |
|---|---|
| `0` / `num1` | One-hop |
| `1` / `multi hop` | Multi-hop |
| `2` / `multi claim` | Conjunction |
| `3` / `existence` | Existence |
| `4` / `negation` | Negation |

---

## Notes

- The file paths in `preprocess.py` assume the script is run **from the `classifier/` directory** (relative paths `../retrieve/model/...` are hardcoded).
- The Relation Predictor uses `pytorch-lightning`'s `Trainer` with mixed precision (`16-mixed`) on GPU.
- The Hop Predictor saves its final model as `model.pth` in the working directory.
- Early stopping in the Classifier triggers after **3 consecutive epochs** with no improvement on the dev set (best checkpoint is kept in memory).

---

## License

CC BY-NC-SA 4.0
