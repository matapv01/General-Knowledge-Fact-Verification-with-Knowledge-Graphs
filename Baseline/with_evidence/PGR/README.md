# PGR

This repository contains the implementation of the method described in the paper  
**[Fact Verification on Knowledge Graph via Programmatic Graph Reasoning](https://aclanthology.org/2025.findings-emnlp.293/)**.

Experiments are conducted on the **[FactKG](https://github.com/jiho283/FactKG)** dataset.

---

## 🧩 Overview

The project reproduces the two main stages of the paper's proposed framework:

1. **Program Generation** – generate graph reasoning programs.  
2. **Program Execution** – execute the generated programs to verify claims on the knowledge graph.

---

## ⚙️ Setup

All dependencies are installed (argparse, json, re, pickle, requests, ast).

Api used for generation and keys need to be given in 'test_program_generation.py' and 'kg_program_bi_reverse.py'. The url and headers should be changed according to your chosen api.

```python
def chat_responce(...):
    url = 'https://api.deepseek.com/chat/completions'
    headers = {"Content-Type": "application/json", "Authorization": 'Bearer your-key'}
```

---

## 🚀 Running the Pipeline

### Step 1. Program Generation

Run the following command to generate reasoning programs, 'path_test_dataset' in this paper is the path of FactKG test dataset (preprocessed by **[tool of KG-GPT](https://github.com/jiho283/KG-GPT/blob/main/factkg/data/preprocess.py)**):

```bash
python test_program_generation.py --model llm_name --test path_test_dataset
```

All generated programs will be saved under './result/program_generate/'.

### Step 2. Program Execution

After generating the programs, execute them on the target knowledge graph (DBpedia used here is preprocessed and can be downloaded in **[FactKG](https://github.com/jiho283/FactKG)** dataset):

```bash
python test_final.py --model llm_name --kg path_kg --test path_test_dataset
```

This step will produce verification results and overall accuracy.

## Few-shot Prompt Design

All prompts in this project are designed using FactKG-based few-shot examples.
By modifying the examples or adjusting the function executions,
the framework can be easily adapted to other tasks on knowledge graphs.
