# General Knowledge Fact Verification with Knowledge Graphs

This repository contains various baseline methods for General Knowledge Fact Verification using Knowledge Graphs. The methods are categorized based on whether they use evidence or only the claim itself.

## Task Sheet
Track the progress and assignments on our Google Sheets task sheet:
[Task Sheet](https://docs.google.com/spreadsheets/d/13vaydmPTO7z-iW3VaIiTr7mDGFod6_7OP8Lcqfw2VyQ/edit?gid=1808260402#gid=1808260402)

## Methods and Instructions

### Claim Only Methods
These methods verify facts using only the claim without external retrieved evidence.

* **BERT-Base & FLAN-T5 (Zero-shot)**: 
  * Location: [`Baseline/claim_only/BERT-Base`](./Baseline/claim_only/BERT-Base)
  * Refer to the `README.md` inside the folder for instructions on how to install requirements and run the classification scripts.

### With Evidence Methods
These methods use retrieved evidence from Knowledge Graphs to verify claims.

* **GEAR**: 
  * Location: [`Baseline/with_evidence/GEAR`](./Baseline/with_evidence/GEAR)
  * Read the folder's `README.md` to see the setup and execution steps.

* **Hybrid Fact-Checking**:
  * Location: [`Baseline/with_evidence/Hybrid_Fact-Checking`](./Baseline/with_evidence/Hybrid_Fact-Checking)
  * A Jupyter notebook implementation. Open the `.ipynb` file to run the experiments.

* **KG-GPT**:
  * Location: [`Baseline/with_evidence/KG-GPT`](./Baseline/with_evidence/KG-GPT)
  * See the `README.md` in the folder for instructions on how to run tests on MetaQA and FactKG datasets.

* **PGR**:
  * Location: [`Baseline/with_evidence/PGR`](./Baseline/with_evidence/PGR)
  * Follow the instructions in the `README.md` within the folder to run the test scripts.

* **SimGRAG**:
  * Location: [`Baseline/with_evidence/SimGRAG`](./Baseline/with_evidence/SimGRAG)
  * Instructions can be found in the folder's `README.md`, including pipeline and retriever setups.
