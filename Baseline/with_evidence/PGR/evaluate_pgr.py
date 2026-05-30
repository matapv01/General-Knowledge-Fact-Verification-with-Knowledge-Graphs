import argparse
import json
import pickle
import sys
from pathlib import Path

SHARED_DIR = Path(__file__).resolve().parents[2] / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from factkg_metrics import print_results_from_tags


def load_qid_to_claim(jsonl_path):
    qid_to_claim = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                qid_to_claim[row["question_id"]] = row["question"]
    return qid_to_claim


def evaluate(result_path, test_data_path, jsonl_path):
    with open(result_path, "rb") as f:
        results = pickle.load(f)
    with open(test_data_path, "rb") as f:
        test_data = pickle.load(f)

    qid_to_claim = load_qid_to_claim(jsonl_path)

    is_correct_list = []
    tag_lists       = []

    for qid, verdict in results.items():
        if verdict in ("No Program", "Another Answer"):
            continue
        claim = qid_to_claim.get(qid)
        if claim is None or claim not in test_data:
            continue
        is_correct_list.append(verdict == "Correct")
        tag_lists.append(test_data[claim].get("types", []))

    print_results_from_tags("PGR", is_correct_list, tag_lists)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result",    default="result_final.pickle",
                        help="Pickle output from test_final.py")
    parser.add_argument("--test-data", default="../data/factkg_test.pickle",
                        help="FactKG test pickle with ground-truth labels")
    parser.add_argument("--jsonl",     default="../data/factkg_test_pgr.jsonl",
                        help="JSONL file mapping question_id to claim text")
    args = parser.parse_args()

    evaluate(args.result, args.test_data, args.jsonl)


if __name__ == "__main__":
    main()
