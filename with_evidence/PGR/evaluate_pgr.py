"""
Script đánh giá kết quả PGR (test_final.py) theo từng dạng lý luận của FactKG.
Chạy sau khi đã hoàn tất cả Bước 1 (generation) và Bước 2 (execution).

Cách dùng:
    python evaluate_pgr.py                          # Đánh giá kết quả DeepSeek (mặc định)
    python evaluate_pgr.py --result result_final_qwen.pickle  # Đánh giá kết quả Qwen
"""

import pickle
import argparse
from collections import defaultdict

# Ánh xạ nhãn phân loại trong FactKG -> Tên đẹp để in ra
CATEGORIES = {
    "num1":        "One-hop",
    "multi claim": "Conjunction",
    "existence":   "Existence",
    "multi hop":   "Multi-hop",
    "negation":    "Negation",
}

def evaluate(result_path, test_data_path, jsonl_path):
    print(f"\nĐang tải kết quả từ: {result_path}")
    with open(result_path, 'rb') as f:
        result = pickle.load(f)

    print(f"Đang tải nhãn phân loại từ: {test_data_path}")
    with open(test_data_path, 'rb') as f:
        test_data = pickle.load(f)

    # Lấy ánh xạ: question_id -> claim (dùng file JSONL)
    print(f"Đang tải ánh xạ câu hỏi từ: {jsonl_path}")
    import json
    qid_to_claim = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                q = json.loads(line)
                qid_to_claim[q['question_id']] = q['question']

    # Khởi tạo bộ đếm
    overall_correct = 0
    overall_total = 0
    cat_correct = defaultdict(int)
    cat_total = defaultdict(int)
    no_type_count = 0

    for qid, verdict in result.items():
        if verdict in ('No Program', 'Another Answer'):
            continue

        claim = qid_to_claim.get(qid, None)
        if claim is None or claim not in test_data:
            no_type_count += 1
            continue

        is_correct = (verdict == 'Correct')
        overall_total += 1
        if is_correct:
            overall_correct += 1

        types = test_data[claim].get('types', [])
        for t in types:
            if t in CATEGORIES:
                cat_total[t] += 1
                if is_correct:
                    cat_correct[t] += 1

    # In báo cáo
    print("\n" + "=" * 52)
    print("   BÁO CÁO BENCHMARK PGR (FactKG Test Set)")
    print("=" * 52)
    if overall_total == 0:
        print("❌ Chưa có kết quả hợp lệ nào.")
        return

    acc = overall_correct / overall_total
    print(f"  Tổng số câu được chấm   : {overall_total}")
    print(f"  Số câu đúng             : {overall_correct}")
    print(f"  OVERALL ACCURACY        : {acc:.2%}  ({overall_correct}/{overall_total})")
    print("-" * 52)
    for cat_key, cat_name in CATEGORIES.items():
        tot = cat_total[cat_key]
        cor = cat_correct[cat_key]
        if tot > 0:
            print(f"  [{cat_name:<12}]  {cor/tot:>7.2%}  ({cor}/{tot})")
        else:
            print(f"  [{cat_name:<12}]  N/A  (chưa có dữ liệu)")
    print("=" * 52 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đánh giá kết quả PGR trên FactKG.")
    parser.add_argument(
        '--result', type=str, default='./result_final.pickle',
        help='Đường dẫn file pickle kết quả (Do test_final.py sinh ra). Mặc định: ./result_final.pickle'
    )
    parser.add_argument(
        '--test_data', type=str, default='../data/factkg_test.pickle',
        help='Đường dẫn file pickle dữ liệu gốc FactKG (chứa nhãn types). Mặc định: ../data/factkg_test.pickle'
    )
    parser.add_argument(
        '--jsonl', type=str, default='../data/factkg_test_pgr.jsonl',
        help='Đường dẫn file JSONL (do convert_test.py tạo ra). Mặc định: ../data/factkg_test_pgr.jsonl'
    )
    args = parser.parse_args()
    evaluate(args.result, args.test_data, args.jsonl)
