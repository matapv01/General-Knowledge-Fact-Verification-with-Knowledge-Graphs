import json
import pickle
import os

output_file = "../results/FactKG_query_qwen_122b.txt"
labels_file = "../../data/factkg_test.pickle"

print("Đang tải dữ liệu nhãn (types) từ bộ FactKG gốc...")
with open(labels_file, "rb") as f:
    test_data = pickle.load(f)

# Ánh xạ theo các dạng reasoning ghi trong FactKG Paper
categories = {
    "num1": "One-hop",
    "multi claim": "Conjunction",
    "existence": "Existence",
    "multi hop": "Multi-hop",
    "negation": "Negation",
}

cat_correct = {cat: 0 for cat in categories.keys()}
cat_total = {cat: 0 for cat in categories.keys()}
overall_correct = 0
overall_total = 0

print(f"Đang đọc dữ liệu đầu ra từ: {output_file}...")
if not os.path.exists(output_file):
    print("❌ Thất bại: Bạn chưa chạy SimGRAG hoặc file chưa được sinh ra!")
    exit()

with open(output_file, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            query = data.get("query", "")
            is_correct = data.get("correct", False)
            
            # Tính điểm Overall
            overall_total += 1
            if is_correct:
                overall_correct += 1
                
            # Tính điểm theo từng Phân lớp (Type)
            if query in test_data:
                types = test_data[query].get("types", [])
                for t in types:
                    if t in categories:
                        cat_total[t] += 1
                        if is_correct:
                            cat_correct[t] += 1
        except Exception as e:
            pass

print("\n" + "="*50)
print(f"🚀 BÁO CÁO KẾT QUẢ ĐÁNH GIÁ (Phân tách theo Dạng lý luận)")
print("="*50)

if overall_total > 0:
    print(f"Tổng số Sample ghi nhận: {overall_total}")
    print(f"🔴 TỔNG ĐỘ CHÍNH XÁC (OVERALL): {overall_correct / overall_total:.2%} ({overall_correct}/{overall_total})")
    print("-" * 50)
    for cat_key, cat_name in categories.items():
        if cat_total[cat_key] > 0:
            acc = cat_correct[cat_key] / cat_total[cat_key]
            print(f"👉 [{cat_name}] Accuracy: {acc:.2%} ({cat_correct[cat_key]} / {cat_total[cat_key]})")
        else:
            print(f"👉 [{cat_name}] Accuracy: N/A (Tiến trình chưa chạy tới dạng này)")
else:
    print("Tiến trình của bạn chưa sinh ra Output hợp lệ nào.")
print("="*50 + "\n")
