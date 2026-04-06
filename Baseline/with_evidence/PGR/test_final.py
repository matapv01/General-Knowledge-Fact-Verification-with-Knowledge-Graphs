import argparse
import json
import os
import time
import re
import pickle

import requests

from kg_program_bi_reverse import execute_program

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()
    
def extract_code_from_string(input_string):
    match = re.search(r'```python\n(.*?)```', input_string, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return input_string
    
PROGRAM_DIR = './result/program_generate'  # Default, bị ghi đè bởi --program_dir
RESULT_FILE = './result_final.pickle'      # Default, bị ghi đè bởi --result_file

def get_answer(model_name: str, qid: int, claim: str, gt_entities: list, KG: dict, max_tokens: int):
    
    if os.path.exists(f'{PROGRAM_DIR}/{qid}_program.json'):
        pass
    else:
        return "No Program"
    with open(f'{PROGRAM_DIR}/{qid}_program.json', 'r') as f:
        program_gen_dict = json.load(f)

    program_extracted = extract_code_from_string(program_gen_dict['program'])
    predicted = execute_program(model_name, program_extracted, KG, claim, qid)

    return predicted

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parsing input arguments.")
    parser.add_argument('--model', type=str, required=True, help='Model name. e.g. gpt-4, deepseek-chat')
    parser.add_argument('--kg', type=str, required=True, help='Path of KG.')
    parser.add_argument('--test', type=str, required=True, help='Test dataset path.')
    parser.add_argument('--program_dir', type=str, default='./result/program_generate',
                        help='Thư mục chứa file program JSON (Sinh ra bởi Bước 1). Mặc định: ./result/program_generate')
    parser.add_argument('--result_file', type=str, default='./result_final.pickle',
                        help='File output lưu kết quả. Mặc định: ./result_final.pickle')
    
    args = parser.parse_args()

    model_name = args.model
    kg_path = args.kg
    test_path = args.test

    # Áp dụng tham số program_dir vào biến toàn cục
    import sys
    sys.modules[__name__].PROGRAM_DIR = args.program_dir
    sys.modules[__name__].RESULT_FILE = args.result_file

    print('Start KG loading...')
    with open(kg_path, 'rb') as f:
        dbp = pickle.load(f)

    print('-----KG loaded-----')
    
    final_results = []
    start_token = 0
    
    ####For new experiment, use it.
    result = {}
    if os.path.exists(RESULT_FILE):
        print(f"🔄 Đang tải lịch sử chấm bài từ {RESULT_FILE} - Chỉ chấm lại những câu Error hoặc chưa làm!")
        with open(RESULT_FILE, 'rb') as f:
            result = pickle.load(f)
            
    questions_dict = {}
    entity_set_dict = {}
    label_set_dict = {}

    with open(test_path) as f:
        for line in f:
            if not line:
                continue
            q = json.loads(line)
            questions_dict[q["question_id"]] = q["question"]
            entity_set_dict[q["question_id"]] = q["entity_set"]
            label_set_dict[q["question_id"]] = q["Label"]

    Correct = []
    Wrong = []
    Error = []
    Another = []

    for qid, question in questions_dict.items():
        # Bỏ qua những câu đã báo Correct, Wrong. Chỉ chạy lại những câu bị Error hoặc chưa làm
        if qid in result and result[qid] in ['Correct', 'Wrong', 'Another Answer']:
            continue
            
        try:
            final_r = get_answer(model_name, qid, question, entity_set_dict[qid], KG=dbp, max_tokens=1024)
            final_results.append(final_r)
            if(final_r == "No Program"):
                continue
            if final_r == 'Another Answer':
                Another.append(qid)
                print(qid, ': ', final_r)
                result[qid] = final_r
            elif final_r == label_set_dict[qid][0]:
                Correct.append(qid)
                print(qid, ': Correct!')
                result[qid] = 'Correct'
            else:
                Wrong.append(qid)
                print(qid, ': Wrong...')
                result[qid] = 'Wrong'
        except:
            Error.append(qid)
            print(qid, ': Error...')
            result[qid] = 'Error'
        with open(RESULT_FILE, 'wb') as f:
            pickle.dump(result, f)

    
    tot_corr = 0
    for tot_id in list(result):
        if result[tot_id] == 'Correct':
            tot_corr += 1
    
    accuracy = tot_corr / len(result) if len(result) > 0 else 0
    print(f'Accuracy: {accuracy:.4f}')
    print('Done! Thống kê đã được lưu tại ./result_statistics.txt')

    # Save statistics directly to readable text file
    with open('./result_statistics.txt', 'w', encoding='utf-8') as f:
        f.write("========== BÁO CÁO KẾT QUẢ THỰC THI (PGR) ==========\n")
        f.write(f"Total processed claims: {len(result)}\n")
        f.write(f"Correct (Đúng): {len(Correct)}\n")
        f.write(f"Wrong (Sai): {len(Wrong)}\n")
        f.write(f"Error (Lỗi): {len(Error)}\n")
        f.write(f"Another/No Program: {len(Another)}\n")
        f.write("----------------------------------------------------\n")
        f.write(f"OVERALL ACCURACY (Độ chính xác): {accuracy:.2%}\n")
        f.write("====================================================\n\n")
        f.write("Chi tiết kết quả từng Câu (Question ID):\n")
        for qid, res in result.items():
            f.write(f"Câu {qid} : {res}\n")