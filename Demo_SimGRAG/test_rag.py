import sys
import os

# Đảm bảo đường dẫn import cho app
sys.path.append(os.path.join(os.path.dirname(__file__), "BE"))

from app.use_cases.qa_usecase import QAUseCase
from app.infrastructure.simgrag_adapter import SimGRAGAdapter

def test():
    print("Khởi tạo Fact Verification RAG Pipeline...")
    adapter = SimGRAGAdapter()
    use_case = QAUseCase(simgrag_adapter=adapter)
    
    claims = [
        # 5 True Claims
        {"claim": "Barack Obama was the 44th president of the United States.", "gt": "True"},
        {"claim": "The capital of France is Paris.", "gt": "True"},
        {"claim": "Albert Einstein was born in Ulm, Germany.", "gt": "True"},
        {"claim": "Microsoft was founded by Bill Gates.", "gt": "True"},
        {"claim": "The Amazon River is located in South America.", "gt": "True"},
        # 5 False Claims
        {"claim": "Neil Armstrong was the first person to walk on Mars.", "gt": "False"},
        {"claim": "Water freezes at 100 degrees Celsius.", "gt": "False"},
        {"claim": "The Eiffel Tower is located in London.", "gt": "False"},
        {"claim": "Apple Inc. is a Japanese multinational technology company.", "gt": "False"},
        {"claim": "Jupiter is the closest planet to the Sun.", "gt": "False"}
    ]
    
    print("\nChọn 1 nhận định (claim) để kiểm chứng sự thật (1-10):")
    for i, c in enumerate(claims, 1):
        print(f"{i}. {c['claim']} (GT: {c['gt']})")
        
    try:
        choice = int(input("\nNhập số (1-10): "))
        if 1 <= choice <= 10:
            selected_claim = claims[choice - 1]
        else:
            print("Lựa chọn không hợp lệ. Mặc định dùng câu 1.")
            selected_claim = claims[0]
    except ValueError:
        print("Đầu vào không hợp lệ. Mặc định dùng câu 1.")
        selected_claim = claims[0]
        
    claim = selected_claim["claim"]
    gt = selected_claim["gt"]
        
    print("\n===============================")
    print(f"Nhận định (Claim): {claim}")
    print(f"Ground Truth (GT): {gt}")
    print("===============================\n")
    
    response = use_case.execute(claim)
    
    print("\n===============================")
    print("🤖 LLM TRẢ LỜI:")
    print(response.answer)
    print("===============================\n")
    
    # Ghi log ra file txt
    log_path = os.path.join(os.path.dirname(__file__), "run_log.txt")
    with open(log_path, "w", encoding="utf-8") as f_log:
        f_log.write("===============================\n")
        f_log.write(f"Nhận định (Claim): {claim}\n")
        f_log.write(f"Ground Truth (GT): {gt}\n")
        f_log.write("===============================\n\n")
        f_log.write("1. LLM TRẢ LỜI:\n")
        f_log.write(f"{response.answer}\n\n")
        f_log.write("2. CÁC SUBGRAPH/EVIDENCE ĐƯỢC CHỌN VÀ ĐƯA VÀO PROMPT:\n")
        if not response.evidences:
            f_log.write("- Không tìm thấy evidence nào\n")
        else:
            for ev in response.evidences:
                f_log.write(f"- {ev}\n")

    # Tạo Markdown file chứa đồ thị Mermaid
    import re
    md_path = os.path.join(os.path.dirname(__file__), "subgraph_preview.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Subgraph Extraction Preview\n\n")
        f.write(f"**Claim**: {claim}\n\n")
        f.write(f"**Ground Truth**: {gt}\n\n")
        f.write(f"**LLM Trả Lời**: {response.answer}\n\n")
        f.write("Dưới đây là các Facts (Tripets) trích xuất được từ Milvus + Neo4j để làm ngữ cảnh cho LLM:\n\n")
        
        if not response.evidences:
            f.write("*Không tìm thấy evidence nào!*")
        else:
            for ev in response.evidences:
                f.write(f"- `{ev}`\n")
                
            f.write("\n## Đồ thị ảo (Mermaid Graph)\n")
            f.write("*(Bạn có thể ấn nút preview Markdown của VS Code để xem đồ thị này)*\n\n")
            f.write("```mermaid\ngraph TD\n")
            for ev in response.evidences:
                # Format mới: [HeadName (Qxxx)] - relation -> [TailName (Qyyy)]
                match = re.search(r'\[(.*?) \((.*?)\)\] - (.*?) -> \[(.*?) \((.*?)\)\]', ev)
                if match:
                    head_name = match.group(1).replace('"', '').replace('(', '').replace(')', '')
                    head_id = match.group(2)
                    rel = match.group(3)
                    tail_name = match.group(4).replace('"', '').replace('(', '').replace(')', '')
                    tail_id = match.group(5)
                    
                    short_head = head_name[:30] + '...' if len(head_name) > 30 else head_name
                    short_tail = tail_name[:30] + '...' if len(tail_name) > 30 else tail_name
                    
                    f.write(f'    {head_id}("{short_head} ({head_id})") -- "{rel}" --> {tail_id}("{short_tail} ({tail_id})")\n')
                else:
                    # Rơi vào trường hợp cũ
                    match2 = re.search(r'\[(.*?) \(ID (.*?)\)\] - (.*?) -> \[(.*?)\]', ev)
                    if match2:
                        name = match2.group(1).replace('"', '').replace('(', '').replace(')', '')
                        head_id = match2.group(2)
                        rel = match2.group(3)
                        tail_id = match2.group(4)
                        short_name = name[:30] + '...' if len(name) > 30 else name
                        f.write(f'    {head_id}("{short_name} ({head_id})") -- "{rel}" --> {tail_id}("{tail_id}")\n')
                    else:
                        # Rơi vào định dạng: [Name] - rel -> [Name]
                        match3 = re.search(r'\[(.*?)\] - (.*?) -> \[(.*?)\]', ev)
                        if match3:
                            head_name = match3.group(1).replace('"', '')
                            rel = match3.group(2)
                            tail_name = match3.group(3).replace('"', '')
                            import hashlib
                            head_id = "N" + hashlib.md5(head_name.encode()).hexdigest()[:8]
                            tail_id = "N" + hashlib.md5(tail_name.encode()).hexdigest()[:8]
                            f.write(f'    {head_id}("{head_name}") -- "{rel}" --> {tail_id}("{tail_name}")\n')
            f.write("```\n")
    
    print(f"📁 Đã xuất Subgraph Context ra file: {md_path}")
    print("Mở file Markdown lên và bật Preview để xem trực quan mạng lưới Neo4j đã cấp cho LLM nhé!")

if __name__ == "__main__":
    test()