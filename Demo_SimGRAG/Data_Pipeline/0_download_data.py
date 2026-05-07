import os
import gdown
import argparse

def download_google_drive_folder(url, output_dir):
    """
    HƯỚNG DẪN HOẠT ĐỘNG:
    Đây là bước đầu tiên (Bước 0) của Data Pipeline.
    Mục đích: Kéo (Download) kho dữ liệu thô khổng lồ (Wikidata5m) từ kho lưu trữ đám mây (Google Drive) 
    xuống ổ cứng cục bộ để chuẩn bị cho quá trình MapReduce.
    
    Yêu cầu: Folder Google Drive phải ở chế độ 'Anyone with the link can view' để gdown truy cập được.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Bắt đầu tải dữ liệu từ Google Drive folder vào: {output_dir}")
    print(f"URL: {url}")
    
    # gdown hỗ trợ tải nguyên folder từ Google Drive
    # Lưu ý: Folder trên Drive phải được set quyền "Anyone with the link"
    gdown.download_folder(url, output=output_dir, quiet=False, use_cookies=False)
    
    print("\nTải dữ liệu hoàn tất!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Link drive mặc định từ bạn cung cấp
    parser.add_argument("--url", default="https://drive.google.com/drive/folders/1F6OlmF3ZsRiKAZ5W2BBpk1jxi3YYjrGg", help="URL thư mục Google Drive")
    parser.add_argument("--output", default="./data/raw/wikidata", help="Thư mục lưu dữ liệu thô tải về")
    args = parser.parse_args()
    
    download_google_drive_folder(args.url, args.output)
