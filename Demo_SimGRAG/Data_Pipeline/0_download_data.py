import os
import gdown
import argparse

def download_google_drive_folder(url, output_dir):
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
