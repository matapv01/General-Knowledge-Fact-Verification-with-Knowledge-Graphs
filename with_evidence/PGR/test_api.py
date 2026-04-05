"""
Script test đơn giản để kiểm tra API Qwen (NVIDIA)
Chạy: python test_api.py
"""

import requests
import json

# ============================================================
# CẤU HÌNH: Thay đổi ở đây nếu cần
# ============================================================
API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
API_KEY = "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"
MODEL   = "qwen/qwen3-next-80b-a3b-instruct"     # Đổi model tại đây nếu muốn thử model khác

# ============================================================
# NỘI DUNG CÂU HỎI TEST
# ============================================================
USER_MESSAGE = "Who is the president of the United States in 2024?"

# ============================================================
# GỌI API
# ============================================================
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

data = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": USER_MESSAGE}
    ],
    "stream": False,
    "max_tokens": 512,
    "temperature": 0.2,
    "top_p": 0.1,
    "chat_template_kwargs": {"enable_thinking": False}
}

print(f"🔗 Endpoint : {API_URL}")
print(f"🤖 Model    : {MODEL}")
print(f"💬 Question : {USER_MESSAGE}")
print("-" * 60)
print("⏳ Đang gọi API...")

try:
    resp = requests.post(API_URL, headers=headers, json=data, timeout=60)
    resp_json = resp.json()

    if resp.status_code != 200:
        print(f"❌ HTTP Error {resp.status_code}")
        print(json.dumps(resp_json, indent=2, ensure_ascii=False))
    elif "error" in resp_json:
        print(f"❌ API Error: {resp_json['error']}")
    else:
        content = resp_json["choices"][0]["message"]["content"]
        usage   = resp_json.get("usage", {})
        print(f"✅ Phản hồi:\n{content}")
        print("-" * 60)
        print(f"📊 Tokens dùng — Prompt: {usage.get('prompt_tokens','?')} | "
              f"Completion: {usage.get('completion_tokens','?')} | "
              f"Total: {usage.get('total_tokens','?')}")

except requests.exceptions.Timeout:
    print("❌ Timeout: API không phản hồi trong 60 giây.")
except requests.exceptions.ConnectionError as e:
    print(f"❌ Lỗi kết nối: {e}")
except Exception as e:
    print(f"❌ Lỗi không xác định: {e}")
