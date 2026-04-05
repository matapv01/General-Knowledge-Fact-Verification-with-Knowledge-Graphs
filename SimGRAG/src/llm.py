from openai import OpenAI


class LLM:
    def __init__(self, configs):
        self.configs = configs['llm']
        print('Loading LLM ...')
        self.client = OpenAI(
            base_url = self.configs['base_url'],
            api_key = self.configs['api_key'],
            timeout = 120.0
        )
        print(self.chat("Hello!"))

    def chat(self, input_text):
        import time
        while True:
            try:
                completion = self.client.chat.completions.create(
                    model=self.configs['model'],
                    messages=[{"role":"user", "content":input_text}],
                    temperature=self.configs['temperature'],
                    top_p=self.configs['top_p'],
                    max_tokens=self.configs['max_tokens'],
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}}
                )
                time.sleep(1.6) # Tránh Rate Limit 40 RPM của API NVIDIA
                content = completion.choices[0].message.content
                if content is None:
                    content = ""
                return content.strip()
            except Exception as e:
                print(f"[CẢNH BÁO LỖI API] Đang treo 10s để hệ thống NVIDIA nhả Limit: {e}")
                time.sleep(10)