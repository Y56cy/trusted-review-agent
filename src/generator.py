import requests


def generate(prompt, server_url="http://127.0.0.1:8080/v1/chat/completions"):
    """把 prompt 发给本地 llama-server，返回生成的文本"""
    response = requests.post(
        server_url,
        json={
            "model": "qwen",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 512
        }
    )
    result = response.json()
    return result["choices"][0]["message"]["content"]