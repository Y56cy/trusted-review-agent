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


# 无证据句改写的失败标记：出现在 LLM 输出中即表示「候选证据不足以支撑，显式未能溯源」。
UNSUPPORTED = "[未能溯源]"

_SENT_END_CHARS = "。！？!?"


def rewrite_sentence(sentence, candidates, server_url="http://127.0.0.1:8080/v1/chat/completions"):
    """依据候选证据改写一句无证据的综述句；无法支撑时返回 UNSUPPORTED。

    只负责「改写」这一件事：把原句 + 候选证据送进 LLM，返回改写后的句子或
    UNSUPPORTED。改写是否真的被证据支持，由调用方再跑一遍 NLI 校验决定，这里
    不自行判断——避免把 LLM 的"自我报告"当作校验结论。
    """
    context = "\n".join(
        f"[{i + 1}] （{c.get('source', '')} 第{c.get('page', '')}页 "
        f"第{c.get('para_id', '')}段）{c.get('text', '')}"
        for i, c in enumerate(candidates)
    )
    prompt = f"""你是一个科研综述辅助助手。下面这句综述在候选证据中没有找到足够支撑：

综述句：{sentence}

候选证据：
{context}

请判断候选证据能否支撑这句综述。
- 如果能支撑，请改写这句话，使其严格依据证据内容、不编造，只输出改写后的句子。
- 如果都不能支撑，请只输出：{UNSUPPORTED}

只输出改写后的句子，或只输出 {UNSUPPORTED}，不要输出任何解释或其他文字。"""

    rewritten = generate(prompt, server_url=server_url).strip()
    if UNSUPPORTED in rewritten:
        return UNSUPPORTED
    rewritten = rewritten.strip("\"'“”‘’").strip()
    if rewritten and rewritten[-1] not in _SENT_END_CHARS:
        rewritten += "。"
    return rewritten