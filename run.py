import os
import json
import time
import requests
from typing import Dict, List, Any

API_URL = "https://cloud.infini-ai.com/maas/v1/chat/completions"
MODEL_NAME = "qwen3-next-80b-a3b-instruct"
#MODEL_NAME = "gpt-oss-120b"

API_KEY = "sk-ypdpbq7ktndb7qhs"
if not API_KEY:
    raise RuntimeError("Please set environment variable API_KEY first.")

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# 你可以调这两个参数
TEMPERATURE = 0.0
MAX_TOKENS = 2000
PRICE_IN_PER_1M = 0.0
PRICE_OUT_PER_1M = 0.0

# 简单限速，避免触发 rate limit
SLEEP_SEC = 0.25


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def call_chat_completions(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "reasoning": {"effort": "none"},
    }
    resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def extract_text(raw: Dict[str, Any]) -> str:
    """
    OpenAI-style response:
      raw["choices"][0]["message"]["content"]
    If provider changes schema, fallback to dump.
    """
    try:
        return raw["choices"][0]["message"]["content"]
    except Exception:
        return json.dumps(raw, ensure_ascii=False)


def main(
    dataset_path: str = "repeat01_dataset.jsonl",
    output_path: str = f"repeat01_{MODEL_NAME}.jsonl",
):
    dataset = load_jsonl(dataset_path)
    print(f"Loaded {len(dataset)} samples from {dataset_path}")

    with open(output_path, "w", encoding="utf-8") as out_f:
        for i, item in enumerate(dataset):
            sid = item.get("id", f"row_{i:06d}")
            messages = item["input"]["messages"]

            print(f"[{i+1}/{len(dataset)}] calling API: {sid}")

            record = {
                "id": sid,
                "meta": item.get("meta", {}),
                "input": item["input"],
            }

            try:
                raw = call_chat_completions(messages)
                content = extract_text(raw)

                # 存原始返回（便于以后debug），以及抽取出来的文本
                record["output"] = {
                    "assistant_text": content,
                    "raw_response": raw,
                }
                record["status"] = "ok"

            except Exception as e:
                record["status"] = "error"
                record["error"] = str(e)

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

            time.sleep(SLEEP_SEC)

    print(f"Saved outputs to {output_path}")


if __name__ == "__main__":
    main()