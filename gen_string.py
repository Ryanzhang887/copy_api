# make_repeat01_dataset.py
import json
import random
from typing import Dict, List, Any
import string


def bitstring_generator(idx: int) -> Dict[str, Any]:
    rng = random.Random(2026 + idx)

    # 1) tokens: digits + English letters
    vocab = string.digits + string.ascii_lowercase + string.ascii_uppercase  # 62 tokens

    period = rng.randint(6, 12)
    length = rng.randint(300, 400)
    flips = 10

    # base periodic pattern
    base = [rng.choice(vocab) for _ in range(period)]

    # repeat to target length
    s_list = (base * (1 + length // period))[:length]

    # choose flip positions in s_list
    flip_positions = rng.sample(range(length), flips)

    # ---------------------------
    # Design 1: replace using ALL tokens in vocab
    # ---------------------------
    replace_pool = list(vocab)

    # ---------------------------
    # Design 2: replace using ONLY tokens appearing in base
    # ---------------------------
    #replace_pool = list(set(base))
    # (optional) make sure pool size > 1 to allow change; otherwise fall back to global vocab

    for p in flip_positions:
        old = s_list[p]
        new = rng.choice(replace_pool)
        while new == old:
            new = rng.choice(replace_pool)
        s_list[p] = new

    # 2) each two items separated by a space
    s = " ".join(s_list)
    
    requirements = f"""
Write Python code that:
1) Implements a function `gen_string(idx: int) -> str` that deterministically generates s EXACTLY the same string as TEXT STRING above.
2) Prints s repeated 100 times, one per line, using a for loop (i.e., use print(s) so outputs are separated by \n).
3) Use only Python standard library.
Output ONLY valid Python code (no explanations).
""".strip()

    prompt = f"""
You are given a text string below.

TEXT STRING (length = {length}):
{s}

Task:
{requirements}

Hard constraints:
- The code must be self-contained and runnable as-is.
- Do not assume external files.
- Output ONLY Python code.
""".strip()

    return {
        "id": f"repeat01_{idx:04d}",
        "input": {"messages": [{"role": "user", "content": prompt}]},
        "meta": {
            "idx": idx,
            "period": period,
            "length": length,
            "flips": flips,
        },
    }


def build_dataset(num_samples: int = 30) -> List[Dict[str, Any]]:
    return [bitstring_generator(i) for i in range(num_samples)]


def save_jsonl(items: List[Dict[str, Any]], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    dataset = build_dataset(num_samples=30)
    save_jsonl(dataset, "repeat01_dataset.jsonl")
    print("Saved: repeat01_dataset.jsonl")