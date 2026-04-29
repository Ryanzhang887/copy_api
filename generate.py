# make_dataset.py
import json
import math
import random
from typing import Dict, List


def y_sequence_generator(idx: int) -> Dict:
    rng = random.Random(2026 + idx)

    scenario = rng.choice([
        "damped_oscillation",
    ])

    n = rng.randint(200, 300)

    # base signal parameters
    A = rng.uniform(0.5, 3.0)
    #f1 = rng.uniform(0.03, 0.09)  # cycles per sample (discrete frequency)
    #w1 = 2 * math.pi * f1
    p = random.randint(2, 10)
    w1 = 2 * math.pi / p
    gamma = rng.uniform(0.001, 0.01)  # damping per sample
    phi = rng.uniform(0, 2 * math.pi)
    #phi = 0.1
    #noise = rng.uniform(0.01, 0.07)
    noise = 0.01

    ys: List[float] = []

    offset = rng.uniform(-0.4, 0.4)
    for k in range(n):
        y = A * math.sin(w1 * k + phi)
        if rng.random() < 0.05:
            y += rng.gauss(0, noise)

        y += offset
        ys.append(y)

    # convert to text lines (one y per line), with optional missing values
    missing_rate = 0.0
    if scenario == "oscillation_with_missing":
        missing_rate = rng.uniform(0.06, 0.16)

    lines = []
    for y in ys:
        if rng.random() < missing_rate:
            lines.append(rng.choice(["NA", "", "missing", "null"]))
        else:
            # sometimes scientific notation
            lines.append(f"{y:.2f}")

        # # occasionally add extra whitespace
        # if rng.random() < 0.08:
        #     lines[-1] = "   " + lines[-1] + "   "

    description = (
        "This is a repeated physical motion signal. "
        "The data is recorded as a 1D sequence of measurements (one value per sample)."
    )

    requirements = """
Write Python code that:
1) Parses the multi-line TEXT DATA below into a 1D numeric array ys.
2) Creates xs = np.arange(len(ys)).
3) Plots ys vs xs (time-series-like plot).
4) Uses only Python standard library + numpy + matplotlib.
5) Adds title, axis labels, grid, and tight_layout.
Output ONLY valid Python code (no explanations).
""".strip()

    return {
        "scenario": scenario,
        "description": description,
        "data_text": " ".join(lines),
        "requirements": requirements,
        "meta": {
            "scenario": scenario,
            "length": n
        }
    }


def build_prompt(sample: Dict) -> str:
    return f"""
You are given experimental physics data as plain text lines.
There is NO explicit time column.

TEXT DATA (one line):
{sample["data_text"]}

Task:
{sample["requirements"]}

Hard constraints:
- The code must be self-contained and runnable as-is.
- Do not assume external files; parse from the text embedded above.
- Output ONLY Python code.
""".strip()


def build_dataset(num_samples: int = 1000) -> List[Dict]:
    items = []
    for i in range(num_samples):
        s = y_sequence_generator(i)
        prompt = build_prompt(s)
        items.append({
            "id": f"phys_yonly_{i:04d}",
            "input": {"messages": [{"role": "user", "content": prompt}]},
            "meta": s["meta"],
        })
    return items


def save_jsonl(items: List[Dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    dataset = build_dataset(num_samples=30)
    save_jsonl(dataset, "physics_yonly_dataset.jsonl")
    print("Saved: physics_yonly_dataset.jsonl")