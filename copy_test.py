import os
import re
import json
import time
import math
import random
import csv
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI


# =========================
# Configuration
# =========================

API_KEY = "sk-AxHXekyDTXQVXrezCo15wDNgpiR5ffgY1fbaPCP4dLWkck2Z"
BASE_URL = os.environ.get("YUNWU_BASE_URL", "https://yunwu.ai/v1")

# Manually specify the frontier models you want to test.
# Keep only the models that actually exist in your Yunwu account.
# MANUAL_MODEL_LIST = [
#     "gpt-5.4",
#     "claude-opus-4-7",
#     "gemini-3.1-pro-preview",
#     "grok-4.2",
#     "deepseek-v3.2",
#     "qwen3.6-plus",
#     "doubao-seed-2-0-lite-260215",
#     "Kimi-k2.5"
# ]
MANUAL_MODEL_LIST = ["deepseek-v4-pro"]

NUM_SAMPLES_PER_K = 50
#K_VALUES = [7, 8, 9, 10, 11]
K_VALUES = [11]
GLOBAL_SEED = 20260414
DEFAULT_MAX_WORKERS = 8
MODEL_MAX_WORKERS = {
    "grok": 3,
    "Kimi": 5,
}

TEMPERATURE = 0.0
TOP_P = 1.0
REQUEST_TIMEOUT_SECONDS = 2000
MAX_RETRIES = 3
RETRY_SLEEP_SECONDS = 2.0

OUTPUT_DIR = "copy_eval_outputs"

# Debug / logging behavior
# PRINT_ALL_SAMPLES: print every sample to terminal
# PRINT_ONLY_FAILURES: if PRINT_ALL_SAMPLES is False, print only failed samples
PRINT_ALL_SAMPLES = False
PRINT_ONLY_FAILURES = True

# Save the exact chat messages sent to the model
SAVE_MESSAGES = True


# =========================
# Data structures
# =========================

@dataclass
class TrialResult:
    model: str
    k: int
    sample_id: int
    target_length: int
    target: str
    raw_output: str
    parsed_output: str
    exact_match: bool
    strict_match: bool
    latency_sec: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    messages: Optional[List[Dict[str, str]]] = None
    error: Optional[str] = None


# =========================
# Copy string generation
# =========================

# def generate_copy_string(k: int, rng: random.Random) -> str:
#     """
#     Start with s = [random(0,1)].
#     Repeat k times:
#         s -> s + [random(0,1)] + s

#     Final length = 2^(k+1) - 1.
#     """
#     s = [rng.randint(0, 1)]
#     for _ in range(k):
#         b = rng.randint(0, 1)
#         s = s + [b] + s
#         while True:
#             if rng.randint(0, 3) == 0:
#                 p = rng.randint(0, len(s) - 1)
#                 s[p] = 1 - s[p]
#             else:
#                 break

#     return ''.join(map(str, s))
# def generate_copy_string(k: int, rng: random.Random) -> str:
#     """
#     Generate a string of length 2^(k+1) - 1,
#     where each position is independently:
#         0 with probability 0.1
#         1 with probability 0.9
#     """
#     length = 2 ** (k + 1) - 1
#     s = ''.join('1' if rng.random() < 0.95 else '0' for _ in range(length))
#     return s

def generate_copy_string(k: int, rng: random.Random) -> str:
    """
    Start with s = [random(0,1) for _ in range(16)].
    Repeat 2^{k-3} times:
        s -> s + [random(0,1)] + s

    Final length = 2^(k+1) - 1.
    """
    s = [rng.randint(0, 1) for _ in range(16)]
    s = s * 256
    while True:
        if rng.randint(0, 10) == 0:
            break
        else:
            p = rng.randint(0, len(s) - 1)
            s[p] = 1 - s[p]

    return ''.join(map(str, s))


# =========================
# Prompting
# =========================

def build_messages(binary_string: str) -> List[Dict[str, str]]:
    system_prompt = (
        "You are taking a copying test.\n"
        "Your task is to copy the sequence exactly.\n"
        "Output only the copied sequence.\n"
        "Do not add any explanation, quotes, punctuation, or formatting.\n"
        "Keep exactly one space between adjacent symbols.\n"
    )

    spaced_string = " ".join(binary_string)

    user_prompt = (
        "Copy the following sequence exactly:\n\n"
        f"{spaced_string}"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


# =========================
# Output parsing
# =========================

_BINARY_RE = re.compile(r"[01]+")
_SYMBOL_RE = re.compile(r"[01]")

def normalize_output(text: str) -> str:
    """
    Extract all 0/1 symbols from the model output and join them with spaces.

    Example:
        "0 1 0 0 1" -> "0 1 0 0 1"
        "Answer: 0 1 0 0 1." -> "0 1 0 0 1"
    """
    symbols = _SYMBOL_RE.findall(text)
    return " ".join(symbols)


# =========================
# Helpers
# =========================

def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def should_print_sample(exact_match: bool) -> bool:
    if PRINT_ALL_SAMPLES:
        return True
    if PRINT_ONLY_FAILURES and not exact_match:
        return True
    return False


def print_sample_log(
    model: str,
    k: int,
    sample_id: int,
    target: str,
    raw_output: str,
    parsed_output: str,
    strict_match: bool,
    exact_match: bool,
    latency: float,
    usage: Optional[Dict[str, Optional[int]]],
    error: Optional[str] = None,
) -> None:
    print("=" * 100)
    print(f"MODEL={model}")
    print(f"k={k}")
    print(f"sample_id={sample_id}")
    print(f"target_length={len(target)}")
    print(f"latency_sec={latency}")
    if usage is not None:
        print(
            "token_usage="
            f"prompt={usage.get('prompt_tokens')}, "
            f"completion={usage.get('completion_tokens')}, "
            f"total={usage.get('total_tokens')}"
        )
    else:
        print("token_usage=None")

    print("\n--- INPUT / TARGET ---")
    print(target)

    print("\n--- RAW OUTPUT ---")
    print(raw_output)

    print("\n--- PARSED OUTPUT ---")
    print(parsed_output)

    print(f"\nSTRICT={strict_match} EXACT={exact_match}")
    if error is not None:
        print(f"ERROR={error}")
    print()


# =========================
# Client
# =========================

def make_client() -> OpenAI:
    if not API_KEY:
        raise ValueError("Missing API key. Please set YUNWU_API_KEY.")
    return OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


# =========================
# Inference
# =========================

def call_model(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    target_length: int,
) -> Tuple[str, float, Optional[Dict[str, Optional[int]]]]:
    # Allow a bit of extra room in case the model adds unwanted text.
    max_tokens = 50000

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            start = time.perf_counter()

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=max_tokens,
            )

            latency = time.perf_counter() - start
            content = response.choices[0].message.content or ""

            usage = getattr(response, "usage", None)
            if usage is not None:
                usage_dict = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                }
            else:
                usage_dict = None

            return content, latency, usage_dict

        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP_SECONDS * attempt)
            else:
                raise last_err


# =========================
# Evaluation
# =========================

def run_single_sample(
    client: OpenAI,
    model: str,
    k: int,
    sample_id: int,
    base_seed: int,
) -> TrialResult:
    rng = random.Random(base_seed + 100000 * k + sample_id)
    # target = generate_copy_string(k, rng)
    # messages = build_messages(target)
    target_raw = generate_copy_string(k, rng)
    target = " ".join(target_raw)
    messages = build_messages(target_raw)


    try:
        raw_output, latency, usage = call_model(
            client=client,
            model=model,
            messages=messages,
            target_length=len(target),
        )

        parsed_output = normalize_output(raw_output)
        strict_match = (raw_output.strip() == target)
        exact_match = (parsed_output == target)

        if should_print_sample(exact_match):
            print_sample_log(
                model=model,
                k=k,
                sample_id=sample_id,
                target=target,
                raw_output=raw_output,
                parsed_output=parsed_output,
                strict_match=strict_match,
                exact_match=exact_match,
                latency=latency,
                usage=usage,
                error=None,
            )

        return TrialResult(
            model=model,
            k=k,
            sample_id=sample_id,
            target_length=len(target),
            target=target,
            raw_output=raw_output,
            parsed_output=parsed_output,
            exact_match=exact_match,
            strict_match=strict_match,
            latency_sec=latency,
            prompt_tokens=usage["prompt_tokens"] if usage else None,
            completion_tokens=usage["completion_tokens"] if usage else None,
            total_tokens=usage["total_tokens"] if usage else None,
            messages=messages if SAVE_MESSAGES else None,
            error=None,
        )

    except Exception as e:
        error_msg = str(e)

        if should_print_sample(exact_match=False):
            print_sample_log(
                model=model,
                k=k,
                sample_id=sample_id,
                target=target,
                raw_output="",
                parsed_output="",
                strict_match=False,
                exact_match=False,
                latency=math.nan,
                usage=None,
                error=error_msg,
            )

        return TrialResult(
            model=model,
            k=k,
            sample_id=sample_id,
            target_length=len(target),
            target=target,
            raw_output="",
            parsed_output="",
            exact_match=False,
            strict_match=False,
            latency_sec=math.nan,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            messages=messages if SAVE_MESSAGES else None,
            error=error_msg,
        )


def get_max_workers_for_model(model: str) -> int:
    for key, value in MODEL_MAX_WORKERS.items():
        if key.lower() in model.lower():
            return value
    return DEFAULT_MAX_WORKERS


def evaluate_model_k(
    client: OpenAI,
    model: str,
    k: int,
    num_samples: int,
    base_seed: int,
) -> List[TrialResult]:
    max_workers = get_max_workers_for_model(model)
    results: List[TrialResult] = []

    print(f"  using max_workers={max_workers}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                run_single_sample,
                client,
                model,
                k,
                sample_id,
                base_seed,
            )
            for sample_id in range(num_samples)
        ]

        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda r: r.sample_id)
    return results


def summarize(results: List[TrialResult]) -> List[Dict]:
    grouped: Dict[Tuple[str, int], List[TrialResult]] = {}
    for r in results:
        grouped.setdefault((r.model, r.k), []).append(r)

    summary_rows: List[Dict] = []
    for (model, k), group in sorted(grouped.items()):
        total = len(group)
        exact = sum(r.exact_match for r in group)
        strict = sum(r.strict_match for r in group)
        valid_latency = [r.latency_sec for r in group if not math.isnan(r.latency_sec)]

        valid_prompt_tokens = [r.prompt_tokens for r in group if r.prompt_tokens is not None]
        valid_completion_tokens = [r.completion_tokens for r in group if r.completion_tokens is not None]
        valid_total_tokens = [r.total_tokens for r in group if r.total_tokens is not None]

        summary_rows.append({
            "model": model,
            "k": k,
            "target_length": group[0].target_length,
            "num_samples": total,
            "accuracy": exact / total if total else 0.0,
            "strict_accuracy": strict / total if total else 0.0,
            "avg_latency_sec": sum(valid_latency) / len(valid_latency) if valid_latency else None,
            "avg_prompt_tokens": (
                sum(valid_prompt_tokens) / len(valid_prompt_tokens) if valid_prompt_tokens else None
            ),
            "avg_completion_tokens": (
                sum(valid_completion_tokens) / len(valid_completion_tokens)
                if valid_completion_tokens else None
            ),
            "avg_total_tokens": (
                sum(valid_total_tokens) / len(valid_total_tokens) if valid_total_tokens else None
            ),
            "num_errors": sum(r.error is not None for r in group),
        })

    return summary_rows


# =========================
# Save outputs
# =========================

def save_outputs(
    all_results: List[TrialResult],
    summary_rows: List[Dict],
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Save all details in one JSONL
    details_path = os.path.join(output_dir, "details.jsonl")
    with open(details_path, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    # Save summary CSV
    summary_path = os.path.join(output_dir, "summary.csv")
    headers = [
        "model",
        "k",
        "target_length",
        "num_samples",
        "accuracy",
        "strict_accuracy",
        "avg_latency_sec",
        "avg_prompt_tokens",
        "avg_completion_tokens",
        "avg_total_tokens",
        "num_errors",
    ]
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(summary_rows)

    # Save selected models list
    selected_models_path = os.path.join(output_dir, "selected_models.txt")
    with open(selected_models_path, "w", encoding="utf-8") as f:
        for model in MANUAL_MODEL_LIST:
            f.write(f"{model}\n")

    # Save one readable txt file and one jsonl file per model
    grouped_by_model: Dict[str, List[TrialResult]] = {}
    for r in all_results:
        grouped_by_model.setdefault(r.model, []).append(r)

    for model, group in grouped_by_model.items():
        group_sorted = sorted(group, key=lambda x: (x.k, x.sample_id))
        safe_model_name = sanitize_filename(model)

        # Human-readable text log
        model_txt_path = os.path.join(output_dir, f"{safe_model_name}.txt")
        with open(model_txt_path, "w", encoding="utf-8") as f:
            f.write(f"MODEL = {model}\n")
            f.write(f"NUM_RECORDS = {len(group_sorted)}\n")
            f.write("=" * 100 + "\n\n")

            for r in group_sorted:
                f.write(f"MODEL: {r.model}\n")
                f.write(f"k: {r.k}\n")
                f.write(f"sample_id: {r.sample_id}\n")
                f.write(f"target_length: {r.target_length}\n")
                f.write(f"latency_sec: {r.latency_sec}\n")
                f.write(f"prompt_tokens: {r.prompt_tokens}\n")
                f.write(f"completion_tokens: {r.completion_tokens}\n")
                f.write(f"total_tokens: {r.total_tokens}\n")
                f.write(f"strict_match: {r.strict_match}\n")
                f.write(f"exact_match: {r.exact_match}\n")
                f.write(f"error: {r.error}\n")

                if r.messages is not None:
                    f.write("\n--- MESSAGES ---\n")
                    f.write(json.dumps(r.messages, ensure_ascii=False, indent=2) + "\n")

                f.write("\n--- INPUT / TARGET ---\n")
                f.write(r.target + "\n")

                f.write("\n--- RAW OUTPUT ---\n")
                f.write(r.raw_output + "\n")

                f.write("\n--- PARSED OUTPUT ---\n")
                f.write(r.parsed_output + "\n")

                f.write("\n" + "=" * 100 + "\n\n")

        # Machine-readable JSONL per model
        model_jsonl_path = os.path.join(output_dir, f"{safe_model_name}.jsonl")
        with open(model_jsonl_path, "w", encoding="utf-8") as f:
            for r in group_sorted:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


# =========================
# Main
# =========================

def main() -> None:
    print(f"BASE_URL = {BASE_URL}")
    print(f"OUTPUT_DIR = {OUTPUT_DIR}")

    if not MANUAL_MODEL_LIST:
        raise ValueError("MANUAL_MODEL_LIST is empty. Please fill in the models you want to test.")

    print("\nSelected models:")
    for model in MANUAL_MODEL_LIST:
        print("  ", model)

    client = make_client()
    all_results: List[TrialResult] = []

    print("\nRunning evaluation...")
    for model in MANUAL_MODEL_LIST:
        for k in K_VALUES:
            target_length = 2 ** (k + 1) - 1
            print(
                f"Evaluating model={model}, "
                f"k={k}, "
                f"target_length={target_length}, "
                f"samples={NUM_SAMPLES_PER_K}"
            )

            group_results = evaluate_model_k(
                client=client,
                model=model,
                k=k,
                num_samples=NUM_SAMPLES_PER_K,
                base_seed=GLOBAL_SEED,
            )
            all_results.extend(group_results)

            acc = sum(r.exact_match for r in group_results) / len(group_results)
            strict_acc = sum(r.strict_match for r in group_results) / len(group_results)

            valid_prompt_tokens = [r.prompt_tokens for r in group_results if r.prompt_tokens is not None]
            valid_completion_tokens = [r.completion_tokens for r in group_results if r.completion_tokens is not None]
            valid_total_tokens = [r.total_tokens for r in group_results if r.total_tokens is not None]

            avg_prompt_tokens = (
                sum(valid_prompt_tokens) / len(valid_prompt_tokens) if valid_prompt_tokens else None
            )
            avg_completion_tokens = (
                sum(valid_completion_tokens) / len(valid_completion_tokens)
                if valid_completion_tokens else None
            )
            avg_total_tokens = (
                sum(valid_total_tokens) / len(valid_total_tokens) if valid_total_tokens else None
            )

            print(
                f"  accuracy={acc:.4f}, "
                f"strict_accuracy={strict_acc:.4f}, "
                f"avg_prompt_tokens={avg_prompt_tokens}, "
                f"avg_completion_tokens={avg_completion_tokens}, "
                f"avg_total_tokens={avg_total_tokens}"
            )

    print("\nSummarizing...")
    summary_rows = summarize(all_results)

    for row in summary_rows:
        print(
            f"model={row['model']:<30} "
            f"k={row['k']} "
            f"len={row['target_length']:<5} "
            f"acc={row['accuracy']:.4f} "
            f"strict={row['strict_accuracy']:.4f} "
            f"avg_total_tokens={row['avg_total_tokens']} "
            f"errors={row['num_errors']}"
        )

    save_outputs(all_results, summary_rows, OUTPUT_DIR)
    print(f"\nDone. Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()