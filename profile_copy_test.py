import argparse
import statistics
import time
from typing import Callable, List, Tuple

import copy_test


def bench(name: str, fn: Callable[[], None], repeats: int) -> Tuple[str, float, float, float]:
    costs: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        costs.append(time.perf_counter() - t0)
    return name, statistics.mean(costs), min(costs), max(costs)


def run_local_benchmarks(k: int, repeats: int) -> None:
    rng = copy_test.random.Random(copy_test.GLOBAL_SEED)
    target_raw = copy_test.generate_copy_string(k, rng)
    messages = copy_test.build_messages(target_raw)

    # Build deterministic reusable output for normalize benchmark.
    fake_output = " ".join(target_raw)

    tests = [
        ("generate_copy_string", lambda: copy_test.generate_copy_string(k, rng)),
        ("build_messages", lambda: copy_test.build_messages(target_raw)),
        ("normalize_output", lambda: copy_test.normalize_output(fake_output)),
    ]

    print("\n=== Local CPU benchmarks ===")
    print(f"k={k}, repeats={repeats}, target_len={len(target_raw)}")
    for name, fn in tests:
        n, avg_s, min_s, max_s = bench(name, fn, repeats)
        print(f"{n:<22} avg={avg_s*1000:.3f}ms min={min_s*1000:.3f}ms max={max_s*1000:.3f}ms")

    print("\n=== One-sample end-to-end (without API) ===")
    t0 = time.perf_counter()
    _ = copy_test.normalize_output(fake_output)
    elapsed = time.perf_counter() - t0
    print(f"local_parse_only={elapsed*1000:.3f}ms")

    # avoid lint-like unused variables
    _ = messages


def run_api_probe(model: str, k: int, repeats: int) -> None:
    print("\n=== API probe ===")
    client = copy_test.make_client()
    latencies: List[float] = []

    for i in range(repeats):
        rng = copy_test.random.Random(copy_test.GLOBAL_SEED + i)
        target_raw = copy_test.generate_copy_string(k, rng)
        messages = copy_test.build_messages(target_raw)
        target = " ".join(target_raw)

        _, latency, usage = copy_test.call_model(
            client=client,
            model=model,
            messages=messages,
            target_length=len(target),
        )
        latencies.append(latency)
        print(
            f"sample={i} latency={latency:.3f}s "
            f"prompt_tokens={usage['prompt_tokens'] if usage else None} "
            f"completion_tokens={usage['completion_tokens'] if usage else None}"
        )

    print(
        f"API latency summary: avg={statistics.mean(latencies):.3f}s "
        f"min={min(latencies):.3f}s max={max(latencies):.3f}s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile copy_test.py bottlenecks")
    parser.add_argument("--k", type=int, default=11)
    parser.add_argument("--local-repeats", type=int, default=200)
    parser.add_argument("--api-repeats", type=int, default=0, help="set >0 to actually call API")
    parser.add_argument("--model", type=str, default=(copy_test.MANUAL_MODEL_LIST[0] if copy_test.MANUAL_MODEL_LIST else ""))
    args = parser.parse_args()

    run_local_benchmarks(k=args.k, repeats=args.local_repeats)

    if args.api_repeats > 0:
        if not args.model:
            raise ValueError("--model is required when --api-repeats > 0")
        run_api_probe(model=args.model, k=args.k, repeats=args.api_repeats)


if __name__ == "__main__":
    main()
