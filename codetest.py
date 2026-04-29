#!/usr/bin/env python3
import json
import os
import re
import subprocess
import sys
import tempfile
from typing import Dict, Any, Tuple

# =========================
# PUT YOUR JSONL PATH HERE
# =========================
JSONL_PATH = r"repeat01_qwen3-next-80b-a3b-instruct.jsonl"

# =========================
# OUTPUT DIR FOR RUN RESULTS
# =========================
OUTPUT_DIR = r"./run_outputs"  # will be created if not exists


EXPECT_NEWLINE_SEPARATED = True


def extract_text_string(prompt: str) -> str:
    m = re.search(r"TEXT STRING\s*\(length\s*=\s*\d+\)\s*:\s*\n(.*?)\n\nTask:", prompt, flags=re.S)
    if not m:
        raise ValueError("Could not find TEXT STRING block in prompt.")
    return m.group(1).strip()


def strip_code_fences(code: str) -> str:
    code = code.strip()
    m = re.match(r"^```(?:python)?\s*\n(.*)\n```$", code, flags=re.S | re.I)
    if m:
        return m.group(1)
    return code


def safe_filename(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    return s[:200] if len(s) > 200 else s


def run_code_capture_stdout(py_code: str, timeout_sec: float = 10.0) -> Tuple[int, str, str]:
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "prog.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(py_code)

        p = subprocess.run(
            [sys.executable, path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
        )
        return p.returncode, p.stdout, p.stderr


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def get_prompt_and_code(item: Dict[str, Any]) -> Tuple[str, str]:
    prompt = item["input"]["messages"][0]["content"]
    code = item["output"]["assistant_text"]
    return prompt, code


def build_expected_output(s: str, n: int = 100) -> str:
    if EXPECT_NEWLINE_SEPARATED:
        # 100 lines, each line exactly s, ending with a trailing newline
        # (matches typical: for _ in range(100): print(s))
        return (s + "\n") * n
    else:
        return s * n


def main():
    if not os.path.exists(JSONL_PATH):
        print(f"ERROR: JSONL_PATH does not exist: {JSONL_PATH}", file=sys.stderr)
        sys.exit(2)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total = 0
    passed = 0

    for line_no, item in load_jsonl(JSONL_PATH):
        total += 1

        item_id = item.get("id", f"line_{line_no}")
        meta = item.get("meta", {})
        idx = meta.get("idx", None)

        base_name = safe_filename(f"{item_id}_line{line_no}_idx{idx}")

        try:
            prompt, code_raw = get_prompt_and_code(item)
            expected_s = extract_text_string(prompt)
            expected_out = build_expected_output(expected_s, 100)

            py_code = strip_code_fences(code_raw)

            rc, out, err = run_code_capture_stdout(py_code)

            code_path = os.path.join(OUTPUT_DIR, base_name + ".py")
            out_path = os.path.join(OUTPUT_DIR, base_name + ".stdout.txt")
            err_path = os.path.join(OUTPUT_DIR, base_name + ".stderr.txt")

            with open(code_path, "w", encoding="utf-8") as f:
                f.write(py_code)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(out)
            with open(err_path, "w", encoding="utf-8") as f:
                f.write(err)

            ok = (rc == 0) and (out == expected_out)

            if ok:
                passed += 1
                print(f"[PASS] {item_id} (line {line_no}, idx={idx}) -> saved to {OUTPUT_DIR}/")
            else:
                print(f"[FAIL] {item_id} (line {line_no}, idx={idx}) -> saved to {OUTPUT_DIR}/")
                print(f"  returncode: {rc}")
                print(f"  expected stdout len: {len(expected_out)}")
                print(f"  actual stdout len:   {len(out)}")

                # helpful mismatch info
                exp_len = len(expected_out)
                out_len = len(out)
                n = min(exp_len, out_len)
                mpos = None
                for i in range(n):
                    if out[i] != expected_out[i]:
                        mpos = i
                        break
                if mpos is None and exp_len != out_len:
                    mpos = n
                if mpos is not None:
                    a = max(0, mpos - 40)
                    b = mpos + 40
                    print(f"  first mismatch at pos {mpos}")
                    print(f"  expected snippet: {repr(expected_out[a:b])}")
                    print(f"  actual snippet:   {repr(out[a:b])}")

                if err:
                    print("  stderr (first 300 chars):")
                    print("  " + err[:300].replace("\n", "\n  "))

        except subprocess.TimeoutExpired:
            print(f"[FAIL] {item_id} (line {line_no}, idx={idx}) -> timeout")
        except Exception as e:
            print(f"[FAIL] {item_id} (line {line_no}, idx={idx})")
            print(f"  error: {type(e).__name__}: {e}")

    print(f"\nSummary: {passed}/{total} passed")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
