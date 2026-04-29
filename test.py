import json
import re


def extract_text_data_from_user_prompt(user_prompt: str) -> str:
    """
    Extract ground-truth TEXT DATA block from the dataset prompt.
    Pattern:
        TEXT DATA (...):
        <data...>
        Task:
    """
    m = re.search(
        r"TEXT DATA\s*\(.*?\)\s*:\s*\n(.*?)\n\s*Task\s*:",
        user_prompt,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return m.group(1).strip() if m else ""


def extract_text_data_from_assistant_code(assistant_code: str) -> str:
    """
    Extract text_data = "..." or text_data = '''...''' from assistant code.
    """
    m = re.search(
        r'base\s*=\s*(?P<q>["\']{1,3})(?P<s>.*?)(?P=q)',
        assistant_code,
        flags=re.DOTALL,
    )
    return m.group("s") if m else ""


def normalize_float_string(s: str) -> str:
    """
    Extract all floats like -2.10, 0.35 (exactly 2 decimals) and normalize.
    """
    nums = re.findall(r"[-+]?\d+\.\d{2}", s)
    return " ".join(nums)


def main(
    input_path: str = "epeat01_qwen3-next-80b-a3b-instruct.jsonl",
    output_path: str = "physics_yonly_extracted_text_data.jsonl",
):
    total = 0
    ok = 0
    bad = 0

    with open(input_path, "r", encoding="utf-8") as f_in, open(
        output_path, "w", encoding="utf-8"
    ) as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue

            total += 1
            item = json.loads(line)

            sid = item.get("id", f"row_{total:06d}")

            # ---- get user prompt ----
            messages = item.get("input", {}).get("messages", [])
            user_prompt = ""
            for msg in messages:
                if msg.get("role") == "user":
                    user_prompt = msg.get("content", "")
                    break

            # ---- get assistant output ----
            assistant_code = (
                item.get("output", {}).get("assistant_text", "")
                or item.get("output", {}).get("raw_response", {})
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )

            gt_raw = extract_text_data_from_user_prompt(user_prompt)
            pred_raw = extract_text_data_from_assistant_code(assistant_code)

            gt_norm = normalize_float_string(gt_raw)
            pred_norm = normalize_float_string(pred_raw)

            match = (gt_norm == pred_norm)

            if match:
                ok += 1
            else:
                bad += 1

            record = {
                "id": sid,
                "match": match,
                "gt_num_floats": len(gt_norm.split()),
                "pred_num_floats": len(pred_norm.split()),
                "gt_text_data_raw": gt_raw,
                "pred_text_data_raw": pred_raw,
                "gt_text_data_norm": gt_norm,
                "pred_text_data_norm": pred_norm,
            }

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print(f"Total : {total}")
    print(f"Match : {ok}")
    print(f"Bad   : {bad}")


if __name__ == "__main__":
    main()