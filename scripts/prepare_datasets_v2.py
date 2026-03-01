#!/usr/bin/env python3
"""Prepare cycle-2 training/evaluation datasets with stronger data hygiene.

Outputs:
- datasets_v2/unique_prompts_train_v2.json
- datasets_v2/unique_prompts_eval_hard.json
- datasets_v2/DATASET_V2_REPORT.md
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare curated train/eval datasets for cycle-2.")
    parser.add_argument("--balanced", type=Path, default=Path("datasets/unique_prompts_balanced.json"))
    parser.add_argument("--clean", type=Path, default=Path("datasets/unique_prompts_clean.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("datasets_v2"))
    parser.add_argument("--train-source-label-cap", type=int, default=220)
    parser.add_argument("--eval-source-label-cap", type=int, default=300)
    parser.add_argument("--eval-max-per-label", type=int, default=600)
    return parser.parse_args()


def normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def normalize_label(raw: Any) -> str:
    value = str(raw).strip().lower()
    if value in {"malicious", "attack", "unsafe", "harmful"}:
        return "malicious"
    return "benign"


def row_hash(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()


def has_placeholder(text: str) -> bool:
    upper = text.upper()
    return any(token in upper for token in ("[INSERT", "<INSERT", "PLACEHOLDER", "YOUR_TASK"))


def is_aim_template(text: str) -> bool:
    upper = text.upper()
    return "NICCOLO" in upper and "AIM" in upper


def has_encoding_marker(text: str) -> bool:
    return re.search(r"\b(BASE64|ROT13|ENCODE|DECODE)\b", text.upper()) is not None


def is_base64_like_whole_prompt(text: str) -> bool:
    compact = text.replace("\n", "").strip()
    return len(compact) > 24 and re.fullmatch(r"[A-Za-z0-9+/=]+", compact) is not None


def is_tiny_outlier(text: str) -> bool:
    return len(text.split()) < 3 and len(text) < 20


def stable_take(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    rows = sorted(rows, key=lambda r: row_hash(r["text"], str(r.get("source", "")), r["label"]))
    return rows[:n]


def dedup_by_text(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        text = row["text"]
        if text in seen:
            continue
        seen.add(text)
        out.append(row)
    return out


def group_cap(rows: list[dict[str, Any]], cap: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        source = str(row.get("source", "unknown"))
        label = row["label"]
        groups[(source, label)].append(row)
    out: list[dict[str, Any]] = []
    for key, group_rows in groups.items():
        del key
        out.extend(stable_take(group_rows, cap))
    return out


def load_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON list.")
    out: list[dict[str, Any]] = []
    for record in data:
        if not isinstance(record, dict):
            continue
        text = normalize_text(record.get("text", ""))
        if not text:
            continue
        row = dict(record)
        row["text"] = text
        row["label"] = normalize_label(record.get("label", "benign"))
        row.setdefault("category", "unknown")
        row.setdefault("source", "unknown")
        out.append(row)
    return out


def build_train_v2(rows_balanced: list[dict[str, Any]], train_cap: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    filtered: list[dict[str, Any]] = []
    dropped = collections.Counter()

    for row in dedup_by_text(rows_balanced):
        text = row["text"]
        if has_placeholder(text):
            dropped["placeholder"] += 1
            continue
        if is_aim_template(text):
            dropped["aim_template"] += 1
            continue
        if is_base64_like_whole_prompt(text):
            dropped["base64_like"] += 1
            continue
        if is_tiny_outlier(text):
            dropped["tiny_outlier"] += 1
            continue
        filtered.append(row)

    capped = group_cap(filtered, train_cap)
    by_label: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in capped:
        by_label[row["label"]].append(row)
    n = min(len(by_label["benign"]), len(by_label["malicious"]))
    train = stable_take(by_label["benign"], n) + stable_take(by_label["malicious"], n)
    train = sorted(train, key=lambda r: row_hash(r["text"], r["label"], str(r.get("source", ""))))
    return train, dict(dropped)


def build_eval_hard(
    rows_clean: list[dict[str, Any]],
    train_texts: set[str],
    eval_cap: int,
    eval_max_per_label: int,
) -> list[dict[str, Any]]:
    hard_rows: list[dict[str, Any]] = []
    hard_categories = {"jailbreak", "prompt_hijacking", "prompt_extraction", "prompt_injection"}
    hard_sources = {"tensor_trust", "jackhhao", "bipia"}

    for row in dedup_by_text(rows_clean):
        text = row["text"]
        if text in train_texts:
            continue
        category = str(row.get("category", ""))
        source = str(row.get("source", ""))
        is_hard = (
            has_placeholder(text)
            or is_aim_template(text)
            or has_encoding_marker(text)
            or is_base64_like_whole_prompt(text)
            or len(text.split()) >= 180
            or category in hard_categories
            or source in hard_sources
        )
        if is_hard:
            hard_rows.append(row)

    capped = group_cap(hard_rows, eval_cap)
    by_label: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in capped:
        by_label[row["label"]].append(row)
    n = min(len(by_label["benign"]), len(by_label["malicious"]), eval_max_per_label)
    eval_rows = stable_take(by_label["benign"], n) + stable_take(by_label["malicious"], n)
    eval_rows = sorted(eval_rows, key=lambda r: row_hash(r["text"], r["label"], str(r.get("source", ""))))
    return eval_rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = collections.Counter(row["label"] for row in rows)
    sources = collections.Counter(str(row.get("source", "unknown")) for row in rows)
    categories = collections.Counter(str(row.get("category", "unknown")) for row in rows)
    return {
        "rows": len(rows),
        "labels": dict(labels),
        "top_sources": sources.most_common(12),
        "top_categories": categories.most_common(12),
    }


def write_report(path: Path, train_summary: dict[str, Any], eval_summary: dict[str, Any], dropped: dict[str, int]) -> None:
    content = [
        "# Dataset V2 Report",
        "",
        "## Train V2",
        f"- Rows: `{train_summary['rows']}`",
        f"- Labels: `{train_summary['labels']}`",
        f"- Top sources: `{train_summary['top_sources']}`",
        f"- Top categories: `{train_summary['top_categories']}`",
        "",
        "## Eval Hard",
        f"- Rows: `{eval_summary['rows']}`",
        f"- Labels: `{eval_summary['labels']}`",
        f"- Top sources: `{eval_summary['top_sources']}`",
        f"- Top categories: `{eval_summary['top_categories']}`",
        "",
        "## Dropped During Train V2 Prep",
        f"- `{dropped}`",
        "",
        "## Notes",
        "- Train v2 removes placeholder/template artifacts and tiny outliers.",
        "- Train v2 is balanced by label and capped by source+label to reduce shortcut learning.",
        "- Eval hard emphasizes difficult attack styles and outlier formats.",
    ]
    path.write_text("\n".join(content) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows_balanced = load_rows(args.balanced)
    rows_clean = load_rows(args.clean)

    train_rows, dropped = build_train_v2(rows_balanced, args.train_source_label_cap)
    train_texts = {row["text"] for row in train_rows}
    eval_rows = build_eval_hard(rows_clean, train_texts, args.eval_source_label_cap, args.eval_max_per_label)

    train_path = args.out_dir / "unique_prompts_train_v2.json"
    eval_path = args.out_dir / "unique_prompts_eval_hard.json"
    report_path = args.out_dir / "DATASET_V2_REPORT.md"

    train_path.write_text(json.dumps(train_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    eval_path.write_text(json.dumps(eval_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    train_summary = summarize(train_rows)
    eval_summary = summarize(eval_rows)
    write_report(report_path, train_summary, eval_summary, dropped)

    print(f"Wrote {train_path} ({train_summary['rows']} rows)")
    print(f"Wrote {eval_path} ({eval_summary['rows']} rows)")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
