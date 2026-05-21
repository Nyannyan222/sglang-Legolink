#!/usr/bin/env python3
"""Evaluate correctness for canonical semantic-positive SGLang replay."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace") if path else ""


def parse_intent(answer: str) -> str:
    try:
        cleaned = answer.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(cleaned)
        intent = str(data.get("intent", "")).strip()
        if intent in {"save_results", "search_budget_limit", "generic_instruction"}:
            return intent
        if "|" in intent:
            facts = " ".join(str(item) for item in data.get("key_facts", []))
            inferred = infer_intent_from_text(facts)
            if inferred:
                return inferred
    except Exception:
        pass
    match = re.search(r'"?intent"?\s*:\s*"?(save_results|search_budget_limit|generic_instruction)"?', answer, re.I)
    if match:
        return match.group(1)
    return infer_intent_from_text(answer)


def infer_intent_from_text(text: str) -> str:
    lowered = text.lower()
    if "searches must not exceed" in lowered or "number of searches" in lowered:
        return "search_budget_limit"
    if "save" in lowered:
        return "save_results"
    if "search" in lowered and ("budget" in lowered or "limit" in lowered):
        return "search_budget_limit"
    return ""


def evidence_preserved(canonical_text: str, expected_intent: str) -> bool:
    lowered = canonical_text.lower()
    if expected_intent == "save_results":
        return "save" in lowered and "<code>" in lowered
    if expected_intent == "search_budget_limit":
        return "search" in lowered and "<num>" in lowered
    return bool(lowered)


def no_hallucinated_facts(canonical_text: str, expected_intent: str) -> bool:
    lowered = canonical_text.lower()
    if expected_intent == "save_results":
        return "/tmp_workspace" not in lowered and "results.md" not in lowered
    if expected_intent == "search_budget_limit":
        return not re.search(r"\b[1-9]\d*\b", lowered.replace("<num>", ""))
    return True


def fmt_num(value: float | None, digits: int = 3) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def evaluate(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_by_id = {row["id"]: row for row in read_jsonl(args.manifest)}
    runtime_rows = read_csv(args.runtime_csv)
    evaluated: list[dict[str, Any]] = []
    for row in runtime_rows:
        manifest = manifest_by_id.get(row.get("manifest_id", ""), {})
        answer = read_text(row.get("answer_path", ""))
        intent = parse_intent(answer)
        expected = manifest.get("expected_intent", "")
        canonical_text = manifest.get("canonical_text", "")
        correct = intent == expected
        preserve = evidence_preserved(canonical_text, expected)
        no_hallucination = no_hallucinated_facts(canonical_text, expected)
        judge_score = 4 if correct and preserve and no_hallucination else (2 if correct else 0)
        evaluated.append(
            {
                **row,
                "source_pair_id": manifest.get("source_pair_id", ""),
                "canonical_key": manifest.get("canonical_key", ""),
                "expected_intent": expected,
                "predicted_intent": intent,
                "answer_equivalence_pass": correct,
                "evidence_preservation_pass": preserve,
                "no_hallucinated_facts_pass": no_hallucination,
                "judge_score_0_to_4": judge_score,
                "answer_preview": answer[:240].replace("\n", " "),
            }
        )

    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evaluated:
        by_pair[row["source_pair_id"]].append(row)
    pair_rows = []
    for pair_id, rows in sorted(by_pair.items()):
        intents = {row["predicted_intent"] for row in rows if row["predicted_intent"]}
        expected = {row["expected_intent"] for row in rows if row["expected_intent"]}
        pair_rows.append(
            {
                "source_pair_id": pair_id,
                "rows": len(rows),
                "expected_intents": ",".join(sorted(expected)),
                "predicted_intents": ",".join(sorted(intents)),
                "answer_equivalence": len(intents) == 1 and intents == expected,
                "evidence_preservation": all(str(row["evidence_preservation_pass"]).lower() == "true" for row in rows),
                "no_hallucinated_facts": all(str(row["no_hallucinated_facts_pass"]).lower() == "true" for row in rows),
                "avg_judge_score": mean(float(row["judge_score_0_to_4"]) for row in rows),
            }
        )
    return evaluated, pair_rows


def condition_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("condition", "")].append(row)
    out = []
    for condition, group in sorted(grouped.items()):
        prompt = [float(row["prompt_tokens"]) for row in group if row.get("prompt_tokens")]
        cached = [float(row["cached_tokens"]) for row in group if row.get("cached_tokens")]
        prefill = [float(row["estimated_prefill_tokens"]) for row in group if row.get("estimated_prefill_tokens")]
        latency = [float(row["latency_s"]) for row in group if row.get("latency_s")]
        prompt_sum = sum(prompt)
        cached_sum = sum(cached)
        out.append(
            {
                "condition": condition,
                "rows": len(group),
                "avg_prompt": mean(prompt) if prompt else None,
                "avg_cached": mean(cached) if cached else None,
                "avg_prefill": mean(prefill) if prefill else None,
                "cached_ratio": cached_sum / prompt_sum if prompt_sum else None,
                "avg_latency": mean(latency) if latency else None,
                "answer_correct": sum(1 for row in group if str(row["answer_equivalence_pass"]).lower() == "true"),
            }
        )
    return out


def render_report(runtime_csv: Path, evaluated: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> str:
    summaries = condition_summary(evaluated)
    return "\n".join(
        [
            "# Canonical Semantic-Positive Correctness And Runtime Report",
            "",
            f"- Runtime CSV: `{runtime_csv.as_posix()}`",
            f"- Rows: {len(evaluated)}",
            f"- Pair groups: {len(pairs)}",
            "",
            "## Baseline Comparison",
            "",
            markdown_table(
                ["condition", "rows", "avg prompt", "avg cached", "avg prefill", "cached ratio", "avg latency s", "answer correct"],
                [
                    [
                        row["condition"],
                        row["rows"],
                        fmt_num(row["avg_prompt"], 1),
                        fmt_num(row["avg_cached"], 1),
                        fmt_num(row["avg_prefill"], 1),
                        fmt_num(row["cached_ratio"], 3),
                        fmt_num(row["avg_latency"], 4),
                        row["answer_correct"],
                    ]
                    for row in summaries
                ],
            ),
            "",
            "## Correctness Validation",
            "",
            markdown_table(
                ["pair", "answer equivalence", "evidence preservation", "no hallucinated facts", "avg judge score"],
                [
                    [
                        row["source_pair_id"],
                        row["answer_equivalence"],
                        row["evidence_preservation"],
                        row["no_hallucinated_facts"],
                        fmt_num(row["avg_judge_score"], 2),
                    ]
                    for row in pairs
                ],
            ),
            "",
            "## Interpretation",
            "",
            "- `original_raw_text_left/right` are different raw texts that share a canonical key.",
            "- `sglang_exact_repeat` is the ordinary SGLang exact-prompt repeat baseline.",
            "- `openclaw_canonical_text` uses the shared canonical text generated from semantic-equivalent templates.",
            "- Correctness is measured as intent equivalence plus evidence preservation and no new concrete facts in the canonical text.",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runtime-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results/canonical_semantic_positive_eval"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluated, pair_rows = evaluate(args)
    write_csv(args.output_dir / "canonical_semantic_correctness_rows.csv", evaluated)
    write_csv(args.output_dir / "canonical_semantic_correctness_pairs.csv", pair_rows)
    (args.output_dir / "canonical_semantic_correctness_report.md").write_text(
        render_report(args.runtime_csv, evaluated, pair_rows),
        encoding="utf-8",
        newline="\n",
    )
    print("Canonical semantic correctness evaluation")
    print("========================================")
    print(f"Rows: {len(evaluated)}")
    print(f"Pairs: {len(pair_rows)}")
    print(f"Report: {args.output_dir / 'canonical_semantic_correctness_report.md'}")


if __name__ == "__main__":
    main()
