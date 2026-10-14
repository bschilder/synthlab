"""Stream fixed independent recipes, retaining failures in an immutable journal.

Each JSONL line is {"case_id": "safe-unique-name", "recipe": <single recipe>}.
Use the same isolated simulation environment as generate_association_bundle.py.
There is no retry, reseeding, statistical selection or association-code import.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from contextlib import ExitStack
from pathlib import Path

from generate_association_bundle import load_recipe_writer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipes", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--max-recipe-bytes", type=int, default=16 * 1024**2)
    parser.add_argument("--max-output-bytes", type=int, default=256 * 1024**2)
    args = parser.parse_args()
    if args.max_recipe_bytes < 1 or args.max_output_bytes < 1:
        raise ValueError("positive input/output byte bounds required")
    if re.fullmatch(r"[0-9a-f]{40}", args.source_revision) is None:
        raise ValueError("full lowercase source revision required")
    producer = load_recipe_writer()
    args.destination.mkdir(mode=0o700)
    seen: set[str] = set()
    attempted, failed, complete_input = 0, 0, True
    input_digest, input_bytes = hashlib.sha256(), 0
    with ExitStack() as stack:
        source = stack.enter_context(args.recipes.open("rb"))
        journal = stack.enter_context((args.destination / "records.jsonl").open("x"))
        retained_input = stack.enter_context((args.destination / "recipes.jsonl").open("xb"))
        while line := source.readline(args.max_recipe_bytes + 1):
            retained_input.write(line)
            retained_input.flush()
            os.fsync(retained_input.fileno())
            input_digest.update(line)
            input_bytes += len(line)
            attempted += 1
            line_complete = len(line) <= args.max_recipe_bytes
            record = {
                "line_number": attempted,
                "case_id": None,
                "input_line_complete": line_complete,
            }
            record["input_line_sha256" if line_complete else "input_prefix_sha256"] = (
                hashlib.sha256(line).hexdigest()
            )
            try:
                if len(line) > args.max_recipe_bytes:
                    complete_input = False
                    raise ValueError("recipe line byte bound exceeded; remaining input unattempted")
                entry = producer.load_recipe_json(line)
                if not isinstance(entry, dict) or set(entry) != {"case_id", "recipe"}:
                    raise ValueError("case_id and recipe required for every line")
                identifier = entry["case_id"]
                if (
                    not isinstance(identifier, str)
                    or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", identifier) is None
                ):
                    raise ValueError("case_id must be a safe directory name")
                record["case_id"] = identifier
                if identifier in seen:
                    raise ValueError("duplicate registered case_id")
                seen.add(identifier)
                receipt = producer.write_association_recipe(
                    args.destination / identifier,
                    entry["recipe"],
                    source_revision=args.source_revision,
                    max_output_bytes=args.max_output_bytes,
                )
                record.update(status="success", receipt=receipt)
            except Exception as error:  # noqa: BLE001 - Retain every failed registered case.
                failed += 1
                record.update(
                    status="failed",
                    error_type=type(error).__name__,
                    error_message=str(error)[:2048],
                )
            serialized = json.dumps(record, sort_keys=True, allow_nan=False)
            journal.write(serialized + "\n")
            journal.flush()
            os.fsync(journal.fileno())
            print(serialized, flush=True)
            if not complete_input:
                break
    completion = {
        "schema_version": 0,
        "source_revision": args.source_revision,
        "attempted": attempted,
        "successful": attempted - failed,
        "failed": failed,
        "complete_input": complete_input,
        "consumed_input_sha256": input_digest.hexdigest(),
        "consumed_input_bytes": input_bytes,
        "scope": "fixed recipes; all observed failures retained; not association calibration",
    }
    (args.destination / "completed.json").write_text(
        json.dumps(completion, sort_keys=True, indent=2) + "\n"
    )
    raise SystemExit(1 if failed or not complete_input else 0)


if __name__ == "__main__":
    main()
