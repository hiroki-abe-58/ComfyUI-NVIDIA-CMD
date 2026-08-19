from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


QUERY = [
    "nvidia-smi",
    "--query-gpu=memory.used,memory.total,name",
    "--format=csv,noheader,nounits",
]


def snapshot(label: str) -> dict[str, object]:
    raw = subprocess.check_output(QUERY, text=True).strip().splitlines()[0]
    used, total, name = [part.strip() for part in raw.split(",")]
    return {
        "label": label,
        "gpu": name,
        "used_mib": int(used),
        "total_mib": int(total),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Record nvidia-smi VRAM snapshots.")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", default="docs/vram-measurements.jsonl")
    args = parser.parse_args()

    row = snapshot(args.label)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_path = path.with_suffix(".csv")
    new_file = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
