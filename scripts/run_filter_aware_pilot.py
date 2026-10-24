#!/usr/bin/env python3
"""Run the three matched arms sequentially on one externally selected idle MIG."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--head-warmup-epochs", type=int, default=3)
    p.add_argument("--skip-completed", action="store_true")
    p.add_argument("--transfer", action="store_true")
    args = p.parse_args()
    common = [sys.executable, str(ROOT / "notebooks/qwen_filter_aware.py"), "--output-dir", str(args.output_dir)]
    for level in (0, 1, 2):
        folder = args.output_dir / f"level{level}-native-qlora"
        completed = folder / "summary.json"
        if args.skip_completed and completed.exists():
            result = json.loads(completed.read_text())
            if (result["config"]["epochs"] != args.epochs or result["config"]["head_warmup_epochs"] != args.head_warmup_epochs
                    or result["level"] != level or result["numerics"] != "native"):
                raise ValueError("Completed arm does not match the requested pilot")
        else:
            with (args.output_dir / f"level{level}-train.log").open("x") as log:
                subprocess.run(common + ["--stage", "train", "--level", str(level), "--epochs", str(args.epochs),
                             "--head-warmup-epochs", str(args.head_warmup_epochs)], stdout=log, stderr=subprocess.STDOUT, check=True)
        if args.transfer and not (folder / "transfer/summary.json").exists():
            with (args.output_dir / f"level{level}-transfer.log").open("x") as log:
                subprocess.run(common + ["--stage", "transfer", "--level", str(level)], stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    main()
