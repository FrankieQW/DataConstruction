from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot aligned TokenLight train/validation logs without GPU dependencies.")
    parser.add_argument("run_directories", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--window", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.window < 1:
        raise ValueError("--window 必须大于等于 1")
    train = merge_records(args.run_directories, "train.jsonl")
    validation = merge_records(args.run_directories, "validation.jsonl")
    checkpoints = merge_checkpoint_records(args.run_directories)
    if not train:
        raise ValueError("没有找到 train.jsonl 记录")
    args.output.mkdir(parents=True, exist_ok=True)
    export_csv(args.output / "merged_loss.csv", train, validation)

    import matplotlib.pyplot as plt

    plot_lines(plt, train, [("loss_total", "train")], args.output / "loss_total_raw.png", checkpoints)
    smoothed = [dict(record, loss_total=moving_average(train, index, "loss_total", args.window)) for index, record in enumerate(train)]
    plot_lines(plt, smoothed, [("loss_total", f"train MA({args.window})")], args.output / "loss_total_smoothed.png", checkpoints)
    task_keys = [(key, key.removeprefix("loss_")) for key in ("loss_ambient", "loss_diffuse", "loss_add_light", "loss_in_scene")]
    plot_lines(plt, train, task_keys, args.output / "loss_by_task.png", checkpoints)
    validation_series = [("val/loss_total", "validation")]
    plot_lines(plt, validation, validation_series, args.output / "validation_loss.png", checkpoints)
    plot_lines(plt, train, [("learning_rate", "learning rate")], args.output / "learning_rate.png", checkpoints)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def merge_records(directories: list[Path], filename: str) -> list[dict[str, Any]]:
    by_step: dict[int, dict[str, Any]] = {}
    for directory in directories:
        for record in read_jsonl(directory / filename):
            by_step[int(record["global_step"])] = record
    return [by_step[step] for step in sorted(by_step)]


def merge_checkpoint_records(directories: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for directory in directories:
        records.extend(read_jsonl(directory / "checkpoints" / "index.jsonl"))
    return records


def moving_average(records: list[dict[str, Any]], index: int, key: str, window: int) -> float | None:
    values = [record.get(key) for record in records[max(0, index - window + 1) : index + 1]]
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else None


def plot_lines(plt, records, series, output: Path, checkpoints) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    has_series = False
    for key, label in series:
        points = [(int(record["global_step"]), record.get(key)) for record in records if record.get(key) is not None]
        if points:
            axis.plot([point[0] for point in points], [float(point[1]) for point in points], label=label)
            has_series = True
    for checkpoint in checkpoints:
        axis.axvline(int(checkpoint["global_step"]), color="gray", alpha=0.15, linewidth=0.8)
        if checkpoint.get("is_best"):
            axis.axvline(int(checkpoint["global_step"]), color="green", alpha=0.5, linewidth=1.2)
    axis.set_xlabel("global step")
    axis.grid(alpha=0.2)
    if has_series:
        axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def export_csv(path: Path, train: list[dict[str, Any]], validation: list[dict[str, Any]]) -> None:
    validation_by_step = {int(record["global_step"]): record for record in validation}
    fields = [
        "global_step", "loss_total", "loss_ambient", "loss_diffuse", "loss_add_light", "loss_in_scene",
        "learning_rate", "val/loss_total", "val/loss_ambient", "val/loss_diffuse",
        "val/loss_add_light", "val/loss_in_scene",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in train:
            step = int(record["global_step"])
            merged = {field: record.get(field) for field in fields}
            merged.update({key: value for key, value in validation_by_step.get(step, {}).items() if key in fields})
            merged["global_step"] = step
            writer.writerow(merged)


if __name__ == "__main__":
    main()
