from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "lumina_next_t2i"))

from tokenlight.config import load_config  # noqa: E402
from tokenlight.dataset import TokenLightDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export deterministic source/target/delta inspection grids.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    config = load_config(parse_args().config)
    dataset = TokenLightDataset(config, "validation")
    samples = select_samples(dataset, config)
    count = len(samples)
    resolution = int(config["data"]["resolution"])
    label_height = 24
    canvas = Image.new("RGB", (resolution * 3, (resolution + label_height) * count), "white")
    metadata = []
    progress = tqdm(range(count), desc="inspect dataset", unit="sample", dynamic_ncols=True)
    for index in progress:
        sample_index, sample_seed, sample = samples[index]
        source = to_uint8(sample["source_image"])
        target = to_uint8(sample["target_image"])
        delta = np.abs(target.astype(np.int16) - source.astype(np.int16)).astype(np.uint8)
        top = index * (resolution + label_height)
        canvas.paste(Image.fromarray(source), (0, top))
        canvas.paste(Image.fromarray(target), (resolution, top))
        canvas.paste(Image.fromarray(delta), (resolution * 2, top))
        ImageDraw.Draw(canvas).text((4, top + resolution + 4), f"{sample['scene_id']} | {sample['task_name']}", fill="black")
        metadata.append(
            {
                "index": sample_index,
                "sample_seed": sample_seed,
                "scene_id": sample["scene_id"],
                "task": sample["task_name"],
            }
        )
        progress.set_postfix(task=sample["task_name"])
    output = Path(config["paths"]["output_root"]) / config["data"]["inspect_output"]
    output.mkdir(parents=True, exist_ok=True)
    tqdm.write("样本处理完成，正在压缩并保存 PNG...")
    canvas.save(output / "source_target_delta.png")
    (output / "samples.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已写入 {output}", flush=True)


def select_samples(dataset: TokenLightDataset, config: dict) -> list[tuple[int, int | None, dict]]:
    data = config["data"]
    if not bool(data.get("inspect_all_tasks", True)):
        count = min(int(data["inspect_samples"]), len(dataset))
        return [(index, None, dataset[index]) for index in range(count)]
    required = list(data["tasks"])
    selected: dict[str, tuple[int, int, dict]] = {}
    limit = int(data.get("inspect_seed_search_limit", 4096))
    base_seed = int(data.get("validation_seed", 0))
    for offset in range(limit):
        index = offset % len(dataset)
        seed = base_seed + offset
        sample = dataset[(index, seed)]
        selected.setdefault(sample["task_name"], (index, seed, sample))
        if all(task in selected for task in required):
            return [selected[task] for task in required]
    missing = [task for task in required if task not in selected]
    raise RuntimeError(
        f"TokenLightDataset smoke 未能覆盖配置任务 {missing}; "
        f"search_limit={limit}，请检查 manifest 组件和任务概率"
    )


def to_uint8(tensor) -> np.ndarray:
    array = tensor.permute(1, 2, 0).numpy()
    return np.uint8(np.round(np.clip((array + 1) * 0.5, 0, 1) * 255))


if __name__ == "__main__":
    main()
