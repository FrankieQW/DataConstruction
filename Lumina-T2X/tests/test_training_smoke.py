from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "lumina_next_t2i"
sys.path.insert(0, str(PACKAGE_ROOT))

from tokenlight.tokens import TASK_NAMES  # noqa: E402
from tokenlight.training_smoke import (  # noqa: E402
    build_smoke_gate,
    finalize_smoke_gate,
    inspect_smoke_batch,
    optimizer_step_marker,
    record_smoke_checkpoint,
    record_smoke_step,
)


class _OneSampleSampler:
    def __init__(self, sample: tuple[int, int]):
        self.sample = sample

    def __iter__(self):
        yield self.sample


class TrainingSmokeTest(unittest.TestCase):
    def test_two_phase_smoke_only_marks_resumed_run_train_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "train.jsonl"
            manifest.write_text('{"id":"smoke"}\n', encoding="utf-8")
            config_path = root / "smoke.yaml"
            config_path.write_text("smoke: true\n", encoding="utf-8")
            run_directory = root / "run"
            run_directory.mkdir()
            config = self._config(manifest, config_path)

            first = build_smoke_gate(
                config,
                run_directory,
                None,
                None,
                _OneSampleSampler((0, 101)),
                global_step=0,
                samples_seen=0,
                resume_signature="signature",
            )
            assert first is not None
            inspect_smoke_batch(first, self._batch((0, 101)))
            optimizer = self._updated_optimizer()
            counts = {name: 1 for name in TASK_NAMES}
            losses = {name: 1.0 for name in TASK_NAMES}
            first_marker = optimizer_step_marker(optimizer)
            self._advance_optimizer(optimizer)
            record_smoke_step(first, 1.0, 0.5, losses, counts, optimizer, first_marker)
            first_checkpoint = run_directory / "checkpoints" / "step_000000001"
            first_checkpoint.mkdir(parents=True)
            record_smoke_checkpoint(first, first_checkpoint)
            finalize_smoke_gate(first, config, global_step=1, samples_seen=len(TASK_NAMES))

            summary_path = run_directory / "smoke_summary.json"
            initial_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(initial_summary["status"], "awaiting-resume")
            self.assertFalse(initial_summary["train_ready"])

            restored = {
                "global_step": 1,
                "samples_seen": len(TASK_NAMES),
                "dataloader_state": {},
            }
            resumed = build_smoke_gate(
                config,
                run_directory,
                str(first_checkpoint),
                restored,
                _OneSampleSampler((4, 202)),
                global_step=1,
                samples_seen=len(TASK_NAMES),
                resume_signature="signature",
            )
            assert resumed is not None
            inspect_smoke_batch(resumed, self._batch((4, 202)))
            resumed_marker = optimizer_step_marker(optimizer)
            self._advance_optimizer(optimizer)
            record_smoke_step(resumed, 0.8, 0.4, losses, counts, optimizer, resumed_marker)
            second_checkpoint = run_directory / "checkpoints" / "step_000000002"
            second_checkpoint.mkdir(parents=True)
            record_smoke_checkpoint(resumed, second_checkpoint)
            finalize_smoke_gate(
                resumed,
                config,
                global_step=2,
                samples_seen=2 * len(TASK_NAMES),
            )

            final_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(final_summary["status"], "pass")
            self.assertTrue(final_summary["train_ready"])
            self.assertTrue(final_summary["resume_verified"])
            self.assertEqual(final_summary["expected_next_sample"], [4, 202])
            self.assertTrue(all(final_summary["task_counts"][name] == 2 for name in TASK_NAMES))

    def test_failed_fixture_assertion_never_marks_train_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "train.jsonl"
            manifest.write_text('{"id":"smoke"}\n', encoding="utf-8")
            config_path = root / "smoke.yaml"
            config_path.write_text("smoke: true\n", encoding="utf-8")
            run_directory = root / "run"
            run_directory.mkdir()
            config = self._config(manifest, config_path)
            gate = build_smoke_gate(
                config,
                run_directory,
                None,
                None,
                _OneSampleSampler((0, 101)),
                global_step=0,
                samples_seen=0,
                resume_signature="signature",
            )
            assert gate is not None
            batch = self._batch((0, 101))
            batch["fixture_mask"].zero_()
            with self.assertRaisesRegex(RuntimeError, "empty fixture mask"):
                inspect_smoke_batch(gate, batch)
            summary = json.loads((run_directory / "smoke_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "failed")
            self.assertFalse(summary["train_ready"])

    @staticmethod
    def _config(manifest: Path, config_path: Path) -> dict:
        return {
            "_config_path": str(config_path),
            "paths": {"train_manifest": str(manifest)},
            "data": {"tasks": list(TASK_NAMES)},
            "train": {
                "smoke": {
                    "enabled": True,
                    "required_tasks": list(TASK_NAMES),
                    "summary_json": "smoke_summary.json",
                }
            },
            "logging": {"run_id": "unit-smoke"},
        }

    @staticmethod
    def _batch(first_sample: tuple[int, int]) -> dict:
        batch_size = len(TASK_NAMES)
        values = torch.ones(batch_size, 2, dtype=torch.float32)
        known = torch.ones(batch_size, 2, dtype=torch.bool)
        valid = torch.ones(batch_size, 2, dtype=torch.bool)
        fixture_mask = torch.zeros(batch_size, 1, 2, 2, dtype=torch.float32)
        in_scene_index = TASK_NAMES.index("in_scene_light")
        fixture_mask[in_scene_index] = 1.0
        fixture_present = torch.zeros(batch_size, dtype=torch.bool)
        fixture_present[in_scene_index] = True
        return {
            "lighting_values": values,
            "lighting_known": known,
            "lighting_valid": valid,
            "fixture_mask": fixture_mask,
            "fixture_present": fixture_present,
            "task": torch.arange(batch_size, dtype=torch.long),
            "sample_index": [first_sample[0], *range(1, batch_size)],
            "sample_seed": [first_sample[1], *range(1, batch_size)],
        }

    @staticmethod
    def _updated_optimizer() -> torch.optim.Optimizer:
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([parameter], lr=0.1)
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        return optimizer

    @staticmethod
    def _advance_optimizer(optimizer: torch.optim.Optimizer) -> None:
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                parameter.grad = torch.ones_like(parameter)
        optimizer.step()


if __name__ == "__main__":
    unittest.main()
