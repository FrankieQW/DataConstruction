from __future__ import annotations

import importlib.util
from pathlib import Path
import random
import tempfile
import unittest

import bpy


ROOT = Path(__file__).resolve().parents[1]
RENDERER_PATH = ROOT / "scripts" / "blender_render_composition.py"


def load_renderer():
    spec = importlib.util.spec_from_file_location("focused_composition_renderer", RENDERER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load renderer: {RENDERER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BlenderCompositionFocusTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.renderer = load_renderer()
        cls.helpers = cls.renderer._load_helpers(ROOT)
        cls.recovery = cls.renderer._load_recovery(ROOT)

    def setUp(self) -> None:
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)

    def test_generated_camera_success_and_all_candidates_rejected(self) -> None:
        inserted = self._cube("Inserted", (0.0, 0.0, 0.5))
        job = self._camera_job()
        render_config = self.renderer._render_config({"minimum_visible_pixels": 64})
        original_render_mask = self.renderer._render_mask
        self.renderer._render_mask = lambda path, objects, config: 256
        try:
            with tempfile.TemporaryDirectory() as temporary:
                partial = Path(temporary)
                camera, _, result = self.renderer._select_camera(
                    job, [], [inserted], partial, render_config
                )
                bounds = result["inserted_ndc_bounds"]
                self.assertLessEqual(0.0, bounds["min_x"])
                self.assertLessEqual(bounds["max_x"], 1.0)
                self.assertLessEqual(0.0, bounds["min_y"])
                self.assertLessEqual(bounds["max_y"], 1.0)
                self.assertEqual(result["visible_pixels"], 256)
                self.assertEqual(result["target_visible_pixels"], 256)
                self.assertEqual(len(result["target_entity_ndc"]), 1)
                bpy.data.objects.remove(camera, do_unlink=True)

                rejected = self._camera_job()
                rejected["camera"]["ndc_x_range"] = [0.0, 0.1]
                with self.assertRaisesRegex(ValueError, "no generated camera"):
                    self.renderer._select_camera(rejected, [], [inserted], partial, render_config)
                diagnostics = partial / "diagnostics" / "camera_failures.json"
                self.assertTrue(diagnostics.is_file())
        finally:
            self.renderer._render_mask = original_render_mask

    def test_fixture_prefers_scene_native_then_uses_procedural_fallback(self) -> None:
        inserted = self._cube("Inserted", (0.0, 0.0, 0.5))
        lamp = self._cube("Lamp", (0.0, 0.0, 1.2), size=0.2)
        lamp["lc_scene_entity_id"] = "scene-1:entity:lamp"
        render_config = self.renderer._render_config({"minimum_visible_pixels": 64})
        original_render_mask = self.renderer._render_mask
        self.renderer._render_mask = lambda path, objects, config: 256
        try:
            with tempfile.TemporaryDirectory() as temporary:
                partial = Path(temporary)
                camera, target, camera_result = self.renderer._select_camera(
                    self._camera_job(), [], [inserted], partial, render_config
                )
                runtime = {
                    "render": {"minimum_visible_pixels": 64},
                    "fixture": {"minimum_visible_pixels": 64},
                }
                native = self.renderer._select_or_create_fixture(
                    {"fixture_candidate_entity_ids": ["scene-1:entity:lamp"]},
                    runtime,
                    camera,
                    target,
                    camera_result["subject_diameter"],
                    partial,
                    self.helpers,
                    random.Random(1),
                )
                self.assertEqual(native["source"], "scene_native")
                self.assertEqual(native["entity_id"], "scene-1:entity:lamp")

                runtime["fixture"].update(
                    {
                        "fallback_candidates": 1,
                        "fallback_position_ranges": {
                            "x": [0.0, 0.0],
                            "y": [-0.5, -0.5],
                            "z": [0.0, 0.0],
                        },
                    }
                )
                fallback = self.renderer._select_or_create_fixture(
                    {"fixture_candidate_entity_ids": []},
                    runtime,
                    camera,
                    target,
                    camera_result["subject_diameter"],
                    partial,
                    self.helpers,
                    random.Random(1),
                )
                self.assertEqual(fallback["source"], "procedural_fallback")
                self.assertIsNone(fallback["entity_id"])
        finally:
            self.renderer._render_mask = original_render_mask

    def test_preflight_failure_does_not_create_partial(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = root / "output"
            job = self._preflight_job()
            runtime = {
                "project_root": str(root),
                "object_root": str(root / "objects"),
                "output_root": str(output_root),
                "render": {"require_verified_license": True},
            }
            with self.assertRaises(FileNotFoundError):
                self.renderer.render_job(job, runtime, self.helpers, self.recovery)
            self.assertFalse(
                (output_root / "components" / f"{job['job_id']}.partial").exists()
            )

    def test_reuse_rejects_stale_render_job_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = root / "output"
            job = self._preflight_job()
            metadata_path = output_root / "components" / job["job_id"] / "metadata.json"
            metadata_path.parent.mkdir(parents=True)
            metadata_path.write_text(
                '{"id":"composition_focus","lineage":{"render_job_digest":"sha256:old"}}',
                encoding="utf-8",
            )
            runtime = {
                "project_root": str(root),
                "object_root": str(root / "objects"),
                "output_root": str(output_root),
                "overwrite": False,
            }
            with self.assertRaisesRegex(FileExistsError, "does not match"):
                self.renderer.render_job(job, runtime, self.helpers, self.recovery)

    @staticmethod
    def _cube(name: str, location, *, size: float = 1.0):
        bpy.ops.mesh.primitive_cube_add(size=size, location=location)
        cube = bpy.context.object
        cube.name = name
        return cube

    @staticmethod
    def _camera_job() -> dict:
        return {
            "seed": 7,
            "target": {
                "relation": "replace",
                "center_world": [0.0, 0.0, 0.5],
                "bottom_center_world": [0.0, 0.0, 0.0],
            },
            "camera": {
                "strategy": "generated_target_visible",
                "focal_length": 50.0,
                "azimuth_degrees": [0.0],
                "elevation_degrees": [30.0],
                "subject_fill_range": [0.2, 0.7],
                "shift_x_range": [0.0, 0.0],
                "shift_y_range": [0.0, 0.0],
                "ndc_x_range": [0.1, 0.9],
                "ndc_y_range": [0.1, 0.9],
                "edge_margin": 0.01,
                "candidate_count": 1,
                "target_minimum_visible_pixels": 64,
            },
        }

    @staticmethod
    def _preflight_job() -> dict:
        return {
            "job_id": "composition_focus",
            "base_scene_blend": "missing.blend",
            "base_scene_digest": "sha256:" + "1" * 64,
            "object_asset_path": "missing.glb",
            "prepared_geometry": {"asset_digest": "sha256:" + "2" * 64},
            "license": {"decision": "allowed"},
            "lineage": {"render_job_digest": "sha256:" + "3" * 64},
        }


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(BlenderCompositionFocusTest)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
