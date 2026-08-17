from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import yaml

from scripts.stage2_config import sha256_file, verify_reused_annotation


class ReusedAnnotationPreflightTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "data").mkdir()
        (self.root / "cache").mkdir()
        self.blend = self.root / "cache" / "scene.blend"
        self.blend.write_bytes(b"blend")

        self.object_path = self.root / "data" / "object.json"
        self.scene_path = self.root / "data" / "scene.json"
        self.annotation_path = self.root / "data" / "annotation_construction.json"
        self.object_path.write_text(json.dumps({"objects": []}), encoding="utf-8")
        self.scene_path.write_text(
            json.dumps({"scenes": [{"normalized_blend": "cache/scene.blend"}]}),
            encoding="utf-8",
        )
        self.annotation_path.write_text(
            json.dumps(
                {
                    "object_digest": f"sha256:{sha256_file(self.object_path)}",
                    "scene_digest": f"sha256:{sha256_file(self.scene_path)}",
                }
            ),
            encoding="utf-8",
        )
        self.config_path = self.root / "project.yaml"
        self.config_path.write_text(
            yaml.safe_dump(
                {
                    "project": {"root": str(self.root)},
                    "paths": {
                        "object_output": "data/object.json",
                        "scene_output": "data/scene.json",
                        "annotation_output": "data/annotation_construction.json",
                    },
                }
            ),
            encoding="utf-8",
        )

    def test_digest_current_artifacts_and_blend_pass(self) -> None:
        summary = verify_reused_annotation(self.config_path)
        self.assertEqual(summary["status"], "reusable")
        self.assertEqual(summary["normalized_blends"], 1)

    def test_stale_object_digest_fails_with_rebuild_instruction(self) -> None:
        self.object_path.write_text(json.dumps({"objects": [{"uid": "changed"}]}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "stale object.json.*REUSE_ANNOTATION=false"):
            verify_reused_annotation(self.config_path)

    def test_missing_normalized_blend_fails_with_rebuild_instruction(self) -> None:
        self.blend.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "normalized blend.*REUSE_ANNOTATION=false"):
            verify_reused_annotation(self.config_path)


if __name__ == "__main__":
    unittest.main()
