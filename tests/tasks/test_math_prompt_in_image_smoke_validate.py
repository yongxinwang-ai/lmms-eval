"""Offline tests for the math prompt-in-image HF smoke validator."""

from pathlib import Path
import importlib.util
import io
import unittest
from unittest import mock
from contextlib import redirect_stdout

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "lmms_eval" / "tasks" / "math_prompt_in_image" / "smoke_validate_hf_dataset.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("math_prompt_in_image_smoke_validate", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeDataset:
    def __init__(self, rows):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, idx):
        return self._rows[idx]


class SmokeValidateTests(unittest.TestCase):
    def test_validate_config_checks_expected_fields_and_image(self):
        module = _load_module()
        fake_dataset = _FakeDataset(
            [
                {
                    "question_id": "1",
                    "question": "Q",
                    "answer": "A",
                    "decoded_image": Image.new("RGB", (12, 10), color="white"),
                }
            ]
        )

        with mock.patch.object(module, "load_dataset", return_value=fake_dataset):
            result = module.validate_config("YongxinWang/math-prompt-in-image", "mathvision_testmini_prompt_in_image", "testmini")

        self.assertEqual(result["config"], "mathvision_testmini_prompt_in_image")
        self.assertEqual(result["rows"], 1)
        self.assertEqual(result["image_size"], (12, 10))
        self.assertEqual(result["keys"], ["answer", "decoded_image", "question", "question_id"])

    def test_main_uses_both_default_configs(self):
        module = _load_module()
        seen = []

        def fake_validate(repo_id, config_name, split):
            seen.append((repo_id, config_name, split))
            return {"config": config_name, "rows": 1, "keys": ["decoded_image"], "image_size": (1, 1)}

        with mock.patch.object(module, "validate_config", side_effect=fake_validate):
            with redirect_stdout(io.StringIO()):
                rc = module.main([])

        self.assertEqual(rc, 0)
        self.assertEqual(
            seen,
            [
                ("YongxinWang/math-prompt-in-image", "mathvision_testmini_prompt_in_image", "testmini"),
                ("YongxinWang/math-prompt-in-image", "mathvista_testmini_prompt_in_image", "testmini"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
