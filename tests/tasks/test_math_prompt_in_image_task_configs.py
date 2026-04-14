"""Resolve prompt-in-image task YAMLs and validate the effective public config."""

from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]
TASKS_ROOT = ROOT / "lmms_eval" / "tasks"


class _FunctionLoader(yaml.SafeLoader):
    pass


def _construct_function(loader, node):
    return loader.construct_scalar(node)


_FunctionLoader.add_constructor("!function", _construct_function)


def _load_yaml(path: Path) -> dict:
    data = yaml.load(path.read_text(), Loader=_FunctionLoader)
    return data or {}


def _resolve_include(path: Path) -> dict:
    data = _load_yaml(path)
    include_name = data.pop("include", None)
    if include_name is None:
        return data

    include_path = path.parent / include_name
    base = _load_yaml(include_path)
    base.update(data)
    return base


class PromptInImageTaskConfigTests(unittest.TestCase):
    def test_mathvision_prompt_in_image_resolves_to_public_hf_dataset(self):
        config = _resolve_include(TASKS_ROOT / "mathvision" / "mathvision_testmini_prompt_in_image.yaml")

        self.assertEqual(config["dataset_path"], "YongxinWang/math-prompt-in-image")
        self.assertEqual(config["dataset_name"], "mathvision_testmini_prompt_in_image")
        self.assertEqual(config["dataset_kwargs"], {})
        self.assertEqual(config["test_split"], "testmini")
        self.assertEqual(config["doc_to_visual"], "utils.mathvision_doc_to_visual")
        self.assertEqual(config["doc_to_text"], "utils.mathvision_doc_to_text_minimal")
        self.assertEqual(config["doc_to_target"], "answer")

    def test_mathvista_prompt_in_image_resolves_to_public_hf_dataset(self):
        config = _resolve_include(TASKS_ROOT / "mathvista" / "mathvista_testmini_prompt_in_image.yaml")

        self.assertEqual(config["dataset_path"], "YongxinWang/math-prompt-in-image")
        self.assertEqual(config["dataset_name"], "mathvista_testmini_prompt_in_image")
        self.assertEqual(config["dataset_kwargs"], {})
        self.assertEqual(config["test_split"], "testmini")
        self.assertEqual(config["doc_to_visual"], "utils.mathvista_doc_to_visual")
        self.assertEqual(config["doc_to_text"], "utils.mathvista_doc_to_text_minimal")
        self.assertEqual(config["doc_to_target"], "answer")

    def test_group_yaml_lists_exact_public_tasks(self):
        group_config = _load_yaml(TASKS_ROOT / "math_prompt_in_image" / "math_prompt_in_image_testmini.yaml")

        self.assertEqual(group_config["group"], "math_prompt_in_image_testmini")
        self.assertEqual(
            group_config["task"],
            [
                "mathvision_testmini_prompt_in_image",
                "mathvista_testmini_prompt_in_image",
            ],
        )
        self.assertEqual(group_config["metadata"]["version"], 0.1)


if __name__ == "__main__":
    unittest.main()
