"""Tests for the public math prompt-in-image dataset builder."""

import importlib.util
import pathlib
import unittest
from unittest import mock

import datasets
from PIL import Image


ROOT = pathlib.Path(__file__).resolve().parents[2]
BUILDER_PATH = ROOT / "lmms_eval" / "tasks" / "math_prompt_in_image" / "build_hf_dataset.py"


def _load_builder_module():
    module_name = "lmms_eval.tasks.math_prompt_in_image.build_hf_dataset"
    spec = importlib.util.spec_from_file_location(module_name, BUILDER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_render_image(color="white"):
    return Image.new("RGB", (8, 6), color=color)


class BuilderDatasetConstructionTests(unittest.TestCase):
    def test_mathvision_dataset_uses_hf_image_feature(self):
        builder = _load_builder_module()
        source_image = Image.new("RGB", (4, 4), color="blue")
        source_docs = [
            {
                "id": 17,
                "question": "What is 2 + 2?",
                "options": ["1", "2", "4"],
                "decoded_image": source_image,
                "answer": "C",
                "solution": "4",
                "level": "easy",
                "subject": "arithmetic",
            }
        ]

        with mock.patch.object(builder, "_load_source_dataset", return_value=source_docs), mock.patch.object(
            builder, "_default_render_fn", return_value=lambda image, question: _fake_render_image("green")
        ):
            dataset = builder.build_mathvision_dataset()

        self.assertEqual(len(dataset), 1)
        self.assertIsInstance(dataset.features["decoded_image"], datasets.Image)
        self.assertEqual(dataset[0]["question_id"], "17")
        self.assertEqual(dataset[0]["question"], "What is 2 + 2?")
        self.assertEqual(dataset[0]["options"], ["1", "2", "4"])
        self.assertEqual(dataset[0]["answer"], "C")
        self.assertIsInstance(dataset[0]["decoded_image"], Image.Image)
        self.assertEqual(dataset[0]["decoded_image"].size, (8, 6))

    def test_mathvista_dataset_uses_explicit_nested_metadata_feature(self):
        builder = _load_builder_module()
        source_image = Image.new("RGB", (5, 5), color="orange")
        source_docs = [
            {
                "pid": 81,
                "question": "How many triangles are shown?",
                "decoded_image": source_image,
                "choices": ["1", "2", "3"],
                "unit": "",
                "precision": 0,
                "answer": "2",
                "question_type": "counting",
                "answer_type": "integer",
                "metadata": {
                    "split": "testmini",
                    "language": "en",
                    "source": "synthetic",
                    "category": "geometry",
                    "task": "counting",
                    "context": "diagram",
                    "grade": "grade-school",
                    "img_height": 480,
                    "img_width": 640,
                    "skills": ["counting", "geometry"],
                },
                "query": "Count the triangles.",
            }
        ]

        with mock.patch.object(builder, "_load_source_dataset", return_value=source_docs), mock.patch.object(
            builder, "_default_render_fn", return_value=lambda image, question: _fake_render_image("red")
        ):
            dataset = builder.build_mathvista_dataset()

        expected_metadata_features = datasets.Features(
            {
                "split": datasets.Value("string"),
                "language": datasets.Value("string"),
                "source": datasets.Value("string"),
                "category": datasets.Value("string"),
                "task": datasets.Value("string"),
                "context": datasets.Value("string"),
                "grade": datasets.Value("string"),
                "img_height": datasets.Value("int64"),
                "img_width": datasets.Value("int64"),
                "skills": datasets.Sequence(datasets.Value("string")),
            }
        )
        self.assertEqual(dataset.features["metadata"], expected_metadata_features)
        self.assertEqual(dataset[0]["question_id"], "81")
        self.assertEqual(dataset[0]["query"], "Count the triangles.")
        self.assertEqual(dataset[0]["metadata"]["split"], "testmini")
        self.assertEqual(dataset[0]["metadata"]["img_height"], 480)
        self.assertEqual(dataset[0]["metadata"]["img_width"], 640)
        self.assertEqual(dataset[0]["metadata"]["skills"], ["counting", "geometry"])
        self.assertIsInstance(dataset[0]["decoded_image"], Image.Image)
        self.assertEqual(dataset[0]["decoded_image"].size, (8, 6))

    def test_transform_helpers_call_renderer_with_source_image_and_question(self):
        builder = _load_builder_module()
        source_image = Image.new("RGB", (3, 3), color="purple")

        mathvision_calls = []

        def mathvision_render(image, question, **kwargs):
            mathvision_calls.append((image, question, kwargs))
            return _fake_render_image("cyan")

        builder.build_mathvision_record(
            {
                "id": 5,
                "question": "What is shown?",
                "decoded_image": source_image,
                "options": [],
                "answer": "A",
                "solution": "",
                "level": "",
                "subject": "",
            },
            render_fn=mathvision_render,
        )

        mathvista_calls = []

        def mathvista_render(image, question, **kwargs):
            mathvista_calls.append((image, question, kwargs))
            return _fake_render_image("yellow")

        builder.build_mathvista_record(
            {
                "pid": 9,
                "question": "How many shapes?",
                "decoded_image": source_image,
                "choices": [],
                "unit": "",
                "precision": 0,
                "answer": "1",
                "question_type": "",
                "answer_type": "",
                "metadata": {},
                "query": "",
            },
            render_fn=mathvista_render,
        )

        self.assertEqual(mathvision_calls, [(source_image, "What is shown?", {})])
        self.assertEqual(mathvista_calls, [(source_image, "How many shapes?", {})])

    def test_record_builders_treat_none_options_and_choices_as_empty_lists(self):
        builder = _load_builder_module()
        source_image = Image.new("RGB", (3, 3), color="white")

        mathvision_record = builder.build_mathvision_record(
            {
                "id": 1,
                "question": "Q",
                "decoded_image": source_image,
                "options": None,
                "answer": "A",
                "solution": "",
                "level": "",
                "subject": "",
            },
            render_fn=lambda image, question: _fake_render_image("black"),
        )
        mathvista_record = builder.build_mathvista_record(
            {
                "pid": 2,
                "question": "Q",
                "decoded_image": source_image,
                "choices": None,
                "unit": "",
                "precision": 0,
                "answer": "A",
                "question_type": "",
                "answer_type": "",
                "metadata": {},
                "query": "",
            },
            render_fn=lambda image, question: _fake_render_image("black"),
        )

        self.assertEqual(mathvision_record["options"], [])
        self.assertEqual(mathvista_record["choices"], [])


if __name__ == "__main__":
    unittest.main()
