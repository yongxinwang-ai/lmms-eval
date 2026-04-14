"""Schema and compatibility tests for math prompt-in-image tasks."""

import contextlib
import importlib.util
import pathlib
import sys
import types
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "lmms_eval" / "tasks" / "math_prompt_in_image" / "SCHEMA.md"
PROMPT_UTILS_PATH = ROOT / "lmms_eval" / "tasks" / "_task_utils" / "prompt_in_image.py"
MATHVISION_UTILS_PATH = ROOT / "lmms_eval" / "tasks" / "mathvision" / "utils.py"
MATHVISTA_UTILS_PATH = ROOT / "lmms_eval" / "tasks" / "mathvista" / "utils.py"


class _SentinelImage:
    def __init__(self, name):
        self.name = name
        self.convert_calls = []

    def convert(self, mode):
        self.convert_calls.append(mode)
        return self


class _FakeFont:
    def __init__(self, size=12):
        self.size = size


class _FakeDraw:
    def __init__(self, image):
        self.image = image

    def textbbox(self, xy, text, font=None):
        font = font or _FakeFont()
        width = max(1, int(len(str(text)) * max(1, font.size) * 0.6))
        height = max(1, int(max(1, font.size)))
        return (0, 0, width, height)

    def text(self, xy, text, font=None, fill=None):
        return None


class _FakeImage:
    def __init__(self, mode, size, color="white"):
        self.mode = mode
        self._size = tuple(size)
        self.color = color

    @property
    def size(self):
        return self._size

    @property
    def width(self):
        return self._size[0]

    @property
    def height(self):
        return self._size[1]

    def convert(self, mode):
        self.mode = mode
        return self

    def copy(self):
        return _FakeImage(self.mode, self._size, self.color)

    def paste(self, image, box):
        return None


def _fake_image_new(mode, size, color="white"):
    return _FakeImage(mode, size, color)


def _fake_draw_factory(image):
    return _FakeDraw(image)


@contextlib.contextmanager
def _temporary_sys_modules(overrides):
    original = {}
    missing = object()
    try:
        for name, module in overrides.items():
            original[name] = sys.modules.get(name, missing)
            sys.modules[name] = module
        yield
    finally:
        for name, module in original.items():
            if module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _load_module(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_task_modules():
    pil_module = types.ModuleType("PIL")
    pil_image = types.ModuleType("PIL.Image")
    pil_image.Image = _FakeImage
    pil_image.new = _fake_image_new
    pil_draw = types.ModuleType("PIL.ImageDraw")
    pil_draw.ImageDraw = _FakeDraw
    pil_draw.Draw = _fake_draw_factory
    pil_font = types.ModuleType("PIL.ImageFont")
    pil_font.ImageFont = _FakeFont
    pil_font.truetype = lambda *args, **kwargs: (_ for _ in ()).throw(OSError("font unavailable"))
    pil_font.load_default = lambda: _FakeFont()
    pil_module.Image = pil_image
    pil_module.ImageDraw = pil_draw
    pil_module.ImageFont = pil_font

    loguru_module = types.ModuleType("loguru")

    class _Logger:
        def warning(self, *args, **kwargs):
            return None

        def error(self, *args, **kwargs):
            return None

        def info(self, *args, **kwargs):
            return None

        def debug(self, *args, **kwargs):
            return None

    loguru_module.logger = _Logger()

    yaml_module = types.ModuleType("yaml")
    yaml_module.safe_load = lambda _text: {"metadata": {"quick_extract": False}}

    lmms_eval_module = types.ModuleType("lmms_eval")
    lmms_eval_module.__path__ = [str(ROOT / "lmms_eval")]
    tasks_module = types.ModuleType("lmms_eval.tasks")
    tasks_module.__path__ = [str(ROOT / "lmms_eval" / "tasks")]
    task_utils_module = types.ModuleType("lmms_eval.tasks._task_utils")
    task_utils_module.__path__ = [str(ROOT / "lmms_eval" / "tasks" / "_task_utils")]
    mathvision_pkg = types.ModuleType("lmms_eval.tasks.mathvision")
    mathvision_pkg.__path__ = [str(ROOT / "lmms_eval" / "tasks" / "mathvision")]
    mathvista_pkg = types.ModuleType("lmms_eval.tasks.mathvista")
    mathvista_pkg.__path__ = [str(ROOT / "lmms_eval" / "tasks" / "mathvista")]

    llm_judge_module = types.ModuleType("lmms_eval.llm_judge")

    class _ServerConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    def _get_server(*args, **kwargs):
        return types.SimpleNamespace(evaluate_binary=lambda **_kwargs: {"success": False, "result": False})

    llm_judge_module.ServerConfig = _ServerConfig
    llm_judge_module.get_server = _get_server

    mathvision_eval_utils = types.ModuleType("lmms_eval.tasks.mathvision.eval_utils")

    def _is_number(s):
        try:
            float(str(s).replace(",", ""))
            return True
        except (TypeError, ValueError):
            return False

    def _find_math_answer(text):
        return str(text).strip()

    def _is_equal(a, b):
        a = str(a).strip().lower()
        b = str(b).strip().lower()
        if a == b:
            return True
        if _is_number(a) and _is_number(b):
            return abs(float(a.replace(",", "")) - float(b.replace(",", ""))) < 1e-6
        return False

    mathvision_eval_utils.is_number = _is_number
    mathvision_eval_utils.find_math_answer = _find_math_answer
    mathvision_eval_utils.is_equal = _is_equal

    mathvista_evals = types.ModuleType("lmms_eval.tasks.mathvista.mathvista_evals")

    class _MathVistaEvaluator:
        def create_one_query(self, *args, **kwargs):
            return "stub-query"

        def extract_answer(self, *args, **kwargs):
            return "stub-extraction"

        def normalize_extracted_answer(self, *args, **kwargs):
            return "stub-answer"

        def safe_equal(self, predicted, answer):
            return predicted == answer

        def get_acc_with_contion(self, *args, **kwargs):
            return 0, 0, 0.0

    mathvista_evals.MathVistaEvaluator = _MathVistaEvaluator

    overrides = {
        "PIL": pil_module,
        "PIL.Image": pil_image,
        "PIL.ImageDraw": pil_draw,
        "PIL.ImageFont": pil_font,
        "loguru": loguru_module,
        "yaml": yaml_module,
        "lmms_eval": lmms_eval_module,
        "lmms_eval.tasks": tasks_module,
        "lmms_eval.tasks._task_utils": task_utils_module,
        "lmms_eval.tasks.mathvision": mathvision_pkg,
        "lmms_eval.tasks.mathvision.eval_utils": mathvision_eval_utils,
        "lmms_eval.tasks.mathvista": mathvista_pkg,
        "lmms_eval.tasks.mathvista.mathvista_evals": mathvista_evals,
        "lmms_eval.llm_judge": llm_judge_module,
    }

    loaded_names = [
        "lmms_eval.tasks._task_utils.prompt_in_image",
        "lmms_eval.tasks.mathvision.utils",
        "lmms_eval.tasks.mathvista.utils",
    ]
    missing = object()
    loaded_originals = {name: sys.modules.get(name, missing) for name in loaded_names}

    with _temporary_sys_modules(overrides):
        _load_module("lmms_eval.tasks._task_utils.prompt_in_image", PROMPT_UTILS_PATH)
        mathvision_utils = _load_module("lmms_eval.tasks.mathvision.utils", MATHVISION_UTILS_PATH)
        mathvista_utils = _load_module("lmms_eval.tasks.mathvista.utils", MATHVISTA_UTILS_PATH)

    for name, module in loaded_originals.items():
        if module is missing:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module

    return types.SimpleNamespace(mathvision_utils=mathvision_utils, mathvista_utils=mathvista_utils)


MODULES = _load_task_modules()
MATHVISION_UTILS = MODULES.mathvision_utils
MATHVISTA_UTILS = MODULES.mathvista_utils


class _MathVistaEvaluatorStub:
    def extract_answer(self, prediction, problem, quick_extract):
        self.last_extract = {
            "prediction": prediction,
            "problem": problem,
            "quick_extract": quick_extract,
        }
        return "extracted"

    def normalize_extracted_answer(self, extraction, choices, question_type, answer_type, precision):
        self.last_normalize = {
            "extraction": extraction,
            "choices": choices,
            "question_type": question_type,
            "answer_type": answer_type,
            "precision": precision,
        }
        return "normalized-answer"

    def safe_equal(self, predicted, answer):
        self.last_safe_equal = {"predicted": predicted, "answer": answer}
        return predicted == answer


class TestMathPromptInImageSchema(unittest.TestCase):
    def test_schema_doc_matches_plan_contract(self):
        content = SCHEMA_PATH.read_text(encoding="utf-8")
        self.assertIn("dataset_path: `YongxinWang/math-prompt-in-image`", content)
        self.assertIn("`mathvision_testmini_prompt_in_image`", content)
        self.assertIn("`mathvista_testmini_prompt_in_image`", content)
        self.assertIn("- `testmini`", content)
        self.assertIn("- `question_id`: string", content)
        self.assertIn("- `decoded_image`: HF `Image` feature", content)
        self.assertIn("- `question`: string", content)
        self.assertIn("- `answer`: string", content)
        self.assertIn("- `source_split`: string", content)
        self.assertIn("- `options`: list[string]", content)
        self.assertIn("- `query`: string", content)
        self.assertIn("- `choices`: list[string]", content)
        self.assertIn("- `question_type`: string", content)
        self.assertIn("- `answer_type`: string", content)
        self.assertIn("- `precision`: int", content)
        self.assertIn("- `metadata`: dict", content)

    def test_prompt_in_image_uses_decoded_image_when_present(self):
        decoded = _SentinelImage("decoded")
        image = _SentinelImage("legacy")
        doc = {"decoded_image": decoded, "image": image, "question": "Q"}

        with mock.patch.object(MATHVISION_UTILS, "render_question_on_image", return_value="rendered") as render_mock:
            out = MATHVISION_UTILS.mathvision_doc_to_visual_prompt_in_image(doc)

        render_mock.assert_called_once_with(decoded, "Q")
        self.assertEqual(out, ["rendered"])

    def test_prompt_in_image_falls_back_to_image_when_decoded_image_missing(self):
        image = _SentinelImage("legacy")
        doc = {"image": image, "question": "Q"}

        with mock.patch.object(MATHVISTA_UTILS, "render_question_on_image", return_value="rendered") as render_mock:
            out = MATHVISTA_UTILS.mathvista_doc_to_visual_prompt_in_image(doc)

        render_mock.assert_called_once_with(image, "Q")
        self.assertEqual(out, ["rendered"])

    def test_prompt_in_image_raises_when_no_image_field_exists(self):
        for utils_mod, func_name in [
            (MATHVISION_UTILS, "mathvision_doc_to_visual_prompt_in_image"),
            (MATHVISTA_UTILS, "mathvista_doc_to_visual_prompt_in_image"),
        ]:
            with self.subTest(func=func_name):
                with self.assertRaises(KeyError):
                    getattr(utils_mod, func_name)({"question": "Q"})

    def test_mathvista_process_results_uses_question_id(self):
        stub = _MathVistaEvaluatorStub()
        original = MATHVISTA_UTILS.mathvista_evaluator
        MATHVISTA_UTILS.mathvista_evaluator = stub
        try:
            result = MATHVISTA_UTILS.mathvista_process_results(
                {
                    "question_id": "question-123",
                    "query": "Query",
                    "choices": ["A", "B"],
                    "question_type": "multi-choice",
                    "answer_type": "string",
                    "precision": 0,
                    "answer": "normalized-answer",
                    "metadata": {"split": "testmini"},
                },
                [" prediction "],
            )
        finally:
            MATHVISTA_UTILS.mathvista_evaluator = original

        self.assertEqual(result["submission"]["question_id"], "question-123")
        self.assertEqual(result["llm_as_judge_eval"]["question_id"], "question-123")
        self.assertTrue(result["submission"]["true_false"])

    def test_mathvista_process_results_falls_back_to_pid(self):
        stub = _MathVistaEvaluatorStub()
        original = MATHVISTA_UTILS.mathvista_evaluator
        MATHVISTA_UTILS.mathvista_evaluator = stub
        try:
            result = MATHVISTA_UTILS.mathvista_process_results(
                {
                    "pid": "legacy-456",
                    "query": "Query",
                    "choices": ["A", "B"],
                    "question_type": "multi-choice",
                    "answer_type": "string",
                    "precision": 0,
                    "answer": "normalized-answer",
                    "metadata": {"split": "testmini"},
                },
                [" prediction "],
            )
        finally:
            MATHVISTA_UTILS.mathvista_evaluator = original

        self.assertEqual(result["submission"]["question_id"], "legacy-456")
        self.assertEqual(result["llm_as_judge_eval"]["question_id"], "legacy-456")
        self.assertTrue(result["submission"]["true_false"])


if __name__ == "__main__":
    unittest.main()
