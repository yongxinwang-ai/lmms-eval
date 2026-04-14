"""Build the public math-prompt-in-image dataset from source HF datasets."""

from __future__ import annotations

import argparse
import importlib.util
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable

SOURCE_SPLIT = "testmini"
DEFAULT_REPO_ID = "YongxinWang/math-prompt-in-image"
MATHVISION_SOURCE = "MathLLMs/MathVision"
MATHVISTA_SOURCE = "AI4Math/MathVista"
MATHVISION_CONFIG = "mathvision_testmini_prompt_in_image"
MATHVISTA_CONFIG = "mathvista_testmini_prompt_in_image"
PROMPT_IN_IMAGE_PATH = Path(__file__).resolve().parents[1] / "_task_utils" / "prompt_in_image.py"


def _datasets():
    import datasets as hf_datasets

    return hf_datasets


@lru_cache(maxsize=1)
def _prompt_in_image_module():
    module_name = "lmms_eval.tasks._task_utils.prompt_in_image"
    spec = importlib.util.spec_from_file_location(module_name, PROMPT_IN_IMAGE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load prompt-in-image helper from {PROMPT_IN_IMAGE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _default_render_fn():
    return _prompt_in_image_module().render_question_on_image


def _get_source_image(doc: dict[str, Any]):
    image = doc.get("decoded_image")
    if image is None:
        image = doc.get("image")
    if image is None:
        raise KeyError("Expected `decoded_image` or `image` in source document")
    return image


def _get_question_id(doc: dict[str, Any]) -> str:
    for key in ("question_id", "id", "pid"):
        if key in doc and doc[key] is not None:
            return str(doc[key])
    raise KeyError("Expected `question_id`, `id`, or `pid` in source document")


def _normalize_string(value: Any) -> str:
    return "" if value is None else str(value)


def _normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [_normalize_string(item) for item in value]
    return [_normalize_string(value)]


def _normalize_int(value: Any, default: int = 0) -> int:
    try:
        return default if value is None else int(value)
    except (TypeError, ValueError):
        return default


def _normalize_mathvista_metadata(metadata: Any) -> dict[str, Any]:
    metadata_dict = metadata or {}
    if not isinstance(metadata_dict, dict):
        metadata_dict = {"source": metadata_dict}
    return {
        "split": _normalize_string(metadata_dict.get("split")),
        "language": _normalize_string(metadata_dict.get("language")),
        "source": _normalize_string(metadata_dict.get("source")),
        "category": _normalize_string(metadata_dict.get("category")),
        "task": _normalize_string(metadata_dict.get("task")),
        "context": _normalize_string(metadata_dict.get("context")),
        "grade": _normalize_string(metadata_dict.get("grade")),
        "img_height": _normalize_int(metadata_dict.get("img_height")),
        "img_width": _normalize_int(metadata_dict.get("img_width")),
        "skills": _normalize_string_list(metadata_dict.get("skills")),
    }


def _build_mathvision_features():
    datasets = _datasets()
    return datasets.Features(
        {
            "question_id": datasets.Value("string"),
            "question": datasets.Value("string"),
            "decoded_image": datasets.Image(),
            "answer": datasets.Value("string"),
            "options": datasets.Sequence(datasets.Value("string")),
            "solution": datasets.Value("string"),
            "level": datasets.Value("string"),
            "subject": datasets.Value("string"),
            "source_dataset": datasets.Value("string"),
            "source_split": datasets.Value("string"),
        }
    )


def _build_mathvista_features():
    datasets = _datasets()
    metadata_features = datasets.Features(
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
    return datasets.Features(
        {
            "question_id": datasets.Value("string"),
            "question": datasets.Value("string"),
            "query": datasets.Value("string"),
            "decoded_image": datasets.Image(),
            "answer": datasets.Value("string"),
            "choices": datasets.Sequence(datasets.Value("string")),
            "unit": datasets.Value("string"),
            "precision": datasets.Value("int64"),
            "question_type": datasets.Value("string"),
            "answer_type": datasets.Value("string"),
            "metadata": metadata_features,
            "source_dataset": datasets.Value("string"),
            "source_split": datasets.Value("string"),
        }
    )


def build_mathvision_record(doc: dict[str, Any], render_fn: Callable[..., Any] | None = None) -> dict[str, Any]:
    render_fn = render_fn or _default_render_fn()
    question = str(doc.get("question", ""))
    rendered_image = render_fn(_get_source_image(doc), question)
    return {
        "question_id": _get_question_id(doc),
        "question": question,
        "decoded_image": rendered_image,
        "answer": str(doc.get("answer", "")),
        "options": _normalize_string_list(doc.get("options")),
        "solution": str(doc.get("solution", "")),
        "level": str(doc.get("level", "")),
        "subject": str(doc.get("subject", "")),
        "source_dataset": MATHVISION_SOURCE,
        "source_split": SOURCE_SPLIT,
    }


def build_mathvista_record(doc: dict[str, Any], render_fn: Callable[..., Any] | None = None) -> dict[str, Any]:
    render_fn = render_fn or _default_render_fn()
    question = str(doc.get("question", ""))
    rendered_image = render_fn(_get_source_image(doc), question)
    return {
        "question_id": _get_question_id(doc),
        "question": question,
        "query": str(doc.get("query", "")),
        "decoded_image": rendered_image,
        "answer": str(doc.get("answer", "")),
        "choices": _normalize_string_list(doc.get("choices")),
        "unit": str(doc.get("unit", "")),
        "precision": int(doc.get("precision", 0) or 0),
        "question_type": str(doc.get("question_type", "")),
        "answer_type": str(doc.get("answer_type", "")),
        "metadata": _normalize_mathvista_metadata(doc.get("metadata", {})),
        "source_dataset": MATHVISTA_SOURCE,
        "source_split": SOURCE_SPLIT,
    }


def _load_source_dataset(dataset_name: str, *, cache_dir: str | None = None):
    datasets = _datasets()
    return datasets.load_dataset(dataset_name, split=SOURCE_SPLIT, cache_dir=cache_dir)


def _build_records(
    source_docs: Iterable[dict[str, Any]],
    record_builder: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for idx, doc in enumerate(source_docs):
        if limit is not None and idx >= limit:
            break
        records.append(record_builder(doc))
    return records


def _build_dataset(
    *,
    config_name: str,
    source_dataset_name: str,
    record_builder: Callable[[dict[str, Any]], dict[str, Any]],
    features,
    cache_dir: str | None = None,
    limit: int | None = None,
):
    source_dataset = _load_source_dataset(source_dataset_name, cache_dir=cache_dir)
    records = _build_records(source_dataset, record_builder, limit=limit)
    if not records:
        raise ValueError(f"No records built for config {config_name}")
    datasets = _datasets()
    return datasets.Dataset.from_list(records, features=features)


def build_mathvision_dataset(*, cache_dir: str | None = None, limit: int | None = None):
    return _build_dataset(
        config_name=MATHVISION_CONFIG,
        source_dataset_name=MATHVISION_SOURCE,
        record_builder=build_mathvision_record,
        features=_build_mathvision_features(),
        cache_dir=cache_dir,
        limit=limit,
    )


def build_mathvista_dataset(*, cache_dir: str | None = None, limit: int | None = None):
    return _build_dataset(
        config_name=MATHVISTA_CONFIG,
        source_dataset_name=MATHVISTA_SOURCE,
        record_builder=build_mathvista_record,
        features=_build_mathvista_features(),
        cache_dir=cache_dir,
        limit=limit,
    )


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Build the public math-prompt-in-image dataset.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--token", default=None)
    parser.add_argument("--mathvision-limit", type=int, default=None)
    parser.add_argument("--mathvista-limit", type=int, default=None)
    return parser.parse_args(argv)


def _save_local_dataset(dataset, output_dir: Path, config_name: str) -> Path:
    config_dir = output_dir / config_name
    config_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(config_dir))
    return config_dir


def _push_dataset_to_hub(dataset, *, repo_id: str, config_name: str, token: str | None):
    dataset.push_to_hub(repo_id=repo_id, config_name=config_name, split=SOURCE_SPLIT, token=token)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets_to_build = [
        (MATHVISION_CONFIG, build_mathvision_dataset, args.mathvision_limit),
        (MATHVISTA_CONFIG, build_mathvista_dataset, args.mathvista_limit),
    ]

    for config_name, builder, limit in datasets_to_build:
        dataset = builder(cache_dir=args.cache_dir, limit=limit)
        _save_local_dataset(dataset, output_dir, config_name)
        if args.push_to_hub:
            _push_dataset_to_hub(dataset, repo_id=args.repo_id, config_name=config_name, token=args.token)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
