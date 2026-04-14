"""Smoke-validate the public math prompt-in-image HF dataset."""

from __future__ import annotations

import argparse

from datasets import load_dataset


DEFAULT_REPO_ID = "YongxinWang/math-prompt-in-image"
DEFAULT_SPLIT = "testmini"
DEFAULT_CONFIGS = (
    "mathvision_testmini_prompt_in_image",
    "mathvista_testmini_prompt_in_image",
)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Smoke-validate the public math prompt-in-image HF dataset.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIGS))
    return parser.parse_args(argv)


def validate_config(repo_id: str, config_name: str, split: str) -> dict[str, object]:
    dataset = load_dataset(repo_id, name=config_name, split=split)
    if len(dataset) == 0:
        raise ValueError(f"{config_name} loaded but is empty")

    row = dataset[0]
    if "decoded_image" not in row:
        raise KeyError(f"{config_name} is missing decoded_image")
    if "question_id" not in row:
        raise KeyError(f"{config_name} is missing question_id")
    if "question" not in row:
        raise KeyError(f"{config_name} is missing question")
    if "answer" not in row:
        raise KeyError(f"{config_name} is missing answer")
    if not hasattr(row["decoded_image"], "size"):
        raise TypeError(f"{config_name} decoded_image did not load as an image-like object")

    return {
        "config": config_name,
        "rows": len(dataset),
        "keys": sorted(row.keys()),
        "image_size": tuple(row["decoded_image"].size),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for config_name in args.configs:
        result = validate_config(args.repo_id, config_name, args.split)
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
