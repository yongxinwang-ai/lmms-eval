from pathlib import Path

import pandas as pd
import yaml

from lmms_eval.tasks._task_utils.prompt_in_image import (
    MINIMAL_SOLVE_PROMPT,
    render_canvas_control_on_image,
    render_question_on_image,
    render_question_on_image_with_panel_crop,
)
from lmms_eval.tasks.mathvista.mathvista_evals import MathVistaEvaluator

with open(Path(__file__).parent / "mathvista.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))


mathvista_evaluator = MathVistaEvaluator()


def _get_doc_image(doc):
    image = doc.get("decoded_image") or doc.get("image")
    if image is None:
        raise KeyError("Expected `decoded_image` or `image` in dataset document")
    return image.convert("RGB")


def _get_doc_question_id(doc):
    question_id = doc.get("question_id")
    if question_id is None:
        question_id = doc.get("pid")
    if question_id is None:
        raise KeyError("Expected `question_id` or legacy `pid` in dataset document")
    return question_id


def mathvista_doc_to_visual(doc):
    return [_get_doc_image(doc)]


def mathvista_doc_to_visual_prompt_in_image(doc, lmms_eval_specific_kwargs=None):
    image = _get_doc_image(doc)
    question = doc.get("question", "")
    return [render_question_on_image(image, question)]


def mathvista_doc_to_visual_prompt_in_image_qpad(doc, lmms_eval_specific_kwargs=None):
    image = _get_doc_image(doc)
    question = doc.get("question", "")
    full_image, panel_crop = render_question_on_image_with_panel_crop(image, question)
    return [full_image, panel_crop]


def mathvista_doc_to_visual_canvas_control(doc, lmms_eval_specific_kwargs=None):
    image = _get_doc_image(doc)
    question = doc.get("question", "")
    return [render_canvas_control_on_image(image, question)]


def mathvista_doc_to_text_minimal(doc, lmms_eval_specific_kwargs=None):
    return MINIMAL_SOLVE_PROMPT


def mathvista_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    problem = {
        "question_type": doc["question_type"],
        "answer_type": doc["answer_type"],
        "question": doc["question"],
        "unit": doc["unit"] if "unit" in doc else "",
        "caption": doc["caption"] if "caption" in doc else "",
        "ocr": doc["ocr"] if "ocr" in doc else "",
        "choices": doc["choices"],
        "answer": doc["answer"] if "answer" in doc else None,
        "precision": doc["precision"] if "precision" in doc else 0,
    }
    query_prompt = mathvista_evaluator.create_one_query(
        problem,
        shot_num=lmms_eval_specific_kwargs["shot"],
        shot_type=lmms_eval_specific_kwargs["shot_type"],
        use_caption=lmms_eval_specific_kwargs["use_caption"],
        use_ocr=lmms_eval_specific_kwargs["use_ocr"],
    )
    return query_prompt


def mathvista_process_results(doc, results):
    prediction = results[0].strip()
    problem = {
        "question_type": doc["question_type"],
        "answer_type": doc["answer_type"],
        "query": doc["query"],
        "choices": doc["choices"],
        "answer": doc["answer"] if "answer" in doc else None,
        "precision": doc["precision"] if "precision" in doc else 0,
    }
    extraction = mathvista_evaluator.extract_answer(prediction, problem, config["metadata"]["quick_extract"])

    prediction = mathvista_evaluator.normalize_extracted_answer(extraction, problem["choices"], problem["question_type"], problem["answer_type"], problem["precision"])
    # set test set answer to None
    true_false = mathvista_evaluator.safe_equal(prediction, problem["answer"]) if problem["answer"] is not None else False

    result = {
        "question_id": _get_doc_question_id(doc),
        "query": doc["query"],
        "choices": doc["choices"],
        "answer": doc["answer"] if "answer" in doc else None,
        "extraction": extraction,
        "prediction": prediction,
        "true_false": true_false,
        "question_type": doc["question_type"],
        "answer_type": doc["answer_type"],
        "precision": doc["precision"] if "precision" in doc else 0,
        "metadata": doc["metadata"],
    }

    return {
        "llm_as_judge_eval": result,
        "submission": result,
    }


def mathvista_aggregate_results(results, args, *, calculate_gain=False, random_scores=None):
    split_flag = results[0]["metadata"]["split"]
    full_pids = [result["question_id"] for result in results]
    total = len(results)
    correct = sum(1 for idx, pid in enumerate(full_pids) if results[idx]["true_false"])
    accuracy = round(correct / total * 100, 2)
    scores = {"average": {"accuracy": accuracy, "correct": correct, "total": total}}

    for result in results:
        result.update(result.pop("metadata"))

    results_dict = {result["question_id"]: result for result in results}
    df = pd.DataFrame(results_dict).T
    target_keys = ["question_type", "answer_type", "language", "source", "category", "task", "context", "grade", "skills"]

    for key in target_keys:
        values = df[key].explode().unique() if key == "skills" else df[key].unique()
        scores[key] = {}
        for value in values:
            correct, total, acc = mathvista_evaluator.get_acc_with_contion(df, key, value)
            if total > 0:
                scores[key][value] = {"accuracy": acc, "correct": correct, "total": total}
        scores[key] = dict(sorted(scores[key].items(), key=lambda item: float(item[1]["accuracy"]), reverse=True))

    if calculate_gain:
        for key in scores:
            if key == "average":
                gain = round(float(scores[key]["accuracy"]) - float(random_scores[key]["accuracy"]), 2)
                scores[key]["acc_gain"] = gain
            else:
                for sub_key in scores[key]:
                    gain = round(float(scores[key][sub_key]["accuracy"]) - float(random_scores[key][sub_key]["accuracy"]), 2)
                    scores[key][sub_key]["acc_gain"] = gain

    if scores["average"]["accuracy"] == 0:
        return None
    return scores["average"]["accuracy"]
