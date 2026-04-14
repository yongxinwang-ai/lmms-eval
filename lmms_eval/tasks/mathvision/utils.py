import os
import re

from loguru import logger as eval_logger

from lmms_eval.llm_judge import ServerConfig, get_server
from lmms_eval.tasks._task_utils.prompt_in_image import (
    MINIMAL_SOLVE_PROMPT,
    render_canvas_control_on_image,
    render_question_on_image,
    render_question_on_image_with_panel_crop,
)

try:
    from lmms_eval.tasks.mathvision.eval_utils import (
        find_math_answer,
        is_equal,
        is_number,
    )
except ImportError as e:
    eval_logger.warning(f"Error importing eval_utils from lmms_eval.tasks.mathvision.eval_utils: {e}")

    def is_number(s):
        try:
            float(str(s).replace(",", ""))
            return True
        except (TypeError, ValueError):
            return False

    def find_math_answer(text):
        text = str(text).strip()
        boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
        if boxed:
            return boxed[-1].strip()
        numeric = re.findall(r"[-+]?\d*\.?\d+(?:/\d+)?", text.replace(",", ""))
        if numeric:
            return numeric[-1].strip()
        return text

    def is_equal(a, b):
        a = str(a).strip().lower()
        b = str(b).strip().lower()
        if a == b:
            return True
        if is_number(a) and is_number(b):
            return abs(float(a.replace(",", "")) - float(b.replace(",", ""))) < 1e-6
        return False

NUM_SECONDS_TO_SLEEP = 5


def _get_doc_image(doc):
    image = doc.get("decoded_image") or doc.get("image")
    if image is None:
        raise KeyError("Expected `decoded_image` or `image` in dataset document")
    return image.convert("RGB")

# Initialize the judge server
API_TYPE = os.getenv("API_TYPE", "openai")
GPT_MODEL = os.getenv("MODEL_VERSION", "gpt-4o-2024-11-20")

server_config = ServerConfig(
    model_name=GPT_MODEL,
)
server = get_server(server_name=API_TYPE, config=server_config)


def mathvision_doc_to_visual(doc):
    return [_get_doc_image(doc)]


def mathvision_doc_to_visual_prompt_in_image(doc, lmms_eval_specific_kwargs=None):
    image = _get_doc_image(doc)
    question = doc.get("question", "")
    return [render_question_on_image(image, question)]


def mathvision_doc_to_visual_prompt_in_image_qpad(doc, lmms_eval_specific_kwargs=None):
    image = _get_doc_image(doc)
    question = doc.get("question", "")
    full_image, panel_crop = render_question_on_image_with_panel_crop(image, question)
    return [full_image, panel_crop]


def mathvision_doc_to_visual_canvas_control(doc, lmms_eval_specific_kwargs=None):
    image = _get_doc_image(doc)
    question = doc.get("question", "")
    return [render_canvas_control_on_image(image, question)]


def mathvision_doc_to_text_minimal(doc, lmms_eval_specific_kwargs=None):
    return MINIMAL_SOLVE_PROMPT


def mathvision_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question, choices = doc["question"], doc["options"]
    len_choices = len(choices)
    options = [chr(ord("A") + i) for i in range(len_choices)]
    choices_str = "\n".join([f"{option}. {choice}" for option, choice in zip(options, choices)])

    mc_prompt = ""
    if lmms_eval_specific_kwargs is not None:
        mc_prompt = "\n" + lmms_eval_specific_kwargs["mc_prompt"]

    query_prompt = 'Please solve the problem step by step and put your answer in one "\\boxed{}".'
    if choices_str:
        query_prompt += f"{question}\nChoices: {choices_str}" + mc_prompt
    else:
        query_prompt += question
    return query_prompt


def mathvision_gpt_eval_process_results(doc, results):
    correct_list = []
    for pred in results:
        model_answer = pred.strip()
        gt_answer = str(doc["answer"])
        question = doc["question"]

        try:
            # Use the llm_judge API for binary evaluation
            result = server.evaluate_binary(question=question, answer=gt_answer, prediction=model_answer, output_format="0/1")

            # Parse the result
            if result["success"]:
                judge_response = result["result"]
                correct_list.append(judge_response)
            else:
                eval_logger.error(f"Judge evaluation failed: {result.get('raw_response', 'Unknown error')}")
                correct_list.append(False)

        except Exception as e:
            eval_logger.error(f"Error getting judge response: {e}")
            correct_list.append(False)

    # Calculate the average score for this document
    avg_score = sum(1 if score else 0 for score in correct_list) / len(correct_list) if correct_list else 0
    return {"llm_as_judge_eval": avg_score}


def mathvision_process_results(doc, results):
    correct_list = []
    for pred in results:
        model_answer = pred.strip()

        gt_answer = str(doc["answer"])
        if len(doc["options"]) > 0:
            gt_answer_value = doc["options"][ord(gt_answer) - ord("A")]
        else:
            gt_answer_value = ""

        for c in "ABCDE":
            if model_answer.endswith(f" {c}.") or model_answer.endswith(f" ({c}).") or model_answer.startswith(f"{c}\n") or model_answer.startswith(f"({c})\n") or model_answer.startswith(f"({c}) {c}\n"):
                model_answer = c
        if is_number(model_answer.split("is ")[-1].rstrip(".")):
            model_answer = model_answer.split("is ")[-1].rstrip(".")
        if "oxed{" not in model_answer:
            for flag in ["the final answer is", "the answer is", "the correct answer is", "the answer should be"]:
                raw_model_answer = model_answer
                model_answer = model_answer.split(flag)[-1].strip()
                if flag in raw_model_answer:
                    model_answer = model_answer.split("\n")[0].split(". ")[0]
                flag = flag.replace("the", "The")
                raw_model_answer = model_answer
                model_answer = model_answer.split(flag)[-1].strip()
                if flag in raw_model_answer:
                    model_answer = model_answer.split("\n")[0].split(". ")[0]
        elif model_answer.count("oxed{") > 1:
            model_answer = "\\boxed{" + model_answer.split("oxed{")[-1]

        model_answer = (
            find_math_answer(model_answer)
            .replace("(a)", "a")
            .replace("(b)", "b")
            .replace("(c)", "c")
            .replace("(d)", "d")
            .replace("(e)", "e")
            .replace("{a}", "a")
            .replace("{b}", "b")
            .replace("{c}", "c")
            .replace("{d}", "d")
            .replace("{e}", "e")
            .rstrip(".")
            .lstrip(":")
            .strip()
        )
        correct = is_equal(gt_answer, model_answer) or is_equal(gt_answer_value, model_answer)
        correct_list.append(correct)
    return {
        "mathvision_standard_eval": {
            # "question": doc["question"],
            # "answer": doc["answer"],
            "response": results,
            # "subject": doc["subject"],
            # "level": doc["level"],
            "scores": correct_list,
        },
    }


def mathvision_aggregate_results_eval(results):
    total = len(results)
    correct = sum(1 for idx, result in enumerate(results) if results[idx]["scores"][0])
    accuracy = round(correct / total * 100, 2)
    return accuracy
