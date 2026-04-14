# Math Prompt-in-Image Benchmark

This benchmark publishes prompt-in-image evaluation splits for MathVision and MathVista as a self-contained Hugging Face dataset:

- dataset repo: `YongxinWang/math-prompt-in-image`
- configs:
  - `mathvision_testmini_prompt_in_image`
  - `mathvista_testmini_prompt_in_image`
- split: `testmini`

The dataset stores the rendered prompt-in-image artwork directly in `decoded_image` as an HF `Image` feature. The public `lmms-eval` tasks consume that image directly and do not re-render prompt text at eval time.

## Public Tasks

- `mathvision_testmini_prompt_in_image`
- `mathvista_testmini_prompt_in_image`
- group: `math_prompt_in_image_testmini`

## Run the Benchmark

The quickest way is to use the dedicated example script:

```bash
bash examples/models/qwen3vl_math_prompt_in_image.sh
```

The script includes separate sections for:

- the benchmark group `math_prompt_in_image_testmini`
- the two single tasks

It also supports environment overrides such as `MODEL`, `TASKS`, `NUM_PROCESSES`, and `OUTPUT_PATH`.

If you want to run the command directly:

```bash
accelerate launch -m lmms_eval \
  --model qwen3_vl \
  --model_args pretrained=Qwen/Qwen3-VL-4B-Instruct,attn_implementation=flash_attention_2,interleave_visuals=False \
  --tasks math_prompt_in_image_testmini \
  --batch_size 1
```

You can also run the two tasks directly:

```bash
accelerate launch -m lmms_eval \
  --model qwen3_vl \
  --model_args pretrained=Qwen/Qwen3-VL-4B-Instruct,attn_implementation=flash_attention_2,interleave_visuals=False \
  --tasks mathvision_testmini_prompt_in_image,mathvista_testmini_prompt_in_image \
  --batch_size 1
```

## Rebuild the HF Dataset

The builder lives in `lmms_eval/tasks/math_prompt_in_image/build_hf_dataset.py` and uses only the public source datasets `MathLLMs/MathVision` and `AI4Math/MathVista`.

```bash
python -m lmms_eval.tasks.math_prompt_in_image.build_hf_dataset \
  --output-dir /tmp/math-prompt-in-image-build
```

To push the dataset to the Hub:

```bash
python -m lmms_eval.tasks.math_prompt_in_image.build_hf_dataset \
  --output-dir /tmp/math-prompt-in-image-build \
  --push-to-hub \
  --repo-id YongxinWang/math-prompt-in-image \
  --token "$HF_TOKEN"
```
