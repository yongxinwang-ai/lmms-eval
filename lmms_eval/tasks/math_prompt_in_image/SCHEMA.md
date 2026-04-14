# Math Prompt-in-Image HF Schema

## Repo
- dataset_path: `YongxinWang/math-prompt-in-image`

## Configs
- `mathvision_testmini_prompt_in_image`
- `mathvista_testmini_prompt_in_image`

## Split
- `testmini`

## Required columns

### Shared
- `question_id`: string
- `decoded_image`: HF `Image` feature
- `question`: string
- `answer`: string
- `source_split`: string

### MathVision-specific
- `options`: list[string]

### MathVista-specific
- `query`: string
- `choices`: list[string]
- `question_type`: string
- `answer_type`: string
- `precision`: int
- `metadata`: dict
