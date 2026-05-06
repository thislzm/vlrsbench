# VLRS-Bench Data Construction Pipeline

This repository contains the data construction pipeline for VLRS-Bench. The scripts generate remote-sensing vision-language reasoning questions in four answer formats.

## Structure

- `yes_or_no/`: true/false question generation.
- `single_answer/`: single-choice question generation.
- `multi_answer/`: multi-answer question generation.
- `fill_blank/`: fill-in-the-blank question generation.

Each folder contains:

- `main_multiprocess.py`: multiprocessing entry point for generation.
- `common_part.py`: shared prompt and utility components.
- `*_dataset_prompt.py`: prompt templates for different dataset/prior settings.

## Usage

Set your OpenAI API key:

```bash
export OPENAI_API_KEY=your_api_key
```

Run the desired pipeline from its folder, for example:

```bash
cd multi_answer
python main_multiprocess.py
```

All `main_multiprocess.py` scripts use `gpt-5` as the generation model.
