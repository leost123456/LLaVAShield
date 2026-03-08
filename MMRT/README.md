## MMRT: Multimodal Multi-turn Red Teaming

This repository contains the official implementation of **MMRT**, a multimodal multi-turn red-teaming framework for efficiently generating unsafe multimodal multi-turn dialogues tageting multiple VLMs. The framework combines:

- **Attackers** that generate adversarial multimodal multi-turn conversations
- **Targets** that respond to the attacks, which is the models you want to evaluate
- **Evaluators** that score the safety of target responses
- **MCTS-based search** that adaptively explores attack path over multiple dialogue turns

## MMRT Structure

- **`main.py`**: Entry point. Parses command-line arguments, initializes `Config`, and runs:
  - **MCTS-based attack** (`pipline/pipeline.py`)
  - **Baseline pipeline** without MCTS (`pipline/pipeline_base.py`)
  - **Parallel pipeline** with Ray (`pipline/pipeline_parallel.py`)
- **`config.py`**: Global configuration paths 
- **`data/`**:
  - `data_loader.py`: Defines the `Task` and `Keyword` objects and loads JSON datasets into structured tasks.
- **`mcts/`**:
  - `mcts.py`: Monte Carlo Tree Search.
  - `node.py`: Node representation and tree utilities.
- **`models/`**:
  - `attacker/`:
    - `base_attacker.py`: Abstract attacker interface.
    - `qwen2_5vl_instruct_attacker.py`: Local Qwen2.5-VL attacker.
    - `qwen2_5vl_instruct_attacker_api.py`: API-based Qwen2.5-VL attacker.
    - `image_generator/`: 
      - `SD3_medium_generate.py`: SD3-based image generation utility.
      - `image_pool.py`: Ray-based SD3 worker pool for parallel image generation.
  - `target/`:
    - `base_target.py`: Abstract target interface.
    - `gpt4o_target.py`, `qwen2_5vl_instruct_target*.py`, `claude3_7sonnet_target.py`, `gemini2_5pro_target.py`,
      `gpt5mini_target.py`, `llava_onevision_target.py`, `internvl3_target.py`: different target models.
  - `evaluator/`:
    - `base_evaluator.py`: Abstract evaluator interface.
    - `gpt4o_evaluator.py`: GPT-4o-based automatic safety evaluator.
- **`pipline/`**:
  - `pipeline.py`: Main MCTS pipeline over tasks.
  - `pipeline_base.py`: A→T→E-loop pipeline without MCTS.
  - `pipeline_parallel.py`: Ray-based parallelization over tasks.
- **`prompts/`**:
  - `attack_prompt.py`: System prompt for the attacker model.
  - `evaluate_prompt.py`: System prompt for the evaluator model.
- **`utils/`**:
  - `utils.py`: JSON helpers, task status utilities, image encoding utilities.
  - `logger.py`: Logging configuration.
  - `timer.py`: Time spent on each task.
  - `time_limit.py`: Time limit for each MCTS run.
  - `intern_vl_utils.py`: InternVL Utilities.