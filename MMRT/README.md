## MMRT: Multimodal Multi-turn Red Teaming

This repository contains the official implementation of **MMRT**, a multimodal multi-turn red-teaming framework for efficiently generating unsafe multimodal multi-turn dialogues tageting multiple VLMs. The framework combines:

- **Attackers** that generate adversarial multimodal multi-turn conversations
- **Targets** that respond to the attacks, which is the models you want to evaluate
- **Evaluators** that score the safety of target responses
- **MCTS-based search** that adaptively explores attack path over multiple dialogue turns


---

## 1. Environment Setup

### 1.1 Create environment

```bash
conda create -n mmrt python=3.10 -y
conda activate mmrt

pip install -r requirements.txt
```

### 1.2 API keys and endpoints

This repository uses the **OpenAI-compatible Chat Completions API** for multiple providers (OpenAI, Qwen, Claude, Gemini, etc.). All targets and evaluators are built on top of the `openai` Python client.

Set the following environment variables:

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
export OPENAI_BASE_URL="https://YOUR_ENDPOINT/v1"
```

### 1.3 Local models and checkpoints

Some components rely on local checkpoints configured in `config.py`:

e.g.
- `MODEL_PATHS["qwen2_5vl_72b_instruct"]` – local path for Qwen2.5-VL-Instruct

Edit `config.py` and replace `"put/your/path/here"` entries with your actual directories.

---

## 2. Repository Structure

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

You can extend the framework by adding new attackers, targets, or evaluators under `models/`.

---

## 3. Data Format and Preparation

### 3.1 Task JSON format

The dataset loader (`data/data_loader.py`) expects a JSON file containing a list of task objects. Each object should follow:

```json
{
  "id": 1,
  "dimension": "XXX",
  "subdimension": "YYY",
  "definition": "Definition of this safety dimension.",
  "example": "Example description or conversation.",
  "malicious_intent": "Natural language description of the harmful objective.",
  "keywords": [
    {
      "text": "keyword string",
      "image_id": "000001.jpg",
      "caption": "Short caption for this image."
    }
  ]
}
```

### 3.2 Image directory layout

Images are organized under the directory specified in `IMAGE_PATHS` inside `config.py`. For example, if you use:

- `DATA_PATHS["test_data"] = "data/test_data.json"`
- `IMAGE_PATHS["test_images"] = "data/test_images"`

and you run with `--data_path DATA_PATHS["test_data"]` and `--images_dir IMAGE_PATHS["test_images"]`, then the expected image structure is:

```text
data/
  test_data.json
  test_images/
    <task_id>/
      000001.jpg
      000002.jpg
      ...
```

The `image_id` fields in the JSON should match the filenames within each `<task_id>/` directory.

Generated images are saved under:

```text
results/
  generated_images/
    <task_id>/
      gen_000001.jpg
      gen_000002.jpg
      ...
```

The exact root directory is controlled by `Config.images_output_dir`.

---

## 4. How to Run

All scripts are launched via `main.py`.

### 4.1 Basic arguments

Key arguments in `main.py`:

- **Data & I/O**
  - `--data_path`: path to input JSON file
  - `--images_dir`: directory containing input images 
  - `--output_dir`: directory to save results 
- **Models**
  - `--attacker`: attacker model (`qwen`, …)
  - `--target`: target model (`gpt4o`, `qwen`, `claude`, `gemini`, `gpt5mini`, `llava`, `intern`, …)
  - `--evaluator`: evaluator model (`gpt4o`, …)
- **Search hyperparameters**
  - `--max_refused`: max number of target refusals before giving up
  - `--max_rounds`: max conversation rounds per attack
  - `--sim_rounds`: rollout depth in MCTS
  - `--max_children`: max number of children per node
  - `--iterations`: number of MCTS iterations per task
- **Execution mode**
  - `--run_base`: run baseline pipeline without MCTS
  - `--run_pipline_parallel`: run parallel pipeline with Ray over task
  - (no flag) → default MCTS pipeline

Additional arguments:

- `--call_local_attacker`: use local Qwen2.5-VL attacker
- `--call_local_target`: use local Qwen2.5-VL target
- Ray-related:
  - `--gpus`: comma-separated GPU IDs, e.g. `"0,1,2,3"`
  - `--slots_per_gpu`: concurrent SD3 workers per GPU
  - `--max_workers`: maximum Ray SD3 worker actors

### 4.2 Run MCTS-based attack

Example: Qwen attacker, GPT-4o target and evaluator, using default test data and images:

```bash
python main.py \
  --attacker qwen \
  --target gpt4o \
  --evaluator gpt4o \
  --max_refused 3 \
  --max_rounds 10 \
  --sim_rounds 1 \
  --max_children 2 \
  --iterations 20
```

Outputs:

- Per-task conversation JSON files under `results/.../tasks/`
- MCTS logs under `results/mcts_logs.txt`

### 4.3 Run baseline pipeline (no MCTS)

```bash
python main.py \
  --run_base \
  --attacker qwen \
  --target gpt4o \
  --evaluator gpt4o
```

This runs a single attack trajectory per task without MCTS search.

### 4.4 Run parallel pipeline with Ray

To enable Ray-based parallelization and SD3 image-generation workers:

```bash
python main.py \
  --run_pipline_parallel \
  --attacker qwen \
  --target gpt4o \
  --evaluator gpt4o \
  --gpus "0,1,2,3" \
  --slots_per_gpu 1 \
  --max_workers 4
```

---

## 5. Reproducibility Notes

- **Randomness**:
  - MCTS search and sampling in attacker models introduce randomness.

---

```

