<h1 align="center">
  <img src="https://raw.githubusercontent.com/leost123456/LLaVAShield/main/figs/logo.png" height="38" style="vertical-align: middle; margin-right: 8px;" alt="LLaVAShield Logo">
  LLaVAShield: Safeguarding Multimodal Multi-Turn Dialogues in Vision-Language Models
</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2509.25896">
    <img src="https://img.shields.io/badge/Paper-arXiv-B31B1B?logo=arxiv&logoColor=white" alt="Paper">
  </a>
  <a href="https://huggingface.co/RealSafe/LLaVAShield-v1.0-7B">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-LLaVAShield--7B-FFD21E" alt="Model">
  </a>
  <a href="https://huggingface.co/datasets/leost233/MMDS">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-MMDS-FFD21E" alt="Dataset">
  </a>
  <a href="https://github.com/leost123456/LLaVAShield">
    <img src="https://img.shields.io/badge/Code-GitHub-181717?logo=github&logoColor=white" alt="Code">
  </a>
</p>

Official implementation of "LLaVAShield: Safeguarding Multimodal Multi-Turn Dialogues in Vision-Language Models" (CVPR, 2026), a dedicated content moderation framework designed to evaluate safety risks in interactive VLM conversations.

![](https://github.com/leost123456/LLaVAShield/blob/main/figs/overall%20workflow.png?raw=true)

## 📑 Table of Contents
- [⚙️ Installation](#️-environment-setup)
- [⚔️ MMRT Framework](#️-mmrt-framework)
  - [1. API & Model Configuration](#1-api--model-configuration)
  - [2. Data Preparation](#2-data-preparation)
  - [3. Usage & Execution](#3-usage--execution)
- [🧠 LLaVAShield Model](#-llavashield-model)
- [📝 Citation](#-citation)

---

## ⚙️ Installation

1.Clone the repository 

```bash
git clone https://github.com/leost123456/LLaVAShield.git
cd LLaVAShield
```
2.Set up the Conda environment and install dependencies

```bash
conda create -n llavashield python=3.10 -y
conda activate llavashield
pip install -e ".[train]"
```

---

## ⚔️ MMRT Framework

### 1. API & Model Configuration

#### 1.1 API Endpoints
This repository uses the **OpenAI-compatible Chat Completions API** for multiple providers (OpenAI, Qwen, Claude, Gemini, etc.). All targets and evaluators are built on top of the `openai` Python client. Set your environment variables accordingly:

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
export OPENAI_BASE_URL="https://YOUR_ENDPOINT/v1"
```

#### 1.2 Local Model Checkpoints
Some components rely on local checkpoints. Please configure them in `config.py`:

**Required Models:**
- `MODEL_PATHS["qwen2_5vl_72b_instruct"]`: Local path for Qwen2.5-VL-Instruct (download from [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct); smaller Qwen2.5-VL variants are also supported).
- `MODEL_PATHS["stable-diffusion-3.5-medium"]`: Local path for Stable Diffusion 3.5 Medium (Download from [Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)).

**Optional Models:**
Configure the remaining model paths in `MODEL_PATHS` only if you intend to use them as target models.

### 2. Data Preparation

We provide the MMRT Input Task Dataset, available at [MMRT-Data](https://huggingface.co/datasets/leost233/MMRT-Data), which contains 348 malicious intent tasks in JSON format along with their associated image datasets. Extract the downloaded datasets into your `data/` directory.

**2.1 Data Structure :**
To give you an idea of what the data looks like, each task sample in the JSON dataset is structured as follows:

```json
{
  "id": 1,
  "dimension": "Safety Dimension",
  "subdimension": "Sub-dimension Detail",
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

**2.2 Image Directory Structure:**
```text
data/images/
  ├── <task_id_1>/
  │   ├── 000001.jpg
  │   └── ...
  └── <task_id_2>/
      └── ...
```

**2.3 Configuration:**
Update the paths in `config.py` to point to your data:
- `DATA_PATHS["test_data"] = "data/MMRT_Data.json"`
- `IMAGE_PATHS["test_images"] = "data/images"`

### 3. Usage & Execution

#### 3.1 Running the Pipeline

You can run the MMRT framework in three different modes depending on your requirements:

**Option A: MCTS-based Attack (Default)**
Run the MCTS-based attack using 
Qwen2.5-VL-72B-Instruct as the attacker, GPT-4o as both the target and evaluator:
```bash
python -m MMRT.main \
  --attacker qwen \
  --target gpt4o \
  --evaluator gpt4o \
  --max_refused 3 \
  --max_rounds 10 \
  --sim_rounds 1 \
  --max_children 2 \
  --iterations 20
```

**Option B: Baseline Pipeline (No MCTS)**
Run a single attack trajectory per task without the MCTS search algorithm:
```bash
python -m MMRT.main \
  --run_base \
  --attacker qwen \
  --target gpt4o \
  --evaluator gpt4o
```

**Option C: Parallel Pipeline with Ray**
Enable Ray-based parallelization and concurrent SD3 image-generation workers for faster execution:
```bash
python -m MMRT.main \
  --run_pipline_parallel \
  --attacker qwen \
  --target gpt4o \
  --evaluator gpt4o \
  --gpus "0,1,2,3" \
  --slots_per_gpu 1 \
  --max_workers 4
```

#### 3.2 Command-Line Arguments

Key arguments available in `main.py`:

| Category | Argument | Description |
| :--- | :--- | :--- |
| **Data & I/O** | `--data_path` | Path to the input JSON file. |
| | `--images_dir` | Directory containing input images. |
| | `--output_dir` | Directory to save results. |
| **Models** | `--attacker` | Attacker model (`qwen`, etc.). |
| | `--target` | Target model (`gpt4o`, `qwen`, `claude`, `gemini`, `gpt5mini`, `llava`, `intern`, etc.). |
| | `--evaluator` | Evaluator model (`gpt4o`, etc.). |
| **MCTS Search**| `--max_refused` | Maximum number of target refusals before skipping. |
| | `--max_rounds` | Maximum conversation rounds per attack. |
| | `--sim_rounds` | Rollout depth in MCTS. |
| | `--max_children` | Maximum number of children per node. |
| | `--iterations` | Number of MCTS iterations per task. |
| **Execution** | `--run_base` | Flag to run the baseline pipeline without MCTS. |
| | `--run_pipline_parallel` | Flag to run the parallel pipeline with Ray. |
| **Local / Ray** | `--call_local_attacker` | Use local Qwen2.5-VL attacker instead of API. |
| | `--call_local_target` | Use local Qwen2.5-VL target instead of API. |
| | `--gpus` | Comma-separated GPU IDs for Ray (e.g., `"0,1,2,3"`). |
| | `--slots_per_gpu` | Concurrent SD3 workers per GPU. |
| | `--max_workers` | Maximum Ray SD3 worker actors. |

#### 3.3 Outputs & Results

All execution results will be automatically saved in the `results/` directory:
- **Conversation Logs:** Per-task conversation JSON files are saved under `results/.../tasks/`.
- **Generated Media:** Images generated during the attack are saved in `results/generated_images/`.
- **Search Logs:** MCTS operation logs are stored in `results/mcts_logs.txt`.

---

## 🧠 LLaVAShield Model

### Inference
To run inference with our safeguarded model, ensure you have downloaded the weights from Hugging Face and run the provided demo script:
```bash
python demo.py 
```

### Training
The training code is located in the `llavashield/` directory, where we provide training scripts for your reference. Both the codebase and the training data formatting are built upon the [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) framework. For more comprehensive details, please refer to the original repository.

Please ensure you manually install the **flash-attn** package before starting the training process.

---

## 📝 Citation
If you find our work helpful, please consider citing our paper and leaving a star on this repository! 🌟

```bibtex
@misc{huang2025llavashield,
      title={LLaVAShield: Safeguarding Multimodal Multi-Turn Dialogues in Vision-Language Models}, 
      author={Guolei Huang and Qinzhi Peng and Gan Xu and Yuxuan Lu and Yongjun Shen},
      year={2025},
      eprint={2509.25896},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```