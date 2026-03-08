import os

class Config:
    data_path = None
    images_dir = None
    output_dir = "results"
    tasks_output_dir = os.path.join("results", "tasks")
    images_output_dir = os.path.join("results", "generated_images")
    skip_score_above = 0

    @classmethod
    def init_config(cls, args): 
        cls.data_path = args.data_path
        cls.images_dir = args.images_dir
        if args.output_dir is None:
            cls.output_dir = OUTPUT_PATHS[args.target]
        else:
            cls.output_dir = args.output_dir
        cls.tasks_output_dir = os.path.join(cls.output_dir, "tasks")
        cls.images_output_dir = os.path.join(cls.output_dir, "generated_images")
        cls.skip_score_above = args.skip_score_above

        os.makedirs(cls.output_dir, exist_ok=True)
        os.makedirs(cls.tasks_output_dir, exist_ok=True)
        os.makedirs(cls.images_output_dir, exist_ok=True)


MODEL_PATHS = {
    "InternVL3-78B": "put/your/path/here", # Optionally, if the target model is InternVL3, a local path needs to be provided.
    "llava-onevision-qwen2-72b-ov-chat-hf": "put/your/path/here", # Optionally, if the target model is LLaVA OneVision, a local path needs to be provided.
    "qwen2_5vl_72b_instruct": "put/your/path/here",
    "stable-diffusion-3.5-medium": "put/your/path/here",
}


DATA_PATHS = {
    "test_data": "data/MMRT_Data.json"
}

IMAGE_PATHS = {
    "test_images": "data/images"
}

OUTPUT_PATHS = {
    "outputs": "results",
    "qwen": "results/qwen",
    "gpt4o": "results/gpt4o",
    "test_gpt4o": "results/test/gpt4o",
    "test_qwen": "results/test/qwen",
    "test_claude": "results/test/claude",
    "test_gemini": "results/test/gemini",
    "test_gpt5mini": "results/test/gpt5mini",
    "test_llava": "results/test/llava",
    "test_intern": "results/test/intern",
    "test_gpt4o_base": "results/test/gpt4o_base",
}
