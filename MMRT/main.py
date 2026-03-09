import os
import argparse

from .config import Config, MODEL_PATHS, DATA_PATHS, IMAGE_PATHS, OUTPUT_PATHS

def build_attacker(args, sd3_pool=None):
    if args.attacker.lower() == "qwen":     
        if args.call_local_attacker:
            from .models.attacker.qwen2_5vl_instruct_attacker import Qwen2_5VLInstructAttacker

            return Qwen2_5VLInstructAttacker(
                model_path=MODEL_PATHS["qwen2_5vl_72b_instruct"]
            )
        elif args.run_pipline_parallel:
            from .models.attacker.qwen2_5vl_instruct_attacker_api import Qwen2_5VLInstructAttacker 

            return Qwen2_5VLInstructAttacker(
                sd3_pool=sd3_pool
            )
        else:
            from .models.attacker.qwen2_5vl_instruct_attacker_api import Qwen2_5VLInstructAttacker 

            return Qwen2_5VLInstructAttacker()

    raise NotImplementedError("Unsupported attacker model.")

def build_target(args):
    if args.target.lower() == "gpt4o":
        from .models.target.gpt4o_target import GPT4oTarget

        return GPT4oTarget()
    
    if args.target.lower() == "qwen":
        if args.call_local_target:
            from .models.target.qwen2_5vl_instruct_target import Qwen2_5VLInstructTarget

            return Qwen2_5VLInstructTarget(model_path=MODEL_PATHS["qwen2_5vl_72b_instruct"])
        else:
            from .models.target.qwen2_5vl_instruct_target_api import Qwen2_5VLInstructTarget

            return Qwen2_5VLInstructTarget()

    if args.target.lower() == "claude":
        from .models.target.claude3_7sonnet_target import Claude3_7SonnetTarget

        return Claude3_7SonnetTarget()
    
    if args.target.lower() == "gemini":
        from .models.target.gemini2_5pro_target import Gemini2_5ProTarget

        return Gemini2_5ProTarget()
    
    if args.target.lower() == "gpt5mini":
        from .models.target.gpt5mini_target import GPT5MiniTarget

        return GPT5MiniTarget()
          
    if args.target.lower() == "llava":
        from .models.target.llava_onevision_target import LlavaOnevisionTarget

        return LlavaOnevisionTarget(
            model_path=MODEL_PATHS["llava-onevision-qwen2-72b-ov-chat-hf"]
        )
    
    if args.target.lower() == "intern":
        from .models.target.internvl3_target import InternVL3Target

        return InternVL3Target(
            model_path=MODEL_PATHS["InternVL3-78B"]
        )
    
    raise NotImplementedError("Unsupported target (use: gpt4o/llava/qwen/intern).")

def build_evaluator(args):
    if args.evaluator.lower() == "gpt4o":
        from .models.evaluator.gpt4o_evaluator import GPT4oEvaluator

        return GPT4oEvaluator()
    
    raise NotImplementedError("Only gpt4o evaluator is implemented.")

def main():
    parser = argparse.ArgumentParser(description="Run Multimodal Attack Pipeline.")
    parser.add_argument("--data_path", type=str, default=DATA_PATHS["test_data"], help="Path to the input JSON file")
    parser.add_argument("--images_dir", type=str, default=IMAGE_PATHS["test_images"], help="Directory to input images referenced by the task")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_PATHS["test_gpt4o_base"], help="Directory to save results")

    parser.add_argument("--attacker", type=str, default="qwen", help="Attacker model (gpt4o/qwen/...)")
    parser.add_argument("--target", type=str, default="gpt4o", help="Target model (gpt4o/gpt5mini/gemini/claude/llava/qwen/intern/...)")
    parser.add_argument("--evaluator", type=str, default="gpt4o", help="Evaluator model (gpt4o/...)")
    parser.add_argument("--max_refused", type=int, default=3, help="Maximum number of target refused to respond")
    parser.add_argument("--max_rounds", type=int, default=10, help="Maximum round number of conversation per attack")
    parser.add_argument("--sim_rounds", type=int, default=1, help="Number of rounds to simulate per candidate during MCTS rollout")
    parser.add_argument("--max_children", type=int, default=2, help="Maximum number of children nodes per MCTS node")
    parser.add_argument("--iterations", type=int, default=20, help="Total number of simulations per attack task")

    parser.add_argument("--call_local_attacker", action="store_true", help="")
    parser.add_argument("--call_local_target", action="store_true", help="")

    parser.add_argument("--run_base", action="store_true", help="")

    parser.add_argument("--run_pipline_parallel", action="store_true", help="")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7", help="Worker GPU ids, e.g., '1,2,3'.")
    parser.add_argument("--slots_per_gpu", type=int, default=1)
    parser.add_argument("--max_workers", type=int, default=4)

    parser.add_argument("--skip_score_above", type=int, default=1, help="")
         
    args = parser.parse_args()
    
    # Initialize configuration
    Config.init_config(args)

    # Run mcts with parallel tasks
    if args.run_pipline_parallel:
        from .pipline.pipeline_parallel import run_parallel
        try:
            import ray
            from .models.attacker.image_generator.image_pool import RaySD3Pool

            gpu_ids = [x.strip() for x in args.gpus.split(",") if x.strip() != ""]
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)

            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True, num_gpus=len(gpu_ids))

            sd3_pool = RaySD3Pool(gpus=len(gpu_ids), slots_per_gpu=args.slots_per_gpu, actors=args.max_workers)

            run_parallel(
                    attacker_factory=lambda: build_attacker(args, sd3_pool),
                    target_factory=lambda: build_target(args),
                    evaluator_factory=lambda: build_evaluator(args),
                    max_refused=args.max_refused,
                    max_rounds=args.max_rounds,
                    sim_rounds=args.sim_rounds,
                    max_children=args.max_children,
                    iterations=args.iterations,
                    max_workers=args.max_workers
            )
            return
        
        finally:
            try:
                ray.shutdown()
            except Exception:
                pass    

    # Run baseline without MCTS
    if args.run_base:
        from .pipline.pipeline_base import run_base

        attacker = build_attacker(args)
        target = build_target(args)
        evaluator = build_evaluator(args)
        
        run_base(
            attacker=attacker,
            target=target,
            evaluator=evaluator,
            max_refused=args.max_refused,
            max_rounds=args.max_rounds,
            iterations=args.iterations
        )
        return
    
    # Run mcts
    from .pipline.pipeline import run
    attacker = build_attacker(args)
    target = build_target(args)
    evaluator = build_evaluator(args)
    run(
        attacker=attacker,
        target=target,
        evaluator=evaluator,
        max_refused=args.max_refused,
        max_rounds=args.max_rounds,
        sim_rounds=args.sim_rounds,
        max_children=args.max_children,
        iterations=args.iterations
    )

if __name__ == "__main__":
    main()
