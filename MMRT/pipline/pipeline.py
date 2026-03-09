from multiprocessing import Pool
from tqdm import tqdm
import os

from ..data.data_loader import load_dataset
from ..config import Config
from ..mcts.mcts import MCTS
from ..utils.utils import  save_task_json, is_task_completed, is_task_score5, reset_generated_images

from ..utils.logger import get_logger
from ..utils.timer import Timer


logger = get_logger(__name__) 
timer = Timer(logger)

def run(
    attacker, 
    target, 
    evaluator, 
    max_refused,
    max_rounds, 
    sim_rounds,
    max_children,
    iterations
):  
    dataset = load_dataset(Config.data_path)

    for item in tqdm(dataset, desc="Processing tasks", unit="task"):

        item_dict = item.to_dict()

        task_info = {
            "task_id": item_dict["id"],
            "task": item_dict["malicious_intent"],
            "image_ids": item.get_image_ids(),
            "captions": item.get_captions(),
        }

        if is_task_score5(task_info['task_id']):
            logger.info(f"Skipping task {task_info['task_id']} ...")
            continue

        if is_task_completed(task_info['task_id'], score_threshold=Config.skip_score_above):
            logger.info(f"Skipping task {task_info['task_id']} ...")
            continue
        
        logger.info("=" * 80)
        logger.info(f"Running MCTS for task {task_info['task_id']}")

        reset_generated_images(task_info['task_id'])

        mcts = MCTS(
            attacker=attacker,
            target=target,
            evaluator=evaluator,
            task_info=task_info,
            max_refused=max_refused,
            max_rounds=max_rounds,
            sim_rounds = sim_rounds,       
            max_children = max_children,       
            iterations = iterations 
        )

        best_path = timer.measure(f"Task {task_info['task_id']}", mcts.search)
        
        output = item.to_dict()
        output["conversations"] = best_path
        
        final_score = 0
        if best_path:
            final_score = best_path[-1]["evaluator_output"]["score"]

        if best_path and final_score == 5:
            logger.info("MCTS DONE! Found score=5 path.")
            save_task_json(output, f"{task_info['task_id']}.json")
        else:

            logger.info(f"MCTS FAILED. Found score={final_score} path.")
            save_task_json(output, f"{task_info['task_id']}_score{final_score}.json")


        mcts.visualize_path(best_path)

    logger.info("Average task time: %.2fs over %d tasks", timer.avg, len(timer.times))
