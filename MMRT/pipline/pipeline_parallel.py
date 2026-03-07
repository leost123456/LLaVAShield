from concurrent.futures import ThreadPoolExecutor, as_completed

from config import Config
from data.data_loader import load_dataset

from mcts.mcts import MCTS

from utils.utils import save_task_json, is_task_score5, is_task_completed, reset_generated_images
from utils.logger import get_logger
from utils.timer import Timer

logger = get_logger(__name__) 
timer = Timer(logger)

def _run_one_task(item, attacker, target, evaluator, max_refused, max_rounds, sim_rounds, max_children, iterations, logger, timer):
    item_dict = item.to_dict()
    task_info = {
        "task_id": item_dict["id"],
        "task": item_dict["malicious_intent"],
        "image_ids": item.get_image_ids(),
        "captions": item.get_captions(),
    }

    if is_task_score5(task_info['task_id']):
        logger.info(f"Skipping task {task_info['task_id']} ...")
        return task_info['task_id'], True
    
    if is_task_completed(task_info['task_id'], score_threshold=Config.skip_score_above):
        logger.info(f"Skipping task {task_info['task_id']} ...")
        return task_info['task_id'], True
    
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
        final_score = int(best_path[-1]["evaluator_output"]["score"])
        
    if best_path and final_score == 5:
        logger.info("MCTS DONE! Found score=5 path.")
        save_task_json(output, f"{task_info["task_id"]}.json")
        success_flag = True
    else:
        logger.info(f"MCTS FAILED. Found score={final_score} path.")
        save_task_json(output, f"{task_info["task_id"]}_score{final_score}.json")
        success_flag = False


    return task_info['task_id'], success_flag

def run_parallel(
    attacker_factory,  
    target_factory,
    evaluator_factory,
    max_refused,
    max_rounds,
    sim_rounds,
    max_children,
    iterations,
    max_workers     
):
    logger = get_logger(__name__)
    timer  = Timer(logger)

    dataset = load_dataset(Config.data_path)

    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for item in dataset:
            attacker = attacker_factory()
            target   = target_factory()
            evaluator= evaluator_factory()
            fut = ex.submit(
                _run_one_task,
                item, attacker, target, evaluator,
                max_refused, max_rounds, sim_rounds, max_children, iterations,
                logger, timer
            )
            futures.append(fut)

        done_cnt = 0
        for fut in as_completed(futures):
            tid, success_flag = fut.result()
            done_cnt += 1
            logger.info("Task %s finished. success_flag=%s (%d/%d)", tid, success_flag, done_cnt, len(futures))

    logger.info("Average task time: %.2fs over %d tasks", timer.avg, len(timer.times))
