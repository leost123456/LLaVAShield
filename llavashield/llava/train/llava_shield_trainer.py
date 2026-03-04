import re
import json
from transformers.trainer_utils import speed_metrics, EvalLoopOutput
from .llava_trainer import LLaVATrainer
#from utils.evaluate import compute_acc_precision_recall_f1

class LLaVAShieldTrainer(LLaVATrainer):
    def __init__(self, data_collator_eval=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_collator_eval = data_collator_eval or self.data_collator
        
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        self._memory_tracker.start()

        tmp = self.data_collator
        self.data_collator = self.data_collator_eval
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        self.data_collator = tmp

        import time, math
        start_time = time.time()
        output = self.evaluation_loop(
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=False,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix, start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )
        self.log(output.metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)
        self._memory_tracker.stop_and_update_metrics(output.metrics)
        return output.metrics

    