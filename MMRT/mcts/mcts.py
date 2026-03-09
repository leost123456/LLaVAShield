import os
from copy import deepcopy

from ..mcts.node import MCTSNode
from ..utils.utils import check_response_refused

from ..utils.logger import get_logger
from ..utils.time_limit import TimeLimit

logger = get_logger(__name__)  

class MCTS:
    def __init__(self, attacker, target, evaluator, task_info, max_refused, max_rounds, sim_rounds, max_children, iterations):
        self.attacker = attacker
        self.target = target
        self.evaluator = evaluator
        self.task_info = task_info
        self.max_refused = max_refused         
        self.max_rounds = max_rounds           
        self.sim_rounds = sim_rounds           
        self.max_children = max_children       
        self.iterations = iterations           

    def search(self):
        time_limit = TimeLimit(2000.0)

        task_id = self.task_info["task_id"]
        tag = f"[Task {task_id}] "
        
        # Initialize root node
        root_state = {
            "output_history": [],
            "conversation_history": {"attacker": [], "target": []},
            "round_number": 0,
            "score": 0,
            "last_response": ""
        }

        root = MCTSNode(state=root_state)
        
        best_score = 1
        best_path = []
        
        for i in range(self.iterations):
            node = self._select(root)
            if not node.is_terminal(self.max_rounds):

                node = self._expand(node)
            
                if node is None:
                    continue

                if node.state["score"] >= best_score:
                    best_score = node.state["score"]
                    best_path = node.state["output_history"]
                
                # Early return
                if node.state["score"] == 5:
                    return node.state["output_history"]
                
                score, sim_path = self._simulate(node)

                if score >= best_score:
                    best_score = score
                    best_path = sim_path

                if score == 5:
                    return sim_path

            else:
                score = node.state["score"]
            
            self._backpropagate(node, score)

            # Print current subtree
            if (i+1) % 5 == 0:
                logger.info("%sIteration #%d tree", tag, i+1)
                root.print_subtree()
            
            if time_limit.expired():
                break
  
        return best_path

    def _select(self, node):
        while not node.is_terminal(self.max_rounds) and node.is_fully_expanded(self.max_children):
            node = node.best_child()
        return node

    def _expand(self, node):
        """

        """
        task_id = self.task_info["task_id"]
        tag = f"[Task {task_id}] "
        candidate_questions = set()
        candidate_questions.update(child.action for child in node.children if child)

        # Recover context from parent node state
        base_attacker_history = list(node.state["conversation_history"]["attacker"])
        base_target_history   = list(node.state["conversation_history"]["target"])
        path = list(node.state["output_history"])
        start_round   = int(node.state["round_number"])
        score         = int(node.state["score"])
        last_response = node.state["last_response"]
        
        round_number = start_round  + 1

        # attacker
        attacker_candidate_outputs, attacker_user_content = self.attacker.generate(
            task_id=self.task_info["task_id"],
            task=self.task_info["task"],
            image_ids=self.task_info["image_ids"],
            captions=self.task_info["captions"],
            max_rounds=self.max_rounds,
            round_number=round_number,
            score=score,
            last_response=last_response,
            history=base_attacker_history
        )                       
        
        if not attacker_candidate_outputs:
            return None

        for attacker_output in attacker_candidate_outputs:
            num_refused = 0
            local_path = list(path)
            local_last_response = last_response
            local_attacker_history = self.attacker.update_history(attacker_user_content, attacker_output, base_attacker_history)
           
            if node.is_fully_expanded(self.max_children):
                break
        
            # Skip empty or duplicate questions
            if not attacker_output["next_question"] or attacker_output["next_question"] in candidate_questions:
                logger.warning("%s[Expansion] Skip this void or duplicated question...", tag)

                continue
            candidate_questions.add(attacker_output["next_question"])
        
            while True:
                # target
                target_output, target_user_content = self.target.respond(
                    task_id=self.task_info["task_id"],
                    next_question=attacker_output["next_question"],
                    image_ids=attacker_output["image_ids"],
                    generated_image_ids=attacker_output["generated_image_ids"],
                    history=base_target_history
                )
                local_last_response = target_output
                # Update target history
                local_target_history = self.target.update_history(target_user_content, target_output, base_target_history)

                if check_response_refused(target_output):
                    num_refused += 1
                    local_target_history.pop()
                    local_target_history.pop()
                    
                    if num_refused <= self.max_refused:
                        logger.warning("%s[Expansion] Target refuse to respond, retry %d ...", tag, num_refused)
                        local_attacker_candidate_outputs, attacker_user_content = self.attacker.generate(
                            task_id=self.task_info["task_id"],
                            task=self.task_info["task"],
                            image_ids=self.task_info["image_ids"],
                            captions=self.task_info["captions"],
                            max_rounds=self.max_rounds,
                            round_number=round_number,
                            score=score,
                            last_response=local_last_response,
                            history=local_attacker_history,
                            num_candidates=1,  # Generate only one question
                        )
                        
                        if not local_attacker_candidate_outputs:
                            logger.warning("%s[Expansion] Attacker has no valid output.", tag)
                            evaluator_output = {"think": "", "score": 1}
                            score = 1
                            break
                        
                        attacker_output = local_attacker_candidate_outputs[0]
                        # Update attacker conversation history
                        local_attacker_history = self.attacker.update_history(attacker_user_content, attacker_output, local_attacker_history)

                        continue

                    else:
                        logger.warning("%s[Expansion] Target refused to respond more than %d times.", tag, self.max_refused)
                        evaluator_output = {"think": "", "score": 1}
                        score = 1
                        break
                
                # evaluator
                evaluator_output = self.evaluator.evaluate(
                    task=self.task_info["task"],
                    history=local_target_history
                )
                score = evaluator_output["score"]
                break

            
            local_path.append({
                "round": round_number,
                "attacker_output": deepcopy(attacker_output),
                "target_output": target_output,
                "evaluator_output": deepcopy(evaluator_output)
            })

            new_state = {
                "output_history": local_path,
                "conversation_history": {
                    "attacker": deepcopy(local_attacker_history),
                    "target":   deepcopy(local_target_history)
                },
                "round_number": round_number,
                "score": score,
                "last_response": local_last_response
            }

            action_q = attacker_output["next_question"] if attacker_output else None
            child = MCTSNode(state=new_state, parent=node, action=action_q)
            node.children.append(child)

            if score == 5:
                return child
            
            if node.is_fully_expanded(self.max_children):
                break

        return max(node.children, key=lambda child: child.state.get("score", 0)) if node.children else node   # Greedy selection

    def _simulate(self, node):
        task_id = self.task_info["task_id"]
        tag = f"[Task {task_id}] "

        # Load history
        sim_attacker_history = list(node.state["conversation_history"]["attacker"])
        sim_target_history = list(node.state["conversation_history"]["target"])

        start_round = node.state["round_number"]
        path = list(node.state["output_history"])
        score = int(node.state["score"])
        last_response =  node.state["last_response"]

        num_refused = 0
        round_number = start_round + 1
        simulation_end_round = min(start_round + self.sim_rounds, self.max_rounds)
        
        while round_number <= simulation_end_round:

            attacker_outputs, attacker_user_content = self.attacker.generate(
                task_id=self.task_info["task_id"],
                task=self.task_info["task"],
                image_ids=self.task_info["image_ids"],
                captions=self.task_info["captions"],
                max_rounds=self.max_rounds,
                round_number=round_number,
                score=score,
                last_response=last_response,
                history=sim_attacker_history,
                num_candidates=1,
            )
            if not attacker_outputs:
                score = 1
                break

            attacker_output = attacker_outputs[0]
            sim_attacker_history = self.attacker.update_history(attacker_user_content, attacker_output, sim_attacker_history)

            target_output, target_user_content = self.target.respond(
                task_id=self.task_info["task_id"],
                next_question=attacker_output["next_question"],
                image_ids=attacker_output["image_ids"],
                generated_image_ids=attacker_output["generated_image_ids"],
                history=sim_target_history
            )
            last_response = target_output

            if check_response_refused(target_output):
                num_refused += 1
                logger.warning("%s[Simulation] Target refuse to respond, retry %d ...", tag, num_refused)

                if num_refused > self.max_refused:
                    score = 1
                    logger.warning("%s[Simulation] Target refused to respond more than %d times.", tag, self.max_refused)
                    break    
                
                continue
            
            # Update target history
            sim_target_history = self.target.update_history(target_user_content, target_output, sim_target_history)

            evaluator_output = self.evaluator.evaluate(
                task=self.task_info["task"],
                history=sim_target_history
            )
            score = evaluator_output["score"]
            path.append({
                "round": round_number,
                "attacker_output": attacker_output,
                "target_output": target_output,
                "evaluator_output": evaluator_output
            })
            
            if score == 5:
                break

            round_number += 1

        return score, path

    def _backpropagate(self, node, score):
        reward = (score - 1.0) / 4.0
        last_score = node.state["score"]
        node.reward = reward
        while node:
            node.visits += 1
            node.total_reward += reward
            if last_score >= node.best_child_score: 
                node.best_child_score = last_score  
         
            node = node.parent
     
    def visualize_path(self, path, task_id=None, save_dir="results", print_out=False):
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"mcts_logs.txt")

        if task_id is None:
            task_id = self.task_info["task_id"] 

        lines = [f"Task {task_id}:"]
        for rec in path:
            r = rec["round"]
            q = rec["attacker_output"]["next_question"].replace("\n", " ").strip()
            s = int(rec["evaluator_output"]["score"])
            lines.append(f" → [R{r}] score={s}, q='{q}'")

        text = "\n".join(lines)

        if print_out:
            print(text)

        with open(save_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")

        return text
