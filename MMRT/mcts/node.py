import math

from ..utils.logger import get_logger

logger = get_logger(__name__)  

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state              # Current state: history, round_number, etc.
        self.parent = parent
        self.action = action            # next_question
        self.children = []
        self.visits = 0
        self.reward = 0.0
        self.total_reward = 0.0
        self.best_child_score = 0

    def is_terminal(self, max_rounds):
        return self.state["round_number"] >= max_rounds

    def is_fully_expanded(self, max_children):
        return len(self.children) >= max_children
    
    def _ucb_with_score(self, c=math.sqrt(2), score_weight=5):
        if self.visits == 0:
            return float('inf')
        
        parent_visits = self.parent.visits if self.parent else 1
        return self.total_reward / self.visits \
            + c * math.sqrt(math.log(parent_visits) / self.visits) \
            + score_weight * (self.state["score"] - 1.0) / 4.0

    def _puct(self, c=1.5, eps=0.10, beta=0.7):
        Q = (self.total_reward / self.visits) if self.visits > 0 else 0.0
        bs = (self.best_child_score - 1.0) / 4.0
        exploit = (1 - beta) * Q + beta * bs

        s = (self.state["score"] - 1.0) / 4.0
        P = eps + (1-eps) * s
        N = self.parent.visits if self.parent else 1
        
        return exploit + c * P * math.sqrt(N) / (1 + self.visits)
        
    def best_child(self):
        
        return max(
            self.children,
            key=lambda child: child._puct()
        )
    
    def print_subtree(self, indent=0):
        prefix = " " * indent
        score = self.state["score"]  if self.state["score"]  else 0
        uct =  self._puct() if self.parent else 0
        action_preview = (self.action) if self.action else "ROOT"
        logger.info(f"{prefix}- [R{self.state['round_number']}] visits={self.visits}, reward={self.reward:.2f}, uct={uct:.2f}, score={score}, q='{action_preview}'")
        for child in self.children:
            child.print_subtree(indent + 2)
