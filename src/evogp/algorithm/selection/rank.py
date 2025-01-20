from typing import Optional
import torch

from ...tree import Forest
from .base import BaseSelection


class RankSelection(BaseSelection):
    """
    RankSelection implements a selection strategy based on fitness rank and selection pressure.

    Individuals are sorted by fitness, and their selection probabilities are calculated using the following formula:
    
    $$ P(R_i) = \frac{1}{n} \left( 1 + sp \left(1 - \frac{2i}{n-1}\right) \right) \quad \text{for } 0 \leq i \leq n-1, \quad 0 \leq sp \leq 1 $$

    where:
        - `n` is the population size,
        - `i` is the individual's rank,
        - `sp` represents the selection pressure (higher values correspond to greater pressure).
    
    Individuals are then selected with replacement based on these probabilities.
    """

    def __init__(
        self,
        selection_pressure: float = 0.5,
        survivor_rate: float = 0.5,
        elite_rate: float = 0,
        survivor_cnt: Optional[int] = None,
        elite_cnt: Optional[int] = None,
    ):
        """
        Args:
            selection_pressure (float): The selection pressure, a value between 0 and 1. 
                - 0 means no selection pressure.
                - 1 means high selection pressure.
            survivor_rate (float): The proportion of individuals that survive based on their rank and selection probability.
            elite_rate (float): The proportion of elite individuals to retain based on fitness.
            survivor_cnt (Optional[int]): The exact number of survivors to retain (if not None).
            elite_cnt (Optional[int]): The exact number of elite individuals to retain (if not None).
        """
        super().__init__()
        assert 0 <= selection_pressure <= 1, "selection_pressure should be in [0, 1]"
        assert 0 <= survivor_rate <= 1, "survivor_rate should be in [0, 1]"
        assert 0 <= elite_rate <= 1, "elite_rate should be in [0, 1]"
        self.sp = selection_pressure
        self.survivor_rate = survivor_rate
        self.survivor_cnt = survivor_cnt
        self.elite_rate = elite_rate
        self.elite_cnt = elite_cnt

    def __call__(self, forest: Forest, fitness: torch.Tensor):
        # preprocess
        if self.survivor_cnt is not None:
            survivor_cnt = self.survivor_cnt
        else:
            survivor_cnt = int(forest.pop_size * self.survivor_rate)

        if self.elite_cnt is not None:
            elite_cnt = self.elite_cnt
        else:
            elite_cnt = int(forest.pop_size * self.elite_rate)

        # survivor selection
        sorted_fitness, sorted_indices = torch.sort(fitness, descending=True)
        rank = sorted_indices.cuda()
        n = forest.pop_size
        random_indices = torch.multinomial(
            (1 / n) * (1 + self.sp * (1 - 2 * rank / (n - 1))),
            survivor_cnt,
            replacement=True,
        ).to(torch.int32)
        survivor_indices = sorted_indices[random_indices]

        # elite selection
        elite_indices = sorted_indices[:elite_cnt]

        return elite_indices, survivor_indices
