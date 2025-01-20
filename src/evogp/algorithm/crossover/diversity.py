from typing import Optional
import torch

from ...tree import Forest, MAX_STACK, randint
from ..selection import BaseSelector
from .base import BaseCrossover


class DiversityCrossover(BaseCrossover):

    def __init__(
        self,
        crossover_rate: int = 0.9,
        recipient_selector: Optional[BaseSelector] = None,
        donor_selector: Optional[BaseSelector] = None,
    ):
        self.crossover_rate = crossover_rate
        self.recipient_selector = recipient_selector
        self.donor_selector = donor_selector

    def __call__(
        self,
        forest: Forest,
        fitness: torch.Tensor,
        survivor_indices: torch.Tensor,
        target_cnt: torch.Tensor,
    ):
        crossover_cnt = int(target_cnt * self.crossover_rate)
        # choose recipient and donor indices
        if self.recipient_selector is not None:
            recipient_indices = self.recipient_selector(fitness, crossover_cnt)
        else:
            random_indices = torch.randint(
                low=0,
                high=survivor_indices.size(0),
                size=(crossover_cnt,),
                dtype=torch.int32,
                device="cuda",
                requires_grad=False,
            )
            recipient_indices = survivor_indices[random_indices]

        if self.donor_selector is not None:
            donor_indices = self.donor_selector(fitness, crossover_cnt)
        else:
            random_indices = torch.randint(
                low=0,
                high=survivor_indices.size(0),
                size=(crossover_cnt,),
                dtype=torch.int32,
                device="cuda",
                requires_grad=False,
            )
            donor_indices = survivor_indices[random_indices]

        # choose recipient and donor positions
        size_tensor = forest.batch_subtree_size
        recipient_pos = randint(
            size=(crossover_cnt,),
            low=0,
            high=size_tensor[recipient_indices, 0],
            dtype=torch.int32,
        )
        donor_pos = randint(
            size=(crossover_cnt,),
            low=0,
            high=size_tensor[donor_indices, 0],
            dtype=torch.int32,
        )

        # crossover the trees
        crossovered_forest = forest.crossover(
            recipient_indices,
            donor_indices,
            recipient_pos,
            donor_pos,
        )
        random_indices = torch.randint(
            low=0,
            high=survivor_indices.size(0),
            size=(target_cnt - crossover_cnt,),
            dtype=torch.int32,
            device="cuda",
            requires_grad=False,
        )
        static_forest = forest[random_indices]

        return crossovered_forest + static_forest
