from typing import Optional
import torch

from ...tree import Forest, MAX_STACK
from ..selection import BaseSelector
from .base import BaseCrossover


class DiversityCrossover(BaseCrossover):

    def __init__(
        self,
        recipient_selector: Optional[BaseSelector] = None,
        donor_selector: Optional[BaseSelector] = None,
    ):
        self.recipient_selector = recipient_selector
        self.donor_selector = donor_selector

    def __call__(self, forest: Forest, target_cnt: int, args_check: bool = True):
        forest_size = forest.pop_size

        # choose left and right indices
        if self.recipient_selector is not None:
            _, recipient_indices = self.recipient_selector(forest, torch.tensor([]))
        else:
            recipient_indices = torch.randint(
                low=0,
                high=forest_size,
                size=(target_cnt,),
                dtype=torch.int32,
                device="cuda",
                requires_grad=False,
            )

        if self.donor_selector is not None:
            donor_indices = self.donor_selector(forest, torch.tensor([]))
        else:
            donor_indices = torch.randint(
                low=0,
                high=forest_size,
                size=(target_cnt,),
                dtype=torch.int32,
                device="cuda",
                requires_grad=False,
            )

        # chhoose left and right positions
        tree_sizes = forest.batch_subtree_size[:, 0]
        recipient_pos_unlimited, donor_pos_unlimited = torch.randint(
            low=0,
            high=MAX_STACK,
            size=(2, target_cnt),
            dtype=torch.int32,
            device="cuda",
            requires_grad=False,
        )
        recipient_pos = recipient_pos_unlimited % tree_sizes[recipient_indices]
        donor_pos = donor_pos_unlimited % tree_sizes[donor_indices]

        return forest.crossover(
            recipient_indices,
            donor_indices,
            recipient_pos,
            donor_pos,
            args_check=args_check,
        )
