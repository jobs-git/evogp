from typing import Optional, Tuple
import torch
from torch import Tensor

from .base import BaseMutation
from ...tree import Forest, MAX_STACK, randint
from .mutation_utils import vmap_subtree


class HoistMutation(BaseMutation):

    def __init__(
        self,
        mutation_rate: float,
        generate_configs: dict,
    ):
        self.mutation_rate = mutation_rate
        self.generate_configs = generate_configs

    def __call__(self, forest: Forest, args_check: bool = True):
        # determine which trees need to mutate
        mutate_indices = torch.rand(forest.pop_size) < self.mutation_rate

        if mutate_indices.sum() == 0:  # no mutation
            return forest
        else:
            forest_to_mutate = forest[mutate_indices]

        # generate mutation positions
        num_mutate = forest_to_mutate.pop_size
        mutate_positions = randint(
            size=(num_mutate,),
            low=0,
            high=forest_to_mutate.batch_subtree_size[:, 0],
            dtype=torch.int32,
        )

        # generate the next mutate positions
        subtree_positions = randint(
            size=(num_mutate,),
            low=0,
            high=forest_to_mutate.batch_subtree_size[
                torch.arange(num_mutate), mutate_positions
            ],
            dtype=torch.int64,
        )

        # put the next mutate positions into the previous mutate positions
        subtrees = vmap_subtree(forest_to_mutate, subtree_positions)
        forest[mutate_indices] = forest_to_mutate.mutate(
            mutate_positions, subtrees, args_check=args_check
        )

        return forest
