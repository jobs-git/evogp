from typing import Optional, Tuple
import torch
from torch import Tensor

from .base import BaseMutation
from ...tree import Forest, MAX_STACK, randint
from .mutation_utils import vmap_subtree


class InsertMutation(BaseMutation):

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

        forest_to_mutate = forest[mutate_indices]

        # extract subtrees
        mutate_positions = randint(
            size=(forest_to_mutate.pop_size,),
            low=0,
            high=forest_to_mutate.batch_subtree_size[:, 0],
            dtype=torch.int64,
        )
        subtrees = vmap_subtree(forest_to_mutate, mutate_positions)

        # generate newtrees
        newtrees = Forest.random_generate(
            pop_size=forest_to_mutate.pop_size,
            gp_len=forest_to_mutate.gp_len,
            input_len=forest_to_mutate.input_len,
            output_len=forest_to_mutate.output_len,
            **self.generate_configs,
            args_check=args_check,
        )
        newtrees_positions = randint(
            size=(newtrees.pop_size,),
            low=1,
            high=newtrees.batch_subtree_size[:, 0],
            dtype=torch.int32,
        )

        # insert the subtrees to newtrees
        newtrees = newtrees.mutate(newtrees_positions, subtrees, args_check=args_check)

        # insert the newtrees to forest
        forest[mutate_indices] = forest_to_mutate.mutate(
            mutate_positions.to(torch.int32), newtrees, args_check=args_check
        )

        return forest
