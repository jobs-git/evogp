from typing import Optional, Tuple
import torch
from torch import Tensor

from .base import BaseMutation
from ...tree import Forest, MAX_STACK


class DefaultMutation(BaseMutation):

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

        # mutate the trees
        # generate sub trees
        sub_forest = Forest.random_generate(
            pop_size=forest_to_mutate.pop_size,
            gp_len=forest_to_mutate.gp_len,
            input_len=forest_to_mutate.input_len,
            output_len=forest_to_mutate.output_len,
            **self.generate_configs,
            args_check=args_check,
        )
        # generate mutate positions
        mutate_positions_unlimited = torch.randint(
            low=0,
            high=MAX_STACK,
            size=(forest_to_mutate.pop_size,),
            dtype=torch.int32,
            device="cuda",
            requires_grad=False,
        )
        mutate_positions = (
            mutate_positions_unlimited % forest_to_mutate.batch_subtree_size[:, 0]
        )

        forest[mutate_indices] = forest_to_mutate.mutate(mutate_positions, sub_forest, args_check=args_check)

        return forest
