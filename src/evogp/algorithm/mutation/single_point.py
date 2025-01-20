from typing import Optional, Tuple
import torch
from torch import Tensor

from .base import BaseMutation
from ...tree import Forest, MAX_STACK, NType, randint


class SinglePointMutation(BaseMutation):

    def __init__(
        self,
        mutation_rate: float,
        generate_configs: dict,
    ):
        self.mutation_rate = mutation_rate
        self.generate_configs = generate_configs

    def __call__(self, forest: Forest):
        # determine which trees need to mutate
        mutate_indices = torch.rand(forest.pop_size) < self.mutation_rate

        if mutate_indices.sum() == 0:  # no mutation
            return forest

        forest_to_mutate = forest[mutate_indices]

        # generate mutation positions
        num_mutate = forest_to_mutate.pop_size
        mutate_positions = randint(
            size=(num_mutate,),
            low=0,
            high=forest_to_mutate.batch_subtree_size[:, 0],
            dtype=torch.int32,
        )

        # random generate constant
        random_idx = torch.randint(
            low=0,
            high=self.generate_configs["const_samples"].shape[0],
            size=(num_mutate,),
            device="cuda",
        )
        random_const = self.generate_configs["const_samples"][random_idx]

        # random generate other
        mutated_node_type = forest_to_mutate.batch_node_type[
            torch.arange(num_mutate), mutate_positions
        ]
        mapping_range = torch.tensor(
            [
                [0, forest.input_len],  # VAR
                [0, 1],  # CONST
                [12, 24],  # UFUNC
                [1, 12],  # BFUNC
                [0, 1],  # TFUNC
            ],
            device="cuda",
        )
        random_other = randint(
            size=(num_mutate,),
            low=mapping_range[mutated_node_type.to(torch.int32)][:, 0],
            high=mapping_range[mutated_node_type.to(torch.int32)][:, 1],
        )

        # mutate the trees
        forest.batch_node_value[mutate_indices, mutate_positions] = torch.where(
            mutated_node_type == NType.CONST, random_const, random_other
        )
        return forest
