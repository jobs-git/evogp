from typing import Optional, Tuple
import torch
from torch import Tensor

from .base import BaseMutation
from ...tree import Forest, MAX_STACK, randint, NType


class MultiPointMutation(BaseMutation):

    def __init__(
        self,
        mutation_rate: float,
        generate_configs: dict,
        mutation_intensity: float = 0.3,
    ):
        self.mutation_rate = mutation_rate
        self.generate_configs = generate_configs
        self.mutation_intensity = mutation_intensity

    def __call__(self, forest: Forest):
        # determine which trees need to mutate
        mutate_indices = torch.rand(forest.pop_size) < self.mutation_rate

        if mutate_indices.sum() == 0:  # no mutation
            return forest

        forest_to_mutate = forest[mutate_indices]

        def choose_mutation_targets(size_tensor):
            tree_size = size_tensor[:, 0].reshape(-1, 1)
            random = torch.rand(tree_size.shape, device="cuda")
            return (random < self.mutation_intensity) & (
                torch.arange(size_tensor.shape[1], device="cuda") < tree_size
            )

        # generate mutation indices and positions
        mutation_targets = choose_mutation_targets(forest_to_mutate.batch_subtree_size)
        num_targets = mutation_targets.sum()

        # random generate constant
        random_idx = torch.randint(
            low=0,
            high=self.generate_configs["const_samples"].shape[0],
            size=(num_targets,),
            device="cuda",
        )
        random_const = self.generate_configs["const_samples"][random_idx]

        # random generate other
        mutated_node_type = forest_to_mutate.batch_node_type[mutation_targets]
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
            size=(num_targets,),
            low=mapping_range[mutated_node_type.to(torch.int32)][:, 0],
            high=mapping_range[mutated_node_type.to(torch.int32)][:, 1],
        )

        # mutate the trees
        forest_to_mutate.batch_node_value[mutation_targets] = torch.where(
            mutated_node_type == NType.CONST, random_const, random_other
        )
        forest[mutate_indices] = forest_to_mutate
        return forest
