from typing import Optional, Tuple
import torch
from torch import Tensor

from .base import BaseMutation
from ...tree import Forest, MAX_STACK, NType


class MultiConstMutation(BaseMutation):

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

        def choose_mutation_targets(size_tensor, type_tensor):
            tree_size = size_tensor[:, 0].reshape(-1, 1)
            random = torch.rand(tree_size.shape, device="cuda")
            return (
                (random < self.mutation_intensity)
                & (type_tensor == NType.CONST)
                & (torch.arange(size_tensor.shape[1], device="cuda") < tree_size)
            )

        # generate mutation indices and positions
        mutation_targets = choose_mutation_targets(
            forest_to_mutate.batch_subtree_size, forest_to_mutate.batch_node_type
        )
        num_targets = mutation_targets.sum()

        # random generate constant
        random_idx = torch.randint(
            low=0,
            high=self.generate_configs["const_samples"].shape[0],
            size=(num_targets,),
            device="cuda",
        )
        random_const = self.generate_configs["const_samples"][random_idx]

        # mutate the trees
        forest_to_mutate.batch_node_value[mutation_targets] = random_const
        forest[mutate_indices] = forest_to_mutate
        return forest
