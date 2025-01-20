from typing import Optional, Tuple
import torch
from torch import Tensor

from .base import BaseMutation
from ...tree import Forest, MAX_STACK, NType


class SingleConstMutation(BaseMutation):

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

        def choose_constant_pos(size_tensor: Tensor, type_tensor: Tensor):
            random = torch.rand(size_tensor.shape, device="cuda")
            # mask out the exceeding parts of the individual
            arange_tensor = torch.arange(size_tensor.shape[1], device="cuda")
            mask = arange_tensor < size_tensor[:, 0].unsqueeze(1)
            random = random * mask
            # mask out the non-constant nodes
            random = torch.where(type_tensor == NType.CONST, random, 0)
            return torch.argmax(random, 1)

        # choose the constant position to mutate
        num_mutate = forest_to_mutate.pop_size
        mutate_positions = choose_constant_pos(
            forest_to_mutate.batch_subtree_size,
            forest_to_mutate.batch_node_type,
        )

        # random generate constant
        random_idx = torch.randint(
            low=0,
            high=self.generate_configs["const_samples"].shape[0],
            size=(num_mutate,),
            device="cuda",
        )
        random_const = self.generate_configs["const_samples"][random_idx]

        # mutate the trees
        # protect operation: if the tree doesn't have constant, may be still choose the normal node
        mutated_node_type = forest_to_mutate.batch_node_type[
            torch.arange(num_mutate), mutate_positions
        ]
        original_node_value = forest_to_mutate.batch_node_value[
            torch.arange(num_mutate), mutate_positions
        ]
        forest.batch_node_value[mutate_indices, mutate_positions] = torch.where(
            mutated_node_type == NType.CONST, random_const, original_node_value
        )
        return forest
