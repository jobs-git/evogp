from typing import Optional, Tuple
import torch
from torch import Tensor

from .base import BaseMutation
from ...tree import Forest, MAX_STACK, GenerateDiscriptor


class DefaultMutation(BaseMutation):

    def __init__(
        self,
        mutation_rate: float,
        descriptor: GenerateDiscriptor,
    ):
        self.mutation_rate = mutation_rate
        self.descriptor = descriptor

    def __call__(self, forest: Forest):
        # determine which trees need to mutate
        mutate_indices = torch.rand(forest.pop_size) < self.mutation_rate

        if mutate_indices.sum() == 0:  # no mutation
            return forest

        forest_to_mutate = forest[mutate_indices]

        # mutate the trees
        # generate sub trees
        sub_forest = Forest.random_generate(
            pop_size=forest_to_mutate.pop_size,
            descriptor=self.descriptor
        )
        # generate mutation positions
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

        forest[mutate_indices] = forest_to_mutate.mutate(mutate_positions, sub_forest)

        return forest
