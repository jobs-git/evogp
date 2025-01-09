from typing import Optional, Tuple
import torch
from torch import Tensor

from .base import BaseMutation
from ...tree import Forest, MAX_STACK, randint, NType
from .mutation_utils import vmap_subtree


class DeleteMutation(BaseMutation):

    def __init__(
        self,
        mutation_rate: float,
        generate_configs: dict,
    ):
        self.mutation_rate = mutation_rate
        self.generate_configs = generate_configs

    def __call__(self, forest: Forest, args_check: bool = True):
        # determine which trees need to mutate
        mutate_indices = (
            torch.rand(forest.pop_size, device="cuda") < self.mutation_rate
        ) & (forest.batch_subtree_size[:, 0] > 1)

        if mutate_indices.sum() == 0:  # no mutation
            return forest

        forest_to_mutate = forest[mutate_indices]

        def choose_nonleaf_pos(size_tensor: Tensor):
            random = torch.rand(size_tensor.shape, device="cuda")
            # mask out the exceeding parts of the individual
            arange_tensor = torch.arange(size_tensor.shape[1], device="cuda")
            mask = arange_tensor < size_tensor[:, 0].unsqueeze(1)
            random = random * mask
            # mask out the leaf nodes
            random = torch.where(size_tensor == 1, 0, random)
            return torch.argmax(random, 1)

        # choose non-leaf position
        size_tensor = forest_to_mutate.batch_subtree_size
        mutate_positions = choose_nonleaf_pos(size_tensor)

        # choose next position
        num_mutate = forest_to_mutate.pop_size
        child_nums = (
            forest_to_mutate.batch_node_type[torch.arange(num_mutate), mutate_positions]
            - NType.UFUNC
            + 1
        )
        nth_childs = randint(
            size=(num_mutate,),
            low=1,
            high=child_nums,
            dtype=torch.int64,
        )

        pos1 = mutate_positions + 1
        pos2 = pos1 + size_tensor[torch.arange(num_mutate), pos1]
        pos3 = pos2 + size_tensor[torch.arange(num_mutate), pos2]
        next_positions = torch.where(nth_childs == 2, pos2, pos1)
        next_positions = torch.where(nth_childs == 3, pos3, next_positions)

        # delete the middle part
        subtrees = vmap_subtree(forest_to_mutate, next_positions)
        forest[mutate_indices] = forest_to_mutate.mutate(
            mutate_positions.to(torch.int32), subtrees, args_check=args_check
        )

        return forest
