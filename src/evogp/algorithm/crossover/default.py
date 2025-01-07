import torch

from ...tree import Forest, MAX_STACK
from .base import BaseCrossover


class DefaultCrossover(BaseCrossover):

    def __init__(self):
        super().__init__()

    def __call__(
        self,
        forest: Forest,
        survivor_indices: torch.Tensor,
        target_cnt: int,
        fitness: torch.Tensor,
        args_check: bool = True,
    ):
        survivor_forest = forest[survivor_indices]
        # choose left and right indices
        left_indices, right_indices = torch.randint(
            low=0,
            high=len(survivor_forest),
            size=(2, target_cnt),
            dtype=torch.int32,
            device="cuda",
            requires_grad=False,
        )

        # choose left and right positions
        tree_sizes = survivor_forest.batch_subtree_size[:, 0]
        left_pos_unlimited, right_pos_unlimited = torch.randint(
            low=0,
            high=MAX_STACK,
            size=(2, target_cnt),
            dtype=torch.int32,
            device="cuda",
            requires_grad=False,
        )
        left_pos = left_pos_unlimited % tree_sizes[left_indices]
        right_pos = right_pos_unlimited % tree_sizes[right_indices]

        # print(f"{tree_sizes=}\n{left_indices=}\n{right_indices=}\n{left_pos=}\n{right_pos=}")

        return survivor_forest.crossover(
            left_indices, right_indices, left_pos, right_pos, args_check=args_check
        )