from typing import Optional, Tuple
import torch
from torch import Tensor

from .base import BaseMutation
from ...tree import Forest, MAX_STACK


class DefaultMutation(BaseMutation):

    def __init__(
        self,
        mutation_rate: float,
        const_prob: float,
        out_prob: Optional[float] = None,
        depth2leaf_probs: Optional[Tensor] = None,
        roulette_funcs: Optional[Tensor] = None,
        const_samples: Optional[Tensor] = None,
        func_prob: Optional[dict] = None,
        max_layer_cnt: Optional[int] = None,
        layer_leaf_prob: Optional[float] = None,
        const_range: Optional[Tuple[float, float]] = None,
        sample_cnt: Optional[int] = None,
    ):
        self.mutation_rate = mutation_rate
        self.const_prob = const_prob
        self.out_prob = out_prob
        self.depth2leaf_probs = depth2leaf_probs
        self.roulette_funcs = roulette_funcs
        self.const_samples = const_samples
        self.func_prob = func_prob
        self.max_layer_cnt = max_layer_cnt
        self.layer_leaf_prob = layer_leaf_prob
        self.const_range = const_range
        self.sample_cnt = sample_cnt

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
            gp_len=forest_to_mutate.gp_len,
            input_len=forest_to_mutate.input_len,
            output_len=forest_to_mutate.output_len,
            const_prob=self.const_prob,
            out_prob=self.out_prob,
            depth2leaf_probs=self.depth2leaf_probs,
            roulette_funcs=self.roulette_funcs,
            const_samples=self.const_samples,
            func_prob=self.func_prob,
            max_layer_cnt=self.max_layer_cnt,
            layer_leaf_prob=self.layer_leaf_prob,
            const_range=self.const_range,
            sample_cnt=self.sample_cnt,
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

        forest[mutate_indices] = forest_to_mutate.mutate(mutate_positions, sub_forest)

        return forest
