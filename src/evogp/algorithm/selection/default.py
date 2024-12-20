from typing import Optional
import torch

from ...tree import Forest
from .base import BaseSelection


class DefaultSelection(BaseSelection):

    def __init__(
        self,
        survival_rate: float = 0.3,
        elite_cnt: Optional[int] = None,
        elite_rate: Optional[float] = None,
    ):
        super().__init__()
        assert 0 <= survival_rate <= 1, "survival_rate should be in [0, 1]"
        assert (
            elite_cnt is None or elite_rate is None
        ), "elite_cnt and elite_rate should not be set at the same time"
        self.survival_rate = survival_rate
        self.elite_cnt = elite_cnt
        self.elite_rate = elite_rate

    def __call__(self, forest: Forest, fitness: torch.Tensor):
        survive_cnt = int(forest.pop_size * self.survival_rate)

        elite_cnt = 0
        if self.elite_cnt is not None:
            elite_cnt = self.elite_cnt
        elif self.elite_rate is not None:
            elite_cnt = int(forest.pop_size * self.elite_rate)

        sorted_fitness, sorted_indices = torch.sort(fitness, descending=True)
        survice_indices = sorted_indices[:survive_cnt]
        elite_indices = sorted_indices[:elite_cnt]

        return forest[elite_indices], forest[survice_indices]
