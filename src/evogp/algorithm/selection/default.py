from typing import Optional
import torch

from ...tree import Forest
from .base import BaseSelection


class DefaultSelection(BaseSelection):
    """
    DefaultSelection is the default selection strategy for the TGP algorithm.

    This strategy first preserves a certain proportion of elite individuals (those ranked at the top of the population by fitness). Then, individuals ranked at the bottom of the population are eliminated based on the survival rate. The strategy allows flexibility in defining the number or proportion of elite individuals to retain.
    """

    def __init__(
        self,
        survival_rate: float = 0.3,
        elite_cnt: Optional[int] = None,
        elite_rate: Optional[float] = None,
    ):
        """
        Args:
            survival_rate (float): The survival rate, determining the proportion of individuals that survive (between 0 and 1).
            elite_cnt (Optional[int]): The exact number of elite individuals to retain (if not None).
            elite_rate (Optional[float]): The proportion of elite individuals to retain (if not None).
        """
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

        return elite_indices, survice_indices
