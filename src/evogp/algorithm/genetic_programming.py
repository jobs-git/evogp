import torch

from . import BaseMutation, BaseCrossover, BaseSelection
from evogp.tree import Forest


class GeneticProgramming:

    def __init__(
        self,
        crossover: BaseCrossover,
        mutation: BaseMutation,
        selection: BaseSelection,
    ):
        self.forest = None
        self.pop_size = -1
        self.crossover = crossover
        self.mutation = mutation
        self.selection = selection

    def initialize(self, forest: Forest):
        self.forest = forest
        self.pop_size = forest.pop_size

    def step(self, fitness: torch.Tensor, args_check: bool = True):
        assert self.forest is not None, "forest is not initialized"
        assert fitness.shape == (
            self.forest.pop_size,
        ), "fitness shape should be ({self.forest.pop_size}, ), but got {fitness.shape}"

        elite_indices, next_indices = self.selection(self.forest, fitness)
        next_forest = self.crossover(
            forest=self.forest,
            survivor_indices=next_indices,
            target_cnt=self.pop_size - elite_indices.shape[0],
            fitness=fitness,
            args_check=args_check,
        )
        next_forest = self.mutation(next_forest, args_check=args_check)
        self.forest = self.forest[elite_indices] + next_forest

        return self.forest
