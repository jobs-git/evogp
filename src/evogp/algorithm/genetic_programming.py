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

    def step(self, fitness: torch.Tensor):
        assert self.forest is not None, "forest is not initialized"
        assert fitness.shape == (
            self.forest.pop_size,
        ), "fitness shape should be ({self.forest.pop_size}, ), but got {fitness.shape}"

        elite_forest, next_forest = self.selection(self.forest, fitness)
        next_forest = self.crossover(next_forest, self.pop_size - len(elite_forest))
        next_forest = self.mutation(next_forest)
        self.forest = elite_forest + next_forest
        
        return self.forest
