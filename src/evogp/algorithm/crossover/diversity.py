from typing import Optional
import torch

from ...tree import Forest, MAX_STACK, randint
from ..selection import BaseSelector
from .base import BaseCrossover

class DiversityCrossover(BaseCrossover):
    """
    Implements a diversity-preserving crossover operation for genetic programming.

    This class is responsible for performing crossover operations between trees
    in a population of genetic programming trees. It allows for the selection of
    donor and recipient trees, as well as the positions within the trees where
    crossover should occur.

    Attributes:
        crossover_rate (int): The probability (between 0 and 1) that a crossover will occur.
        recipient_selector (Optional[BaseSelector]): A selection mechanism to choose
            recipient trees for crossover. If None, random selection is used.
        donor_selector (Optional[BaseSelector]): A selection mechanism to choose
            donor trees for crossover. If None, random selection is used.
    """

    def __init__(
        self,
        crossover_rate: int = 0.9,
        recipient_selector: Optional[BaseSelector] = None,
        donor_selector: Optional[BaseSelector] = None,
    ):
        """
        Initializes the DiversityCrossover instance.

        Args:
            crossover_rate (int): The proportion of target individuals to be generated
                via crossover. Default is 0.9.
            recipient_selector (Optional[BaseSelector]): Selector for choosing recipient
                trees. Default is None, which uses random selection.
            donor_selector (Optional[BaseSelector]): Selector for choosing donor trees.
                Default is None, which uses random selection.
        """
        self.crossover_rate = crossover_rate
        self.recipient_selector = recipient_selector
        self.donor_selector = donor_selector

    def __call__(
        self,
        forest: Forest,
        fitness: torch.Tensor,
        survivor_indices: torch.Tensor,
        target_cnt: torch.Tensor,
    ):
        """
        Executes the crossover operation on the provided forest.

        Args:
            forest (Forest): The population of trees represented as a Forest object.
            fitness (torch.Tensor): Fitness scores of the individuals in the population.
            survivor_indices (torch.Tensor): Indices of individuals selected to survive
                into the next generation.
            target_cnt (torch.Tensor): The target number of individuals for the next
                generation.

        Returns:
            Forest: A new forest containing individuals created via crossover and
            individuals that were statically copied.
        """
        # Calculate the number of crossovers to perform
        crossover_cnt = int(target_cnt * self.crossover_rate)

        # Choose recipient indices for crossover
        if self.recipient_selector is not None:
            recipient_indices = self.recipient_selector(fitness, crossover_cnt)
        else:
            random_indices = torch.randint(
                low=0,
                high=survivor_indices.size(0),
                size=(crossover_cnt,),
                dtype=torch.int32,
                device="cuda",
                requires_grad=False,
            )
            recipient_indices = survivor_indices[random_indices]

        # Choose donor indices for crossover
        if self.donor_selector is not None:
            donor_indices = self.donor_selector(fitness, crossover_cnt)
        else:
            random_indices = torch.randint(
                low=0,
                high=survivor_indices.size(0),
                size=(crossover_cnt,),
                dtype=torch.int32,
                device="cuda",
                requires_grad=False,
            )
            donor_indices = survivor_indices[random_indices]

        # Select positions within the recipient and donor trees for crossover
        size_tensor = forest.batch_subtree_size
        recipient_pos = randint(
            size=(crossover_cnt,),
            low=0,
            high=size_tensor[recipient_indices, 0],
            dtype=torch.int32,
        )
        donor_pos = randint(
            size=(crossover_cnt,),
            low=0,
            high=size_tensor[donor_indices, 0],
            dtype=torch.int32,
        )

        # Perform the crossover operation to generate new trees
        crossovered_forest = forest.crossover(
            recipient_indices,
            donor_indices,
            recipient_pos,
            donor_pos,
        )

        # Select remaining individuals to copy directly to the new generation
        random_indices = torch.randint(
            low=0,
            high=survivor_indices.size(0),
            size=(target_cnt - crossover_cnt,),
            dtype=torch.int32,
            device="cuda",
            requires_grad=False,
        )
        static_forest = forest[random_indices]

        # Combine crossovered trees and statically copied trees
        return crossovered_forest + static_forest