from typing import Optional, Tuple
import torch
from torch import Tensor

from .base import BaseMutation
from ...tree import Forest, MAX_STACK, randint
from .mutation_utils import vmap_subtree


class InsertMutation(BaseMutation):
    """
    InsertMutation implements a mutation strategy where a random basic operator (subtree) is inserted 
    into a GP individual at a randomly selected position. This operation helps introduce new structure 
    into the individual by replacing part of the original tree with a newly generated subtree.
    """

    def __init__(
        self,
        mutation_rate: float,
        generate_configs: dict,
    ):
        """
        Args:
            mutation_rate (float): The probability of each individual undergoing mutation. Should be between 0 and 1.
            generate_configs (dict): Configuration dictionary for subtree generation (used in mutation).
        """
        self.mutation_rate = mutation_rate
        self.generate_configs = generate_configs

    def __call__(self, forest: Forest):
        """
        Perform the insert mutation by selecting a random subtree from the individual and inserting 
        it into a random position within the tree.

        This mutation introduces new structure into the individual by replacing part of the original tree 
        with a newly generated subtree.

        Args:
            forest (Forest): The current population of trees (Forest object).

        Returns:
            Forest: The updated population after mutation, where some individuals have undergone the insert operation.
        """
        # Determine which trees need to mutate based on the mutation rate
        mutate_indices = torch.rand(forest.pop_size) < self.mutation_rate

        # If no trees are selected for mutation, return the original forest
        if mutate_indices.sum() == 0:  
            return forest

        # Extract the subset of trees that need to mutate
        forest_to_mutate = forest[mutate_indices]

        # Generate random mutation positions for selected trees
        mutate_positions = randint(
            size=(forest_to_mutate.pop_size,),
            low=0,
            high=forest_to_mutate.batch_subtree_size[:, 0],
            dtype=torch.int64,
        )

        # Extract subtrees from the selected mutation positions
        subtrees = vmap_subtree(forest_to_mutate, mutate_positions)

        # Generate new trees (subtrees) to insert into the individual
        newtrees = Forest.random_generate(
            pop_size=forest_to_mutate.pop_size,
            gp_len=forest_to_mutate.gp_len,
            input_len=forest_to_mutate.input_len,
            output_len=forest_to_mutate.output_len,
            **self.generate_configs,
        )

        # Generate positions within the new trees where subtrees will be inserted
        newtrees_positions = randint(
            size=(newtrees.pop_size,),
            low=1,
            high=newtrees.batch_subtree_size[:, 0],
            dtype=torch.int32,
        )

        # Insert the subtrees into the new trees at selected positions
        newtrees = newtrees.mutate(newtrees_positions, subtrees)

        # Insert the new trees into the original forest at the selected positions
        forest[mutate_indices] = forest_to_mutate.mutate(
            mutate_positions.to(torch.int32), newtrees
        )

        return forest
