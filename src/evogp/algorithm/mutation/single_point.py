from typing import Optional, Tuple
import torch
from torch import Tensor

from .base import BaseMutation
from ...tree import Forest, MAX_STACK, NType, randint, GenerateDiscriptor


class SinglePointMutation(BaseMutation):
    """
    SinglePointMutation implements a mutation strategy where a random node in the tree is selected 
    and replaced with a new node of the same type, chosen randomly from a node pool.
    This operation helps maintain the structure of the tree while introducing variation by changing individual nodes.
    """

    def __init__(
        self,
        mutation_rate: float,
        descriptor: GenerateDiscriptor,
    ):
        """
        Args:
            mutation_rate (float): The probability of each individual undergoing mutation. Should be between 0 and 1.
            descriptor (GenerateDiscriptor): The descriptor used to generate random subtrees for mutation.
        """
        self.mutation_rate = mutation_rate
        self.descriptor = descriptor

    def __call__(self, forest: Forest):
        """
        Perform the single-point mutation by randomly selecting a node in the tree and replacing it 
        with a new node of the same type from the node pool.

        Args:
            forest (Forest): The current population of trees (Forest object).

        Returns:
            Forest: The updated population after mutation, where some individuals have undergone the single-point mutation.
        """
        # Determine which trees need to mutate based on the mutation rate
        mutate_indices = torch.rand(forest.pop_size) < self.mutation_rate

        # If no trees are selected for mutation, return the original forest
        if mutate_indices.sum() == 0:  
            return forest

        # Extract the subset of trees that need to mutate
        forest_to_mutate = forest[mutate_indices]

        # Generate random mutation positions within the trees
        num_mutate = forest_to_mutate.pop_size
        mutate_positions = randint(
            size=(num_mutate,),
            low=0,
            high=forest_to_mutate.batch_subtree_size[:, 0],
            dtype=torch.int32,
        )

        # Randomly generate a constant for mutation
        random_idx = torch.randint(
            low=0,
            high=self.descriptor.const_samples.shape[0],
            size=(num_mutate,),
            device="cuda",
        )
        random_const = self.descriptor.const_samples[random_idx]

        # Randomly generate other types of node values based on the mutated node type
        mutated_node_type = forest_to_mutate.batch_node_type[
            torch.arange(num_mutate), mutate_positions
        ]
        mapping_range = torch.tensor(
            [
                [0, forest.input_len],  # VAR
                [0, 1],  # CONST
                [12, 24],  # UFUNC
                [1, 12],  # BFUNC
                [0, 1],  # TFUNC
            ],
            device="cuda",
        )
        random_other = randint(
            size=(num_mutate,),
            low=mapping_range[mutated_node_type.to(torch.int32)][:, 0],
            high=mapping_range[mutated_node_type.to(torch.int32)][:, 1],
        )

        # Mutate the selected nodes by replacing them with the new node values (either constant or other)
        forest.batch_node_value[mutate_indices, mutate_positions] = torch.where(
            mutated_node_type == NType.CONST, random_const, random_other
        )

        return forest
