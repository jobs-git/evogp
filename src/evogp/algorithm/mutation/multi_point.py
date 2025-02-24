from typing import Optional, Tuple
import torch
from torch import Tensor

from .base import BaseMutation
from ...tree import Forest, MAX_STACK, randint, NType, GenerateDescriptor


class MultiPointMutation(BaseMutation):
    """
    MultiPointMutation implements a mutation strategy where a specific number of nodes 
    within an individual (tree) are selected based on the `mutation_intensity` parameter 
    and each of these selected nodes undergoes a SinglePointMutation. This helps introduce 
    more diversity into the individual by making multiple small changes.
    """

    def __init__(
        self,
        mutation_rate: float,
        descriptor: GenerateDescriptor,
        mutation_intensity: float = 0.3,
    ):
        """
        Args:
            mutation_rate (float): The probability of each individual undergoing mutation.
            descriptor (GenerateDescriptor): The descriptor used to generate random subtrees for mutation.
            mutation_intensity (float): Determines the proportion of nodes in the tree that will be mutated. 
                                        It is a fraction between 0 and 1.
        """
        self.mutation_rate = mutation_rate
        self.descriptor = descriptor
        self.mutation_intensity = mutation_intensity

    def __call__(self, forest: Forest):
        """
        Perform the multi-point mutation where a specific proportion of nodes in the tree are selected 
        and each undergoes SinglePointMutation.

        Args:
            forest (Forest): The current population of trees (Forest object).

        Returns:
            Forest: The updated population after mutation, where some individuals have undergone multi-point mutation.
        """
        # Determine which trees need to mutate based on the mutation rate
        mutate_indices = torch.rand(forest.pop_size) < self.mutation_rate

        # If no trees are selected for mutation, return the original forest
        if mutate_indices.sum() == 0:  
            return forest

        # Extract the subset of trees that need to mutate
        forest_to_mutate = forest[mutate_indices]

        def choose_mutation_targets(size_tensor):
            """
            Randomly choose mutation targets (nodes to mutate) based on mutation intensity.
            A specific number of nodes are selected as mutation targets in each tree.

            Args:
                size_tensor (Tensor): The size of each tree.

            Returns:
                Tensor: A tensor indicating which nodes should be mutated (True/False).
            """
            tree_size = size_tensor[:, 0].reshape(-1, 1)
            random = torch.rand(tree_size.shape, device="cuda")
            return (random < self.mutation_intensity) & (
                torch.arange(size_tensor.shape[1], device="cuda") < tree_size
            )

        # Generate mutation indices and positions based on mutation intensity
        mutation_targets = choose_mutation_targets(forest_to_mutate.batch_subtree_size)
        num_targets = mutation_targets.sum()

        # Randomly generate constant values for the mutation
        random_idx = torch.randint(
            low=0,
            high=self.descriptor.const_samples.shape[0],
            size=(num_targets,),
            device="cuda",
        )
        random_const = self.descriptor.const_samples[random_idx]

        # Randomly generate other types of node values based on the mutated node type
        mutated_node_type = forest_to_mutate.batch_node_type[mutation_targets]
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
            size=(num_targets,),
            low=mapping_range[mutated_node_type.to(torch.int32)][:, 0],
            high=mapping_range[mutated_node_type.to(torch.int32)][:, 1],
        )

        # Mutate the selected nodes by replacing them with new node values (either constant or other)
        forest_to_mutate.batch_node_value[mutation_targets] = torch.where(
            mutated_node_type == NType.CONST, random_const, random_other
        )
        
        # Update the forest with the mutated trees
        forest[mutate_indices] = forest_to_mutate
        return forest
