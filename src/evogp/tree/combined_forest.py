from typing import Optional, Tuple, Callable, List, Tuple, Union
import time
import torch
from torch import Tensor
import numpy as np

from evogp.tree.utils import check_tensor

from .utils import check_formula
from .forest import Forest
from .descriptor import GenerateDiscriptor
from .combined_tree import CombinedTree


class CombinedForest:
    def __init__(
        self,
        formula: Callable,
        forests: List[Forest],
        share_input: bool = True,
    ):
        self.parameter_names = check_formula(formula)
        self.formula = formula
        self.forests = forests
        self.input_len = (
            forests[0].input_len
            if share_input
            else sum([f.input_len for f in forests])
        )
        self.output_len = forests[0].output_len
        for forest in forests:
            assert (
                forest.output_len == self.output_len
            ), f"all forests should have the same output_len, but got {forest.output_len} and {self.output_len}"

        self.pop_size = forests[0].pop_size
        self.share_input = share_input

    @staticmethod
    def random_generate(
        pop_size: int,
        formula: Callable,
        descriptors: Union[List, GenerateDiscriptor],
        share_input: bool = True,
    ):
        parameter_names = check_formula(formula)
        if isinstance(descriptors, GenerateDiscriptor):
            descriptors = [descriptors] * len(parameter_names)
        assert isinstance(descriptors, list) and len(descriptors) == len(
            parameter_names
        ), f"there are {len(parameter_names)} parameters, but got {len(descriptors)} descriptors"
        forests = [
            Forest.random_generate(pop_size=pop_size, descriptor=d) for d in descriptors
        ]
        return CombinedForest(formula=formula, forests=forests, share_input=share_input)

    # (pop_size, input_len) -> (pop_size, output_len)
    def forward(
        self,
        x: Union[torch.Tensor, List[torch.Tensor]],
    ):
        if self.share_input:
            assert not isinstance(
                x, list
            ), "x should not be a list when share_input=True"
            x = [x] * len(self.forests)
        else:
            assert isinstance(x, list), "x should be a list when share_input=False"
            assert len(x) == len(
                self.forests
            ), "x should have the same length as forests"
            for i in x:
                assert (
                    x[i].shape[0] == self.pop_size
                ), f"len(x) should be equal to pop_size, but got {x[i].shape[0]} and {self.pop_size}"

        pop_outputs = []
        for i, forest in enumerate(self.forests):
            inputs = check_tensor(x[i])
            pop_outputs.append(forest.forward(inputs))

        # redundant check
        for o in pop_outputs:
            assert o.shape == (self.pop_size, self.output_len)

        return self.formula(*pop_outputs)

    # (batch_size, input_len) -> (pop_size, batch_size, output_len)
    def batch_forward(self, x: Union[torch.Tensor, List[torch.Tensor]]):
        if self.share_input:
            assert not isinstance(
                x, list
            ), "x should not be a list when share_input=True"
            batch_size = x.shape[0]
            x = [x] * len(self.forests)

        else:
            assert isinstance(x, list), "x should be a list when share_input=False"
            assert len(x) == len(
                self.forests
            ), "x should have the same length as forests"
            batch_size = x[0].shape[0]
            for i in range(1, len(x)):
                assert (
                    x[i].shape[0] == batch_size
                ), f"x should have the same batch_size, but got {x[i].shape[0]} and {x[0].shape[0]}"

        pop_outputs = []
        for i, forest in enumerate(self.forests):
            inputs = check_tensor(x[i])
            pop_outputs.append(forest.batch_forward(inputs))
        
        # redundant check
        for o in pop_outputs:
            assert o.shape == (self.pop_size, batch_size, self.output_len), o.shape

        return self.formula(*pop_outputs)

    def __getitem__(self, index):
        if isinstance(index, int):
            return CombinedTree(
                formula=self.formula,
                trees=[f[index] for f in self.forests],
                share_input=self.share_input,
            )
        elif (
            isinstance(index, slice)
            or isinstance(index, Tensor)
            or isinstance(index, np.ndarray)
        ):
            return CombinedForest(
                formula=self.formula,
                forests=[f[index] for f in self.forests],
                share_input=self.share_input,
            )
        else:
            raise Exception("Do not support index type {}".format(type(index)))

    def __setitem__(self, index, value):
        if isinstance(index, int):
            assert isinstance(
                value, CombinedTree
            ), f"value should be Tree when index is int, but got {type(value)}"
            for i in range(len(self.forests)):
                self.forests[i][index] = value.trees[i]

        elif (
            isinstance(index, slice)
            or isinstance(index, Tensor)
            or isinstance(index, np.ndarray)
        ):
            assert isinstance(
                value, CombinedForest
            ), f"value should be Forest when index is slice, but got {type(value)}"
            for i in range(len(self.forests)):
                self.forests[i][index] = value.forests[i]
        else:
            raise NotImplementedError

    def __iter__(self):
        self.iter_index = 0
        return self

    def __next__(self):
        if self.iter_index < self.pop_size:
            return self[self.iter_index]
        else:
            raise StopIteration

    def __len__(self):
        return self.pop_size

    def __add__(self, other):
        assert self.formula == other.formula
        assert self.share_input == other.share_input
        if isinstance(other, CombinedForest):
            new_forests = []
            for i in range(len(self.forests)):
                new_forests.append(self.forests[i] + other.forests[i])
            return CombinedForest(
                self.formula,
                new_forests,
                self.share_input,
            )
        if isinstance(other, CombinedTree):
            new_forests = []
            for i in range(len(self.forests)):
                new_forests.append(self.forests[i] + other.tree[i])
            return CombinedForest(
                self.formula,
                new_forests,
                self.share_input,
            )
        else:
            raise NotImplementedError

    def __radd__(self, other):
        return self.__add__(other)
