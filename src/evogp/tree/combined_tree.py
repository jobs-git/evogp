from typing import Optional, Tuple, Callable, List, Tuple, Union
import warnings
import torch
from torch import Tensor
import sympy as sp

from .utils import check_formula
from .descriptor import GenerateDiscriptor
from .combined_forest import check_formula


class CombinedTree:
    def __init__(self, formula, trees, share_input: bool = True):
        self.parameter_names = check_formula(formula)
        self.formula = formula
        self.trees = trees
        self.input_len = (
            trees[0].input_len
            if share_input
            else sum([tree.input_len for tree in trees])
        )
        self.output_len = trees[0].output_len
        for tree in trees:
            assert (
                tree.output_len == self.output_len
            ), f"all forests should have the same output_len, but got {tree.output_len} and {self.output_len}"

        self.share_input = share_input

        for i, parameter in enumerate(self.parameter_names):
            setattr(self, parameter, trees[i])

    @staticmethod
    def random_generate(
        formula: Callable,
        descriptors: Union[List, GenerateDiscriptor],
        share_input: bool = True,
    ):
        from .combined_forest import CombinedForest

        return CombinedForest.random_generate(
            pop_size=1,
            formula=formula,
            descriptors=descriptors,
            share_input=share_input,
        )[0]

    def forward(self, x):
        if self.share_input:
            assert not isinstance(
                x, list
            ), "x should not be a list when share_input=True"
            is_batch = x.dim() == 2  # (batch_size, input_len)
        else:
            assert isinstance(x, list), "x should be a list when share_input=False"
            is_batch = x[0].dim() == 2  # (batch_size, input_len)

        if not is_batch:
            pop_res = self.to_combined_forest().forward(x)
        else:
            pop_res = self.to_combined_forest().batch_forward(x)

        return pop_res[0]  # remove the pop dimension

    def to_combined_forest(self):
        from .combined_forest import CombinedForest

        return CombinedForest(
            formula=self.formula,
            forests=[tree.to_forest() for tree in self.trees],
            share_input=self.share_input,
        )

    def to_sympy_expr(self, sympy_formula=None):
        if sympy_formula is None:
            warnings.warn(
                "sympy_formula is None, using default formula", RuntimeWarning
            )
            sympy_formula = self.formula

        sub_exprs = [t.to_sympy_expr() for t in self.trees]

        return sympy_formula(*sub_exprs)
