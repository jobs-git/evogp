from typing import Optional, Callable

import torch
from torch import Tensor
from . import BaseProblem
from evogp.tree import Forest


class SymbolicRegression(BaseProblem):
    def __init__(
        self,
        datapoints: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        func: Optional[Callable] = None,
        num_inputs: Optional[int] = None,
        num_data: Optional[int] = 100,
        lower_bounds: Optional[Tensor] = -1,
        upper_bounds: Optional[Tensor] = 1,
    ):
        if datapoints is not None and labels is not None:
            self.datapoints = datapoints
            self.labels = labels
            return
        assert (
            func is not None and num_inputs is not None
        ), "func and num_inputs, must be provided when datapoints and labels are not provided"

        self.datapoints, self.labels = self.generate_data(
            func, num_inputs, num_data, lower_bounds, upper_bounds
        )

    def generate_data(self, func, num_inputs, num_data, lower_bounds, upper_bounds):
        if isinstance(upper_bounds, int) or isinstance(lower_bounds, float):
            upper_bounds = torch.full(
                (num_inputs,), upper_bounds, device="cuda", requires_grad=False
            )
        if isinstance(lower_bounds, int) or isinstance(lower_bounds, float):
            lower_bounds = torch.full(
                (num_inputs,), lower_bounds, device="cuda", requires_grad=False
            )
        upper_bounds = upper_bounds[None, :]
        lower_bounds = lower_bounds[None, :]

        inputs = (
            torch.rand(num_data, num_inputs, device="cuda", requires_grad=False)
            * (upper_bounds - lower_bounds)
            + lower_bounds
        )

        outputs = torch.vmap(func)(inputs)

        return inputs, outputs

    def evaluate(
        self,
        forest: Forest,
        use_MSE: bool = True,
        execute_mode: str = "auto",
    ):
        return -forest.SR_fitness(
            self.datapoints,
            self.labels,
            use_MSE,
            execute_mode,
        )
