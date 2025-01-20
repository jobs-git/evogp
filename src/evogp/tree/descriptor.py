import warnings
from typing import Optional, Tuple, Union
import torch
from torch import Tensor
from utils import MAX_STACK, MAX_FULL_DEPTH, dict2cdf, check_tensor, Func


class GenerateDiscriptor:
    def __init__(
        self,
        tree_max_len: int,
        input_len: int,
        output_len: int,
        const_prob: float,
        out_prob: float = 0.0,
        depth2leaf_probs: Optional[Tensor] = None,
        roulette_funcs: Optional[Tensor] = None,
        const_samples: Optional[Tensor] = None,
        using_funcs: Optional[Union[dict, list]] = None,
        max_layer_cnt: Optional[int] = None,
        layer_leaf_prob: Optional[float] = None,
        const_range: Optional[Tuple[float, float]] = None,
        sample_cnt: Optional[int] = None,
    ):

        self.__params = {key: value for key, value in locals().items() if key != "self"}

        assert (
            tree_max_len <= MAX_STACK
        ), f"tree_max_len={tree_max_len} is too large, MAX_STACK={MAX_STACK}"

        assert (
            isinstance(input_len, int) and input_len > 0
        ), "input_len should be a positive integer"
        assert (
            isinstance(output_len, int) and output_len > 0
        ), "output_len should be a positive integer"
        assert (
            const_prob >= 0.0 and const_prob <= 1.0
        ), "const_prob should be in [0.0, 1.0]"

        assert out_prob >= 0.0 and out_prob <= 1.0, "out_prob should be in [0.0, 1.0]"

        if output_len > 1 and out_prob == 0.0:
            warnings.warn(
                f"output_len={output_len} > 1, but out_prob={out_prob} is 0.0."
            )

        if depth2leaf_probs is None:
            assert (
                max_layer_cnt is not None
            ), "max_layer_cnt should not be None when depth2leaf_probs is None"
            assert (
                layer_leaf_prob is not None
            ), "layer_leaf_prob should not be None when depth2leaf_probs is None"
            assert (
                2**max_layer_cnt <= gp_len
            ), f"max_layer_cnt is too large for gp_len={gp_len}"

            depth2leaf_probs = torch.tensor(
                [layer_leaf_prob] * max_layer_cnt
                + [1.0] * (MAX_FULL_DEPTH - max_layer_cnt),
                device="cuda",
                requires_grad=False,
            )

        if roulette_funcs is None:
            assert (
                using_funcs is not None
            ), "func_prob should not be None when roulette_funcs is None"
            assert isinstance(using_funcs, dict) or isinstance(
                using_funcs, list
            ), "func_prob should be a dictionary or a list"

            if isinstance(using_funcs, list):
                using_funcs = {f: 1.0 for f in using_funcs}

            roulette_funcs = torch.tensor(
                dict2cdf(using_funcs),
                dtype=torch.float32,
                device="cuda",
                requires_grad=False,
            )

        if const_samples is None:
            assert (
                const_range is not None
            ), "const_range should not be None when const_samples is None"
            assert (
                sample_cnt is not None
            ), "sample_cnt should not be None when const_samples is None"
            const_samples = (
                torch.rand(sample_cnt, device="cuda")
                * (const_range[1] - const_range[0])
                + const_range[0]
            )

        check_tensor(depth2leaf_probs)
        check_tensor(roulette_funcs)
        check_tensor(const_samples)

        assert depth2leaf_probs.shape == (
            MAX_FULL_DEPTH,
        ), f"depth2leaf_probs shape should be ({MAX_FULL_DEPTH}), but got {depth2leaf_probs.shape}"

        assert roulette_funcs.shape == (
            Func.END,
        ), f"roulette_funcs shape should be ({Func.END}), but got {roulette_funcs.shape}"
        assert (
            const_samples.dim() == 1
        ), f"const_samples dim should be 1, but got {const_samples.dim()}"

        self.tree_max_len = tree_max_len
        self.input_len = input_len
        self.output_len = output_len
        self.const_prob = const_prob
        self.out_prob = out_prob
        self.depth2leaf_probs = depth2leaf_probs
        self.roulette_funcs = roulette_funcs
        self.const_samples = const_samples

    def update(self, **kwargs):
        self.__params.update(kwargs)
        return self.__class__(**self.__params)
