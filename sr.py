from typing import Callable
import torch

torch.random.manual_seed(0)
torch.cuda.manual_seed(0)

import time
from evogp.tree import Tree, Forest, MAX_STACK
from evogp.algorithm import (
    GeneticProgramming,
    DefaultSelection,
    DefaultMutation,
    DefaultCrossover,
)


def func(x):
    val = x[0] ** 4 / (x[0] ** 4 + 1) + x[1] ** 4 / (x[1] ** 4 + 1)
    return val.reshape(-1)

def create_dataset(
    func: Callable, num_samples: int, input_size: int, sample_range: tuple = (0, 1)
) -> torch.Tensor:
    inputs = torch.empty((num_samples, input_size), device="cuda", requires_grad=False).uniform_(*sample_range)
    outputs = torch.vmap(func)(inputs)

    return inputs, outputs


data_inputs, data_outputs = create_dataset(func, int(1e3), 2, (-5, 5))


def evaluate(forest: Forest):
    tic = time.time()
    constant_loss = forest.SR_fitness(data_inputs, data_outputs, execute_mode="constant")
    constant_time = time.time() - tic
    # print("constant time: {}".format(constant_time))

    tic = time.time()
    normal_loss = forest.SR_fitness(data_inputs, data_outputs, execute_mode="normal")
    normal_time = time.time() - tic

    assert torch.allclose(constant_loss, normal_loss)
    print("constant time: {}, normal time: {}".format(constant_time, normal_time))
    return -constant_loss


generate_configs = {
    "const_prob": 0.5,
    "out_prob": 0.5,
    "func_prob": {"+": 0.20, "-": 0.20, "*": 0.20, "/": 0.20, "pow": 0.20},
    "layer_leaf_prob": 0.2,
    "const_range": (-5, 5),
    "sample_cnt": 8,
}

algorithm = GeneticProgramming(
    crossover=DefaultCrossover(),
    mutation=DefaultMutation(mutation_rate=0.2, max_layer_cnt=3, **generate_configs),
    selection=DefaultSelection(survival_rate=0.3, elite_rate=0.01),
)

# initialize the forest
forest = Forest.random_generate(
    pop_size=100000,
    gp_len=512,
    input_len=2,
    output_len=1,
    max_layer_cnt=5,
    **generate_configs,
)

algorithm.initialize(forest)
fitness = evaluate(forest)

for i in range(1000):
    tic = time.time()
    forest = algorithm.step(fitness)
    fitness = evaluate(forest)
    toc = time.time()
    # print(
    #     f"step: {i}, max_fitness: {fitness.max()}, mean_fitness: {fitness.mean()}, time: {toc - tic}"
    # )
