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
from evogp.problem import SymbolicRegression

# Forest.set_timmer_mode(True)

def func(x):
    val = x[0] ** 4 / (x[0] ** 4 + 1) + x[1] ** 4 / (x[1] ** 4 + 1)
    return val.reshape(-1)

problem = SymbolicRegression(func=func, num_inputs=2, num_data=2000, lower_bounds=-5, upper_bounds=5)

generate_configs = Forest.random_generate_check(
    pop_size=1,
    gp_len=1024,
    input_len=2,
    output_len=1,
    const_prob=0.5,
    out_prob=0.5,
    func_prob={"+": 0.20, "-": 0.20, "*": 0.20, "/": 0.20, "pow": 0.20},
    layer_leaf_prob=0.2,
    const_range=(-5, 5),
    sample_cnt=8,
    max_layer_cnt=5
)

algorithm = GeneticProgramming(
    crossover=DefaultCrossover(),
    mutation=DefaultMutation(mutation_rate=0.2, generate_configs=generate_configs),
    selection=DefaultSelection(survival_rate=0.3, elite_rate=0.01),
)

# initialize the forest
forest = Forest.random_generate(
    pop_size=int(1e5),
    gp_len=128,
    input_len=2,
    output_len=1,
    **generate_configs,
)

algorithm.initialize(forest)
fitness = problem.evaluate(forest)

for i in range(30):
    tic = time.time()
    forest = algorithm.step(fitness, args_check=False)
    fitness = problem.evaluate(forest, execute_code=0, args_check=False)
    torch.cuda.synchronize()
    toc = time.time()
    print(
        f"step: {i}, max_fitness: {fitness.max()}, mean_fitness: {fitness.mean()}, time: {toc - tic}"
    )