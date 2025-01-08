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

problem = SymbolicRegression(func=func, num_inputs=2, num_data=128, lower_bounds=-5, upper_bounds=5)

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
    pop_size=int(1000000),
    gp_len=400,
    input_len=2,
    output_len=1,
    **generate_configs,
)

algorithm.initialize(forest)
fitness0 = problem.evaluate(forest)

for i in range(50):
    torch.cuda.synchronize()
    total_tic = time.time()

    forest = algorithm.step(fitness0, args_check=False)

    torch.cuda.synchronize()
    advance_tic = time.time()
    fitness0 = problem.evaluate(forest, execute_code=1, args_check=False)
    torch.cuda.synchronize()
    advance_time = time.time() - advance_tic

    # torch.cuda.synchronize()
    # normal_tic = time.time()
    # fitness0 = problem.evaluate(forest, execute_code=0, args_check=False)
    # torch.cuda.synchronize()
    # normal_time = time.time() - normal_tic

    total_time = time.time() - total_tic
    print(
        f"step: {i}, max_fitness_normal: {fitness0.max()}, mean_fitness: {fitness0.mean()}, time: {advance_time}"
    )
    # print(f"step: {i}, max_fitness_advance: {fitness3.max()}, mean_fitness: {fitness3.mean()}, time: {advance_time}")
    print(total_time)