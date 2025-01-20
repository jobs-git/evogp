import torch

torch.random.manual_seed(0)
torch.cuda.manual_seed(0)

import time
from evogp.tree import Forest, GenerateDiscriptor
from evogp.algorithm import (
    GeneticProgramming,
    DefaultSelection,
    DefaultMutation,
    DefaultCrossover,
)
from evogp.problem import SymbolicRegression

def func(x):
    val = x[0] ** 4 / (x[0] ** 4 + 1) + x[1] ** 4 / (x[1] ** 4 + 1)
    return val.reshape(-1)

problem = SymbolicRegression(func=func, num_inputs=2, num_data=20000, lower_bounds=-5, upper_bounds=5)

descriptor = GenerateDiscriptor(
    max_tree_len=128,
    input_len=2,
    output_len=1,
    const_prob=0.5,
    out_prob=0.5,
    using_funcs=["+", "-", "*", "/"],
    layer_leaf_prob=0.2,
    const_range=(-5, 5),
    sample_cnt=8,
    max_layer_cnt=5
)

algorithm = GeneticProgramming(
    crossover=DefaultCrossover(),
    mutation=DefaultMutation(mutation_rate=0.2, descriptor=descriptor.update(max_layer_cnt=3)),
    selection=DefaultSelection(survival_rate=0.3, elite_rate=0.01),
)

# initialize the forest
forest = Forest.random_generate(
    pop_size=int(1000),
    descriptor=descriptor
)

algorithm.initialize(forest)
fitness = problem.evaluate(forest)

for i in range(50):
    tic = time.time()
    forest = algorithm.step(fitness)
    fitness = problem.evaluate(forest)
    torch.cuda.synchronize()
    toc = time.time()
    print(f"step: {i}, max_fitness: {fitness.max()}")