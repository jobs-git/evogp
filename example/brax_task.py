from evogp.problem.brax_problem import BraxProblem

import torch

torch.random.manual_seed(0)
torch.cuda.manual_seed(0)

import time
from evogp.tree import Forest
from evogp.algorithm import (
    GeneticProgramming,
    DefaultSelection,
    DefaultMutation,
    DefaultCrossover,
)
from evogp.problem import SymbolicRegression

# Forest.set_timmer_mode(True)

problem = BraxProblem(
    "swimmer",
    max_episode_length=1000,
)

generate_configs = Forest.random_generate_check(
    pop_size=1,
    gp_len=128,
    input_len=problem.obs_dim,
    output_len=problem.action_dim,
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
    pop_size=int(1000),
    gp_len=128,
    input_len=problem.obs_dim,
    output_len=problem.action_dim,
    **generate_configs,
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
    # print(
    #     f"step: {i}, max_fitness: {fitness.max()}, mean_fitness: {fitness.mean()}, time: {toc - tic}"
    # )