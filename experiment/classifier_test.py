import torch

torch.random.manual_seed(0)
torch.cuda.manual_seed(0)

import time
from evogp.tree import Forest
from evogp.algorithm import (
    GeneticProgramming,
    DefaultSelection,
    TruncationSelection,
    DefaultMutation,
    DefaultCrossover,
    DiversityCrossover,
    TournamentSelector,
)
from evogp.problem import Classification


def print_color(text):
    print(f"\033[33m{text}\033[0m")


classification_problem = {
    "iris": {"input_len": 4, "output_len": 3, "sample_cnt": 150},
    "wine": {"input_len": 13, "output_len": 3, "sample_cnt": 178},
    "breast_cancer": {"input_len": 30, "output_len": 2, "sample_cnt": 569},
    "digits": {"input_len": 64, "output_len": 10, "sample_cnt": 1797},
}

name = list(classification_problem.keys())[2]
multi_output = True
print_color(f"Problem: {name}, multi_output: {multi_output}")

problem = Classification(multi_output, dataset=name)

generate_configs = Forest.random_generate_check(
    pop_size=1,
    gp_len=256,
    input_len=classification_problem[name]["input_len"],
    output_len=classification_problem[name]["output_len"] if multi_output else 1,
    const_prob=0.5,
    out_prob=0.5,
    func_prob={"+": 0.20, "-": 0.20, "*": 0.20, "/": 0.20, "pow": 0.20},
    layer_leaf_prob=0.2,
    const_range=(-5, 5),
    sample_cnt=8,
    max_layer_cnt=5,
)

algorithm = GeneticProgramming(
    crossover=DiversityCrossover(
        crossover_rate=0.9,
        recipient_selector=TournamentSelector(20, best_probability=0.9),
        donor_selector=TournamentSelector(20, best_probability=0.9),
    ),
    mutation=DefaultMutation(mutation_rate=0.2, generate_configs=generate_configs),
    selection=TruncationSelection(survivor_rate=1, elite_rate=0.01),
)

# initialize the forest
forest = Forest.random_generate(
    pop_size=int(5000),
    gp_len=256,
    input_len=classification_problem[name]["input_len"],
    output_len=classification_problem[name]["output_len"] if multi_output else 1,
    **generate_configs,
)

algorithm.initialize(forest)
fitness = problem.evaluate(forest)

for i in range(100):
    tic = time.time()
    forest = algorithm.step(fitness, args_check=False)
    fitness = problem.evaluate(forest)
    torch.cuda.synchronize()
    toc = time.time()
    print(f"step: {i}, max_fitness: {fitness.max()}")
    # print(
    #     f"step: {i}, max_fitness: {fitness.max()}, mean_fitness: {fitness.mean()}, time: {toc - tic}"
    # )
