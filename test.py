import torch

torch.random.manual_seed(0)
torch.cuda.manual_seed(0)

import time
from evogp.tree import Tree, Forest, MAX_STACK
from evogp.algorithm import *
from evogp.problem import SymbolicRegression

XOR_INPUTS = torch.tensor(
    [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ],
    dtype=torch.float,
    requires_grad=False,
    device="cuda",
)

XOR_OUTPUTS = torch.tensor(
    [[0], [1], [1], [0], [1], [0], [0], [1]],
    dtype=torch.float,
    requires_grad=False,
    device="cuda",
)

problem = SymbolicRegression(datapoints=XOR_INPUTS, labels=XOR_OUTPUTS)


algorithm = GeneticProgramming(
    crossover=LeafBiasedCrossover(
        crossover_rate=0.8,
        donor_selector=RouletteSelector(),
        recipient_selector=RouletteSelector(),
    ),
    mutation=MultiConstMutation(
        mutation_rate=0.2,
        generate_configs=Forest.random_generate_check(
            pop_size=1,
            gp_len=1024,
            input_len=2,
            output_len=1,
            const_prob=0.5,
            out_prob=0.5,
            func_prob={"+": 0.25, "-": 0.25, "*": 0.25, "/": 0.25},
            layer_leaf_prob=0.2,
            const_range=(-1, 1),
            sample_cnt=100,
            max_layer_cnt=3,
        ),
    ),
    selection=RouletteSelection(),
)

# initialize the forest
forest = Forest.random_generate(
    pop_size=100,
    gp_len=512,
    input_len=3,
    output_len=1,
    const_prob=0.5,
    out_prob=0.5,
    func_prob={"+": 0.25, "-": 0.25, "*": 0.25, "/": 0.25},
    layer_leaf_prob=0.2,
    const_range=(-1, 1),
    sample_cnt=100,
    max_layer_cnt=5,
)

algorithm.initialize(forest)
fitness = problem.evaluate(forest)

for i in range(50):
    tic = time.time()
    forest = algorithm.step(fitness)
    fitness = problem.evaluate(forest)
    toc = time.time()
    print(
        f"step: {i}, max_fitness: {fitness.max()}, mean_fitness: {fitness.mean()}, time: {toc - tic}"
    )
