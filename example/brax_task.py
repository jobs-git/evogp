from evogp.problem import BraxProblem

import torch

torch.random.manual_seed(0)
torch.cuda.manual_seed(0)

from evogp.tree import Forest, GenerateDescriptor
from evogp.algorithm import (
    GeneticProgramming,
    DefaultSelection,
    DefaultMutation,
    DefaultCrossover,
)
from evogp.pipeline import StandardPipeline

problem = BraxProblem(
    "swimmer",
    max_episode_length=1000,
)

descriptor = GenerateDescriptor(
    max_tree_len=128,
    input_len=problem.problem_dim,
    output_len=problem.solution_dim,
    using_funcs=["+", "-", "*", "/"],
    max_layer_cnt=5,
    const_samples=[-1, 0, 1],
)


algorithm = GeneticProgramming(
    initial_forest=Forest.random_generate(pop_size=100, descriptor=descriptor),
    crossover=DefaultCrossover(),
    mutation=DefaultMutation(
        mutation_rate=0.2, descriptor=descriptor.update(max_layer_cnt=3)
    ),
    selection=DefaultSelection(survival_rate=0.3, elite_rate=0.01),
)

pipeline = StandardPipeline(
    algorithm,
    problem,
    generation_limit=100,
)

pipeline.run()
