import torch

torch.random.manual_seed(0)
torch.cuda.manual_seed(0)

from evogp.pipeline import StandardPipeline
from evogp.tree import Forest, GenerateDiscriptor
from evogp.algorithm import (
    GeneticProgramming,
    DefaultSelection,
    DefaultMutation,
    DefaultCrossover,
)
from evogp.problem import Classification


dataset_name = ["iris", "wine", "breast_cancer", "digits"]


multi_output = True

problem = Classification(multi_output, dataset="iris")

descriptor = GenerateDiscriptor(
    max_tree_len=64,
    input_len=problem.problem_dim,
    output_len=problem.solution_dim,
    using_funcs=["+", "-", "*", "/"],
    max_layer_cnt=4,
    const_samples=[-1, 0, 1],
)


algorithm = GeneticProgramming(
    initial_forest=Forest.random_generate(pop_size=1000, descriptor=descriptor),
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

best = pipeline.run()

best.to_png("./imgs/classifier_tree.png")
sympy_expression = best.to_sympy_expr()
print(sympy_expression)