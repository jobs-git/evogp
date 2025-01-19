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
from evogp.problem import Transformation


def print_color(text):
    print(f"\033[33m{text}\033[0m")


transformation_problem = {
    "diabetes": {"input_len": 10},
}
name = list(transformation_problem.keys())[0]
print_color(f"Problem: {name}")

problem = Transformation(dataset=name)

generate_configs = Forest.random_generate_check(
    pop_size=1,
    gp_len=128,
    input_len=transformation_problem[name]["input_len"],
    output_len=1,
    const_prob=0.5,
    out_prob=0.5,
    func_prob={"+": 0.20, "-": 0.20, "*": 0.20, "/": 0.20, "pow": 0.20},
    layer_leaf_prob=0.2,
    const_range=(-5, 5),
    sample_cnt=8,
    max_layer_cnt=5,
)

algorithm = GeneticProgramming(
    crossover=DefaultCrossover(),
    mutation=DefaultMutation(mutation_rate=0.2, generate_configs=generate_configs),
    selection=DefaultSelection(survival_rate=0.3, elite_rate=0.01),
)

# initialize the forest
forest = Forest.random_generate(
    pop_size=int(5000),
    gp_len=128,
    input_len=transformation_problem[name]["input_len"],
    output_len=1,
    **generate_configs,
)

algorithm.initialize(forest)
fitness = problem.evaluate(forest)


from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge

diabetes = load_diabetes()
estimator = Ridge()

# before transformation
estimator.fit(diabetes.data, diabetes.target)
print(estimator.score(diabetes.data, diabetes.target))

# transformation
for i in range(100):
    tic = time.time()
    forest = algorithm.step(fitness, args_check=False)
    fitness = problem.evaluate(forest)
    torch.cuda.synchronize()
    toc = time.time()
    print(f"step: {i}, max_fitness: {fitness.max()}")

new_diabetes = torch.cat(
    (problem.datapoints, problem.new_feature(forest, 100, 10)), dim=1
)

# after transformation
estimator.fit(new_diabetes.cpu().numpy(), diabetes.target)
print(estimator.score(new_diabetes.cpu().numpy(), diabetes.target))
