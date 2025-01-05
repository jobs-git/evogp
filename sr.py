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

Forest.set_timmer_mode(True)

def func(x):
    val = x[0] ** 4 / (x[0] ** 4 + 1) + x[1] ** 4 / (x[1] ** 4 + 1)
    return val.reshape(-1)


def create_dataset(
    func: Callable, num_samples: int, input_size: int, sample_range: tuple = (0, 1)
) -> torch.Tensor:
    inputs = torch.empty(
        (num_samples, input_size), device="cuda", requires_grad=False
    ).uniform_(*sample_range)
    outputs = torch.vmap(func)(inputs)

    return inputs, outputs


data_inputs, data_outputs = create_dataset(func, int(100), 2, (-5, 5))


def evaluate(forest: Forest):

    # advanced_loss = forest.SR_fitness(
    #     data_inputs, data_outputs, execute_mode="advanced"
    # )
    # advanced_loss = forest.SR_fitness(
    #     data_inputs, data_outputs, execute_mode="advanced"
    # )
    # advanced_loss = forest.SR_fitness(
    #     data_inputs, data_outputs, execute_mode="advanced"
    # )
    # advanced_loss = forest.SR_fitness(
    #     data_inputs, data_outputs, execute_mode="advanced"
    # )
    # advanced_loss = forest.SR_fitness(
    #     data_inputs, data_outputs, execute_mode="advanced"
    # )
    # torch.cuda.empty_cache()
    # tic = time.time()
    # advanced_loss = forest.SR_fitness(
    #     data_inputs, data_outputs, execute_mode="advanced"
    # )
    # torch.cuda.synchronize()
    # advanced_time = time.time() - tic

    # normal_loss = forest.SR_fitness(data_inputs, data_outputs, execute_mode="normal")
    # normal_loss = forest.SR_fitness(data_inputs, data_outputs, execute_mode="normal")
    # normal_loss = forest.SR_fitness(data_inputs, data_outputs, execute_mode="normal")
    # normal_loss = forest.SR_fitness(data_inputs, data_outputs, execute_mode="normal")
    # normal_loss = forest.SR_fitness(data_inputs, data_outputs, execute_mode="normal")
    # normal_loss = forest.SR_fitness(data_inputs, data_outputs, execute_mode="normal")
    # torch.cuda.empty_cache()
    # tic = time.time()
    # normal_loss = forest.SR_fitness(data_inputs, data_outputs, execute_mode="normal")
    # torch.cuda.synchronize()
    # normal_time = time.time() - tic


    # tic = time.time()
    # tree_loop_loss = forest.SR_fitness(
    #     data_inputs, data_outputs, execute_mode="tree_loop"
    # )
    # tree_loop_time = time.time() - tic
    # tree_loop_time = 0


    # for i in range(len(forest)):
    #     assert torch.allclose(tree_loop_loss[i], advanced_loss[i], rtol=1e-5), f"{tree_loop_loss[i]}, {advanced_loss[i]}"

    # assert torch.allclose(tree_loop_loss, normal_loss), f"{tree_loop_loss}, {normal_loss}"

    # assert torch.allclose(tree_loop_loss, advanced_loss), f"{tree_loop_loss}, {advanced_loss}"
    # assert torch.allclose(normal_loss, advanced_loss), f"{normal_loss}, {advanced_loss}"

    # print(
    #     f"normal time: {normal_time}, advanced time: {advanced_time}"
    # )

    normal_loss = forest.SR_fitness(data_inputs, data_outputs, execute_mode="normal")
    advanced_loss = forest.SR_fitness(
        data_inputs, data_outputs, execute_mode="advanced"
    )

    return -advanced_loss


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
    pop_size=int(1e6),
    gp_len=128,
    input_len=2,
    output_len=1,
    max_layer_cnt=5,
    **generate_configs,
)

algorithm.initialize(forest)
fitness = evaluate(forest)

for i in range(10):
    tic = time.time()
    forest = algorithm.step(fitness)
    fitness = evaluate(forest)
    toc = time.time()
    print(
        f"step: {i}, max_fitness: {fitness.max()}, mean_fitness: {fitness.mean()}, time: {toc - tic}"
    )

print(Forest.get_timer_record())