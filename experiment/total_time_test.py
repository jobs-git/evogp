import torch

torch.random.manual_seed(0)
torch.cuda.manual_seed(0)

import time
from evogp.tree import Forest, MAX_STACK
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


def run_once(popsize, data_size):
    problem = SymbolicRegression(
        func=func, num_inputs=2, num_data=data_size, lower_bounds=-5, upper_bounds=5
    )
    algorithm = GeneticProgramming(
        crossover=DefaultCrossover(),
        mutation=DefaultMutation(
            mutation_rate=0.2,
            generate_configs=Forest.random_generate_check(
                pop_size=1,
                gp_len=MAX_STACK,
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
        selection=DefaultSelection(survivor_rate=0.3, elite_rate=0.01),
    )
    forest = Forest.random_generate(
        pop_size=popsize,
        gp_len=MAX_STACK,
        input_len=2,
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

    total_time_tic = time.time()
    for i in range(50):
        forest = algorithm.step(fitness, args_check=False)
        fitness = problem.evaluate(forest, execute_code=3, args_check=False)
    torch.cuda.synchronize()
    total_time_toc = time.time()
    print(total_time_toc - total_time_tic)



def main():
    run_once(10000, 10000)


if __name__ == "__main__":
    main()
