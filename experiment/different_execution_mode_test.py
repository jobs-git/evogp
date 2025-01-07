import json
import time
import os
import torch
import gc

torch.random.manual_seed(0)
torch.cuda.manual_seed(0)

from evogp.tree import Forest, MAX_STACK
from evogp.algorithm import (
    GeneticProgramming,
    DefaultSelection,
    DefaultMutation,
    DefaultCrossover,
)
from evogp.problem import SymbolicRegression

OUTPUT_FILE = os.path.join(
    "./experiment/data", os.path.splitext(os.path.basename(__file__))[0] + ".json"
)
REPEAT_TIMES = 10


def func(x):
    val = x[0] ** 4 / (x[0] ** 4 + 1) + x[1] ** 4 / (x[1] ** 4 + 1)
    return val.reshape(-1)


def run_once(popsize, data_size, execution_code=0):
    problem = SymbolicRegression(
        func=func, num_inputs=2, num_data=data_size, lower_bounds=-5, upper_bounds=5
    )
    print("generate dataset success")
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
        selection=DefaultSelection(survival_rate=0.3, elite_rate=0.01),
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

    info = {
        "popsize": popsize,
        "data_size": data_size,
        "execution_code": execution_code,
    }

    algorithm.initialize(forest)
    fitness = problem.evaluate(forest)
    torch.cuda.synchronize()
    total_sr_time = 0
    total_time_tic = time.time()
    for i in range(50):
        print(i)
        forest = algorithm.step(fitness, args_check=False)
        torch.cuda.synchronize()
        SR_tic = time.time()
        fitness = problem.evaluate(
            forest, execute_code=execution_code, args_check=False
        )
        torch.cuda.synchronize()
        total_sr_time += time.time() - SR_tic

    torch.cuda.synchronize()
    info["SR_time"] = total_sr_time
    info["total_time"] = time.time() - total_time_tic

    # write file
    with open(OUTPUT_FILE, "r") as f:
        previous_records = json.load(f)

    previous_records.append(info)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(previous_records, f, indent=4)


def main():

    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "w") as f:
            json.dump([], f)

    # create file
    for p in [ 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000,]:
        for d in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]:
            for _ in range(REPEAT_TIMES):
                gc.collect()
                torch.cuda.empty_cache()
                run_once(p, d, execution_code=3)  # advanced

if __name__ == "__main__":
    print(OUTPUT_FILE)
    main()
