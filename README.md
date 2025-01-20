<h1 align="center">
  <a href="https://github.com/EMI-Group/evox">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./imgs/evox_logo_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="./imgs/evox_logo_light.png">
      <img alt="EvoX Logo" height="50" src="./imgs/evox_logo_light.png">
  </picture>
  </a>
  <br>
</h1>
<p align="center">
🌟 EvoGP: A GPU-accelerated Library for Scalable Tree-Based Genetic Programming 🌟
</p>

<p align="center">
  <a href="https://arxiv.org/abs/">
    <img src="https://img.shields.io/badge/paper-arxiv-red?style=for-the-badge" alt="TensorNEAT Paper on arXiv">
  </a>
</p>

## Introduction
EvoGP is a GPU-accelerated library for tree-based genetic programming (TGP), designed to optimize the evolution of complex tree structures in tasks such as symbolic regression and feature engineering. By tensorizing tree structures, EvoGP enables parallel execution, improving computational speed and scalability through efficient GPU utilization. Its unified framework for genetic operations and parallel fitness evaluation further enhances performance, making it suitable for large-scale evolutionary computations. EvoGP is compatible with the [EvoX](https://github.com/EMI-Group/evox/) framework.

## Key Features
- **CUDA-based parallel approach for Tree-Based Genetic Programming (TGP)**:
  
    - Leverage specialized CUDA kernels to optimize critical TGP operations.
    - Enhance computational efficiency, especially for large populations, enabling faster execution compared to traditional TGP methods.

- **GPU-accelerated EvoGP framework**:
  
    - Integrates CUDA kernels into Python via Pybind and PyTorch, ensuring compatibility with modern computational ecosystems.
    - Seamlessly integrates with machine learning frameworks, making EvoGP easy to incorporate into existing workflows.
    
- **Customizable evolutionary operations**:
  
    - Offers a range of evolutionary operation variants, allowing users to tailor configurations for specific tasks.
    - Supports multi-output trees, making it suitable for complex problems like classification and policy optimization.
    
- **Significant performance improvements**:
  
    - Achieve up to a **100x** speedup compared to existing TGP implementations while maintaining or improving solution quality.
    - Extensive experiments demonstrate EvoGP's scalability and efficiency in real-world applications.
    

## Installation

To install EvoGP, please follow the steps below:

### 1. Install NVIDIA CUDA Toolkit  
Ensure you have the NVIDIA CUDA Toolkit installed, including `nvcc`. You can download it from [NVIDIA's official website](https://developer.nvidia.com/cuda-downloads).  
- Check your CUDA version:  
   ```bash
   nvcc --version
   ```

### 2. Install a C++ Compiler  
Ensure you have a compatible C++ compiler installed:  
- **Linux/macOS:** Install GCC (9.x or later is recommended).  
   ```bash
   sudo apt install build-essential  # On Ubuntu
   gcc --version
   ```
- **Windows:** Install the **Visual C++ Build Tools**. You can download it from [this](https://visualstudio.microsoft.com/visual-cpp-build-tools/). During installation, ensure that the **C++ workload** is selected. 

### 3. Install PyTorch  
Install the version of PyTorch that matches your installed CUDA Toolkit version.  
For example, if you are using CUDA 11.8:  
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
⚠️ **Important:** Make sure to select the PyTorch version compatible with the **CUDA Toolkit** version (`nvcc -V`), not the Nvidia driver version.  

You can find more details on the [PyTorch installation page](https://pytorch.org/get-started/locally/).

### 4. Install EvoGP  
Finally, install EvoGP:  
```bash
pip install git+https://github.com/EMI-Group/evogp.git
```

## Basic API Usage

Start your journey with EvoGP in a few simple steps:

1. **Import necessary modules**:
```python
from evogp.tree import Forest
from evogp.algorithm import (
    GeneticProgramming,
    DefaultSelection,
    DefaultMutation,
    DefaultCrossover,
)
from evogp.problem import SymbolicRegression
```

2. **Define a problem**:
```python
def func(x):
    val = x[0] ** 4 / (x[0] ** 4 + 1) + x[1] ** 4 / (x[1] ** 4 + 1)
    return val.reshape(-1)

problem = SymbolicRegression(func=func, num_inputs=2, num_data=20000, lower_bounds=-5, upper_bounds=5)
```

3. **Configure the algorithm**:

```python
generate_configs = Forest.random_generate_check(
   pop_size=1,
   gp_len=128,
   input_len=2,
   output_len=1,
   const_prob=0.5,
   out_prob=0.5,
   func_prob={"+": 0.20, "-": 0.20, "*": 0.20, "/": 0.20, "pow": 0.20},
   max_layer_cnt=5,
   layer_leaf_prob=0.2,
   const_range=(-5, 5),
   sample_cnt=8,
)
algorithm = GeneticProgramming(
   crossover=DefaultCrossover(),
   mutation=DefaultMutation(mutation_rate=0.2, generate_configs=generate_configs),
   selection=DefaultSelection(survival_rate=0.3, elite_rate=0.01),
)
```

4. **Initialize the program**:

```python
forest = Forest.random_generate(
   pop_size=int(5000),
   gp_len=128,
   input_len=2,
   output_len=1,
   **generate_configs,
)
algorithm.initialize(forest)
fitness = problem.evaluate(forest)
```

5. **Run the program**:

```python
for i in range(10):
    forest = algorithm.step(fitness, args_check=False)
    fitness = problem.evaluate(forest, execute_code=4, args_check=False)
    print(f"step: {i}, max_fitness: {fitness.max()}")

```
You will obtain the result like below:

```
step: 0, max_fitness: -0.11724221706390381
step: 1, max_fitness: -0.10351374745368958
step: 2, max_fitness: -0.07007355988025665
step: 3, max_fitness: -0.07007355242967606
step: 4, max_fitness: -0.07007355242967606
step: 5, max_fitness: -0.07007355242967606
step: 6, max_fitness: -0.07007355242967606
step: 7, max_fitness: -0.07007354497909546
step: 8, max_fitness: -0.07007354497909546
step: 9, max_fitness: -0.036281898617744446
step: 10, max_fitness: -0.036281898617744446
```
6. **Visualize the best individual**:
```python
forest[int(fitness.argmax())].to_png("sr_tree.png")
print(forest[int(fitness.argmax())].to_infix())
```
<div style="text-align: center;">
    <img src="./imgs/sr_tree.png" alt="Visualization of the individual"  width="300" height="300">
</div>

```
((1.40--0.31)/(-0.31+((1.40--1.22)/((1.40--1.22)/1.40))))
```



## Advanced Genetic Operations

### Selection

1. **DefaultSelection**: In the default selection strategy, a certain proportion of elite individuals (those ranked at the top of the population by fitness) are preserved first. Then, individuals ranked at the bottom of the population are eliminated based on the survival rate.

   ```python
   DefaultSelection(survival_rate=0.3, elite_rate=0.01)
   ```

2. **RouletteSelection**: Each individual is selected with a probability proportional to its fitness, ensuring that individuals with higher fitness are more likely to be chosen for the next generation.

   ```python
   DefaultSelection(survival_rate=0.3, elite_rate=0.01)
   ```

3. **TruncationSelection**: All individuals are sorted by their fitness, and a certain proportion of low-fitness individuals are excluded. The next generation is then created by randomly sampling from the remaining individuals with replacement and equal probability.

   ```python
   TruncationSelection(survival_rate=0.3, elite_rate=0.01)
   ```

4. **RankSelection**: Individuals are sorted by fitness, and their selection probabilities are calculated using the following formula:
   $$
   P(R_i) = \frac{1}{n} \left( 1 + sp \left(1 - \frac{2i}{n-1}\right) \right) \quad \text{for } 0 \leq i \leq n-1, \quad 0 \leq sp \leq 1
   $$
   where $n$ is the population size, $i$ is the individual's rank, and $sp$ represents the selection pressure (higher values correspond to greater pressure). Individuals are then selected with replacement based on these probabilities.

   ```python
   RankSelection(survival_rate=0.3, elite_rate=0.01, selection_pressure=0.5)
   ```

5. **TournamentSelection**: A specified number of individuals are randomly selected from the population to compete in a tournament. The winner is chosen based on a probability parameter favoring the best-performing individual. Both with-replacement and without-replacement sampling modes are supported. For without-replacement sampling, individuals with fewer prior selections are preferred in forming tournaments.

   ```python
   TournamentSelection(
       survival_rate=0.3,
       elite_rate=0.01,
       tournament_size=20,
       best_probability=0.9,
       replace=False,
   )
   ```

---

### Crossover

1. **DefaultCrossover**: A random subtree of the recipient is replaced with a random subtree from the donor. Both the recipient and donor are chosen randomly.

   ```python
   DefaultCrossover()
   ```

2. **DiversityCrossover**: Similar to `DefaultCrossover`, but the recipient and donor can be selected using specific selection operators from the surviving individuals. Additionally, a `crossover_rate` parameter allows a certain proportion of individuals to bypass the crossover process.

   ```python
   DiversityCrossover(
       crossover_rate=0.9,
       recipient_selector=RouletteSelector,
       donor_selector=RankSelector(selection_pressure=0.8),
   )
   ```

3. **LeafBiasedCrossover**: Builds on `DiversityCrossover` by introducing a `leaf_bias` parameter. With a specified probability, crossover is restricted to exchanges between leaf nodes, ensuring more stable crossover operations.

   ```python
   DiversityCrossover(
       crossover_rate=0.9,
       recipient_selector=RouletteSelector,
       donor_selector=RankSelector(selection_pressure=0.8),
       leaf_bias=0.3,
   )
   ```

---

### Mutation

1. **DefaultMutation**: A randomly selected subtree is replaced with a newly generated random subtree.

   ```python
   DefaultMutation(mutation_rate=0.2, generate_configs=generate_configs)
   ```

2. **HoistMutation**: A subtree of a GP individual is randomly selected, and then a subtree within it is further selected and moved to replace the original subtree's root. This operation helps mitigate excessive growth (bloating) in GP individuals.

   ```python
   HoistMutation(mutation_rate=0.2)
   ```

3. **SinglePointMutation**: A random node is selected and replaced with a new node of the same type, selected randomly from the node pool.

   ```python
   SinglePointMutation(mutation_rate=0.2)
   ```

4. **MultiPointMutation**: A specific number of nodes (calculated as `mutation_intensity × tree_size`) are randomly selected within an individual, and each undergoes `SinglePointMutation`.

   ```python
   MultiPointMutation(mutation_rate=0.2, mutation_intensity=0.3)
   ```

5. **InsertMutation**: A random basic operator is inserted into the individual.

   ```python
   InsertMutation(mutation_rate=0.2, generate_configs=generate_configs)
   ```

6. **DeleteMutation**: A random basic operator is removed from the individual.

   ```python
   DeleteMutation(mutation_rate=0.2)
   ```

7. **SingleConstMutation**: A random constant node is selected and modified to a new constant value.

   ```python
   SingleConstMutation(mutation_rate=0.2)
   ```

8. **MultiConstMutation**: A specific number of constant nodes (calculated as `mutation_intensity × tree_size`) are randomly selected and modified to new constant values.

   ```python
   MultiConstMutation(mutation_rate=0.2, mutation_intensity=0.3)
   ```

9. **CombinedMutation**: Combines multiple mutation strategies into a single comprehensive mutation operation.

   ```python
   CombinedMutation(
       [
           DefaultMutation(mutation_rate=0.2, generate_configs=generate_configs),
           HoistMutation(mutation_rate=0.2),
           MultiPointMutation(mutation_rate=0.2, mutation_intensity=0.3),
           InsertMutation(mutation_rate=0.2, generate_configs=generate_configs),
       ]
   )
   ```

## Supported Tasks

### Symbolic Regression

Symbolic regression is a powerful method for modeling data relationships using mathematical expressions. In symbolic regression tasks, EvoGP provides flexible and robust functionalities. Users can define custom functions according to their needs or pass their own datasets using `datapoints` and `label` to meet personalized modeling requirements. Below is an example to set up the symbolic regression problem, where the target function is a mathematical combination of two inputs:

```python
from evogp.problem import SymbolicRegression

def func(x):
    val = x[0] ** 4 / (x[0] ** 4 + 1) + x[1] ** 4 / (x[1] ** 4 + 1)
    return val.reshape(-1)

problem = SymbolicRegression(func=func, num_inputs=2, num_data=20000, lower_bounds=-5, upper_bounds=5)
```

------

### Classification

Classification tasks aim to categorize data points into different classes. EvoGP integrates the sklearn library and includes four classic classification tasks: Iris, Wine, Breast Cancer, and Digits. For users with specific requirements, EvoGP supports custom datasets by allowing them to pass their own `datapoints` and `label`. EvoGP offers two modes of operation: single-output and multi-output. In the single-output mode, EvoGP directly outputs classification results, with fitness corresponding to the number of correctly classified instances. In the multi-output mode, EvoGP outputs the probabilities for each class, selects the class with the highest probability as the result, and evaluates fitness based on log loss. Below is an example to set up the classification problem:

```python
from evogp.problem import Classification
problem = Classification(multi_output=False, dataset="iris")
```

------

### Transformation

The transformation task aims to generate new features from raw data to enhance the performance of subsequent models. EvoGP provides support for sklearn datasets, including a built-in example with the Diabetes dataset, allowing users to get started quickly. For custom requirements, users can pass their own datasets using `datapoints` and `label`. During execution, EvoGP automatically generates features most linearly correlated with the `label` based on the input data. The quality of the generated features is evaluated using the Pearson correlation coefficient, which serves as the basis for fitness optimization. Lastly, users can leverage the `new_feature` interface to generate the new features. Below is an example to set up the tranformation problem:

```python
from evogp.problem import Transformation
problem = Transformation(dataset="diabetes")
```

------

### RL Tasks

The RL (Reinforcement Learning) task focuses on training agents to maximize rewards through interactions with an environment. EvoGP integrates with Google's Brax library and includes built-in examples of MuJoCo tasks, such as Swimmer, HalfCheetah, and Hopper, enabling users to quickly set up and experiment with RL problems. Below is an example of how to set up the RL task problem:

```python
from evogp.problem import BraxProblem
problem = BraxProblem("swimmer", max_episode_length=1000)
```

------

Detailed examples for the above tasks are available in the [**examples folder**](https://github.com/EMI-Group/evogp/tree/main/example). Users can refer to these examples to quickly explore EvoGP's capabilities and performance in real-world applications.

## Future Work

1. Improve EvoGP documentation and tutorials.
2. Implement more GP-related algorithms, such as LGP, CGP, GEP.
3. Add more multi-output methods for EvoGP.
4. Further optimize EvoGP to increase computation speed and reduce memory usage.

We warmly welcome community developers to contribute to EvoGP and look forward to your pull requests!


## Community & Support

- Engage in discussions and share your experiences on [GitHub Issues](https://github.com/EMI-Group/evogp/issues).
- Join our QQ group (ID: 297969717).


## Acknowledgements

1. Thanks to John R. Koza for the [genetic programming (GP) algorithm](https://www.genetic-programming.com/), which provided an excellent automatic programming technique and laid the foundation for the development of EvoGP.
2. Thanks to [PyTorch](https://pytorch.org/) and [CUDA](https://developer.nvidia.com/cuda-toolkit) for providing flexible and efficient GPU-accelerated tools, which are essential for optimizing the performance of EvoGP.
3. Thanks to the following projects for their valuable contributions to GP research, which provided inspiration and guidance for EvoGP's design: [DEAP](https://github.com/DEAP/deap), [gplearn](https://github.com/trevorstephens/gplearn), [Karoo GP](https://github.com/kstaats/karoo_gp), [TensorGP](https://github.com/cdvetal/TensorGP) and [SymbolicRegressionGPU](https://github.com/RayZhhh/SymbolicRegressionGPU).
4. Thanks to [scikit-learn](https://github.com/scikit-learn/scikit-learn) and [Brax](https://github.com/google/brax) for their benchmarking frameworks, which have helped validate the performance improvements in EvoGP.
5. Thanks to [EvoX](https://github.com/EMI-Group/evox) for providing a flexible framework that allows EvoGP to integrate with other evolutionary algorithms, expanding its potential.



## Citing EvoGP

If you use EvoGP in your research and want to cite it in your work, please use:
```

```
