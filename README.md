# EvoGP Torch Version

## Install Pytorch and Related CUDA tools

test you're success: `python test_torch_env.py`

## Build CUDA kernels

`python setup.py build_ext --inplace`

test you're success: `python test_bind_success.py`

## Run GP Demo
`python main.py`