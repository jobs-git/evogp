# EvoGP Torch Version

## Add Current location to PYTHONPATH
Windows Powershell:
```bash
$env:PYTHONPATH = ".\src"
```
to examine: 
```
echo $env:PYTHONPATH
```

Linux: 
```
export PYTHONPATH=./src
```
to examine: 
```
echo $PYTHONPATH
```

## Install Pytorch and Related CUDA tools

test you're success: 
```
python test_torch_env.py
```

## Build CUDA kernels

```
python setup.py build_ext --inplace
```

test you're success: 
```
python test_bind_success.py
```

## Run GP Demo
```
python main.py
```
