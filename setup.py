import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

cpu_count = os.cpu_count()
if cpu_count:
    os.environ["MAX_JOBS"] = str(cpu_count)
else:
    os.environ["MAX_JOBS"] = "1"

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="evogp",
    version="0.1.0",
    author="Lishuang Wang, Zhihong Wu, Kebin Sun",
    author_email="zhihong2718@gmai.com",
    description="Evolutionary Genetic Programming with CUDA-based GPU acceleration.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EMI-Group/evogp",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=[
        CUDAExtension(
            name="evogp_cuda",
            sources=[
                "./src/evogp/cuda/torch_wrapper.cu",
                "./src/evogp/cuda/generate.cu",
                "./src/evogp/cuda/mutation.cu",
                "./src/evogp/cuda/forward.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--expt-relaxed-constexpr",
                    "--ptxas-options=-v",
                    "-Xptxas=-O3",
                    "-lineinfo",
                    "-lcudart",
                    "-use_fast_math",
                    "-maxrregcount=32",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(parallel=True)},
    # options={"build_ext": {"build_lib": "./src/evogp"}},
    install_requires=[
        "torch",
        "numpy",
    ],
    include_package_data=True,
    python_requires=">=3.12",
)
