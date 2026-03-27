import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


extra_compile_args = {"cxx": ["/O2", "/std:c++17"]} if os.name == "nt" else {"cxx": ["-O3", "-std=c++17"]}


setup(
    name="hyperedge_prediction_native",
    ext_modules=[
        CppExtension(
            name="subgraph_sampler_native",
            sources=["subgraph_sampler_native.cpp"],
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
