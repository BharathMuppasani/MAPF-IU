# setup_cpp_astar.py
#
# Build script for cpp_astar extension. Requires:
#   pip install pybind11

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "cpp_astar",
        ["cpp_astar_module.cpp"],  # path to the C++ file above
        cxx_std=17,
        extra_compile_args=["-O3"],
    ),
]

setup(
    name="cpp_astar",
    version="0.1",
    description="C++ A* for grid envs",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
