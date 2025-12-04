# setup_cpp_joint_astar.py
#
# Build script for cpp_joint_astar extension. Requires:
#   pip install pybind11

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "cpp_joint_astar",
        ["cpp_joint_astar_module.cpp"],
        cxx_std=17,
        extra_compile_args=["-O3"],
    ),
]

setup(
    name="cpp_joint_astar",
    version="0.1",
    description="C++ joint A* for multi-agent pathfinding",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
