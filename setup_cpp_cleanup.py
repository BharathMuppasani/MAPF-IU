from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "cpp_cleanup",
        ["cpp_cleanup_module.cpp"],
        cxx_std=17,
        extra_compile_args=["-O3"],
    ),
]

setup(
    name="cpp_cleanup",
    version="0.1",
    description="C++ cleanup operations for MAPF plans (trailing/synchronized WAIT removal)",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
