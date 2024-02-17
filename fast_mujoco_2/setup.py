from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

import os

os.environ["CC"] = "/usr/bin/clang"
os.environ["CXX"] = "/usr/bin/clang++"

# GLFW and Eigen include and lib directories
GLFW_LIB_DIR = '/opt/homebrew/opt/glfw/lib'
EIGEN_INCLUDE_DIR = '/opt/homebrew/opt/eigen/include/eigen3'

extensions = [
    Extension(
        name="myenv",
        sources=["myenv.pyx", 'GLFWWindowManager.cpp', "CustomAntEnv.cpp", "not_environment.cpp"],
        include_dirs=[
            np.get_include(),  # NumPy headers
            '/Applications/mujoco.framework/Versions/A/Headers',  # MuJoCo headers
            EIGEN_INCLUDE_DIR,  # Eigen headers
            '/usr/local/include',  # Additional dependencies
        ],
        library_dirs=[
            GLFW_LIB_DIR,  # GLFW lib directory
        ],
        libraries=[
            'glfw',  # GLFW library
        ],
        extra_compile_args=[
            '-std=c++17', 
            '-fcolor-diagnostics', 
            '-fansi-escape-codes',
        ],
        extra_link_args=[
            '-F/Applications',  # Framework search path for MuJoCo
            '-framework', 'mujoco',  # Link against the MuJoCo framework
            '-Wl,-rpath,/Applications/MuJoCo.app/Contents/Frameworks',  # Runtime path for the framework
            '-L/opt/homebrew/opt/llvm/lib',  # Ensure the linker can find the LLVM provided libc++
            '-L/opt/homebrew/Cellar/llvm/17.0.6_1/lib/c++',  # Add the directory containing libc++.dylib
            '-lc++',  # Link against libc++.dylib
        ],
        language='c++',
    )
]

setup(
    ext_modules=cythonize(extensions, language_level=3),
    include_dirs=[np.get_include()],  # Include NumPy headers
)
