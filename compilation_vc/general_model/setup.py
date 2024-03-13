from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

import os

os.environ["CC"] = "/usr/bin/clang"
os.environ["CXX"] = "/usr/bin/clang++"

# GLFW, Eigen, and tinyxml2 include and lib directories
GLFW_LIB_DIR = '/opt/homebrew/opt/glfw/lib'
EIGEN_INCLUDE_DIR = '/opt/homebrew/opt/eigen/include/eigen3'
TINYXML2_LIB_DIR = '/opt/homebrew/lib'  # Assuming default Homebrew installation paths
TINYXML2_INCLUDE_DIR = '/opt/homebrew/include'

extensions = [
    Extension(
        name="myenv",
        # sources=["myenv.pyx", 'GLFWWindowManager.cpp', "CustomAntEnv.cpp", "not_environment.cpp"],
        sources=["myenv.pyx", "CustomAntEnv.cpp"],
        include_dirs=[
            np.get_include(),  # NumPy headers
            '/Users/anooprehman/Documents/uoft/extracurricular/design_teams/utmist2/bodies/venv/lib/python3.9/site-packages/mujoco/include',  # MuJoCo headers
            # '/Applications/mujoco.framework/Versions/A/Headers',  # MuJoCo headers
            EIGEN_INCLUDE_DIR,  # Eigen headers
            '/usr/local/include',  # Additional dependencies
            TINYXML2_INCLUDE_DIR,  # tinyxml2 headers
        ],
        library_dirs=[
            GLFW_LIB_DIR,  # GLFW lib directory
            TINYXML2_LIB_DIR,  # tinyxml2 lib directory
        ],
        libraries=[
            'glfw',  # GLFW library
            'tinyxml2',  # tinyxml2 library
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
