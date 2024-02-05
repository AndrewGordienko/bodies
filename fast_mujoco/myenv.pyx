from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector
from libcpp.string cimport string
import numpy as np
cimport numpy as cnp

# Initialize NumPy API
cnp.import_array()

cdef extern from "StepResult.h":
    struct StepResult:
        bint done
        double reward

cdef extern from "Environment.h" nogil:
    cdef cppclass Environment:
        Environment(string modelFilePath, const vector[vector[int]]& legInfo, int maxSteps, int numCreatures)
        void reset()
        StepResult stepHelper(double* data, int rows, int cols)
        int getActionSize()
        void render_environment()

cdef class PyEnvironment:
    cdef Environment* thisptr

    def __cinit__(self, model_file_path: str, leg_info: list, int max_steps=1000, int num_creatures=9):
        cdef vector[vector[int]] leg_info_cpp = vector[vector[int]]()
        for li in leg_info:
            temp_vec = vector[int]()
            for item in li:
                temp_vec.push_back(int(item))
            leg_info_cpp.push_back(temp_vec)
        self.thisptr = new Environment(model_file_path.encode('utf-8'), leg_info_cpp, max_steps, num_creatures)

    def __dealloc__(self):
        if self.thisptr:
            del self.thisptr

    def reset(self):
        self.thisptr.reset()

    def step(self, actions_in):
        # Ensure actions_in is a NumPy array of type np.float64 and is contiguous
        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] c_actions_in = np.ascontiguousarray(actions_in, dtype=np.float64)
        cdef int rows = c_actions_in.shape[0]
        cdef int cols = c_actions_in.shape[1]
        # Use a memoryview to obtain a pointer to the data
        cdef double* actions_ptr = &c_actions_in[0, 0]
        cdef StepResult result = self.thisptr.stepHelper(actions_ptr, rows, cols)
        return result.done, None, result.reward

    def getActionSize(self):
        return self.thisptr.getActionSize()
    
    def render_environment(self):
        self.thisptr.render_environment()
