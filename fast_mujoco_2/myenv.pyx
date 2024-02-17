# cython: language_level=3
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from libcpp.vector cimport vector
from libcpp.string cimport string
import numpy as np
cimport numpy as np

np.import_array()

cdef extern from "StepResult.h":
    cdef struct StepResult:
        bint done
        double* observation
        int observation_length
        double* rewards
        int rewards_length

cdef extern from "Environment.h" nogil:
    cdef cppclass Environment:
        Environment(string modelFilePath, const vector[vector[int]]& legInfo, int maxSteps, int numCreatures)
        void reset()
        StepResult stepHelper(double* data, int rows, int cols)
        int getObservationSize() const
        double* getObservationData() const
        int getRewardsSize() const
        double* getRewardsData() const
        int getActionSize() const
        void render_environment()

cdef class PyEnvironment:
    cdef Environment* thisptr

    def __cinit__(self, str model_file_path, list leg_info, int max_steps=1000, int num_creatures=9):
        cdef vector[vector[int]] leg_info_cpp = to_vector_vector_int(leg_info)
        self.thisptr = new Environment(model_file_path.encode('utf-8'), leg_info_cpp, max_steps, num_creatures)

    def __dealloc__(self):
        del self.thisptr

    def reset(self):
        self.thisptr.reset()

    def step(self, np.ndarray[np.float64_t, ndim=2] actions_in):
        cdef int rows = actions_in.shape[0]
        cdef int cols = actions_in.shape[1]
        cdef double* actions_ptr = <double*> actions_in.data

        cdef StepResult result = self.thisptr.stepHelper(actions_ptr, rows, cols)

        # Handling observation data
        cdef np.ndarray observation_array = np.empty(result.observation_length, dtype=np.float64)
        if result.observation_length > 0:
            memcpy(observation_array.data, result.observation, sizeof(double) * result.observation_length)

        # Handling rewards data
        cdef np.ndarray rewards_array = np.empty(result.rewards_length, dtype=np.float64)
        if result.rewards_length > 0:
            memcpy(rewards_array.data, result.rewards, sizeof(double) * result.rewards_length)

        return result.done, observation_array, rewards_array

    def getActionSize(self):
        return self.thisptr.getActionSize()

    def render_environment(self):
        self.thisptr.render_environment()

# Correct the declaration of to_vector_vector_int to use cdef
cdef vector[vector[int]] to_vector_vector_int(list leg_info_py):
    cdef vector[vector[int]] leg_info_cpp = vector[vector[int]]()
    cdef vector[int] temp_vec
    for li in leg_info_py:
        temp_vec = vector[int]()
        for item in li:
            temp_vec.push_back(item)
        leg_info_cpp.push_back(temp_vec)
    return leg_info_cpp

