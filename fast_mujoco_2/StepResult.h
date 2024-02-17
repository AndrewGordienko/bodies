#ifndef STEP_RESULT_H
#define STEP_RESULT_H

#include <algorithm> // for std::copy

struct StepResult {
    bool done;
    double* observation;
    int observation_length;
    double* rewards;
    int rewards_length;

    // Constructors and destructors for managing dynamic memory
    StepResult() : done(false), observation(nullptr), observation_length(0), rewards(nullptr), rewards_length(0) {}

    ~StepResult() {
        delete[] observation;
        delete[] rewards;
    }

    // Copy constructor (deep copy)
    StepResult(const StepResult& other)
    : done(other.done), observation_length(other.observation_length), rewards_length(other.rewards_length) {
        if (other.observation != nullptr) {
            observation = new double[observation_length];
            std::copy(other.observation, other.observation + observation_length, observation);
        } else {
            observation = nullptr;
        }
        if (other.rewards != nullptr) {
            rewards = new double[rewards_length];
            std::copy(other.rewards, other.rewards + rewards_length, rewards);
        } else {
            rewards = nullptr;
        }
    }

    // Assignment operator (deep copy)
    StepResult& operator=(const StepResult& other) {
        if (this != &other) { // protect against self-assignment
            done = other.done;
            observation_length = other.observation_length;
            rewards_length = other.rewards_length;
            
            delete[] observation; // free existing resources
            delete[] rewards;

            if (other.observation != nullptr) {
                observation = new double[observation_length];
                std::copy(other.observation, other.observation + observation_length, observation);
            } else {
                observation = nullptr;
            }

            if (other.rewards != nullptr) {
                rewards = new double[rewards_length];
                std::copy(other.rewards, other.rewards + rewards_length, rewards);
            } else {
                rewards = nullptr;
            }
        }
        return *this;
    }
};

#endif // STEP_RESULT_H
