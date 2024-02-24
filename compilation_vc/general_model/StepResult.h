#ifndef STEP_RESULT_H
#define STEP_RESULT_H

#include <algorithm> // for std::copy

struct StepResult {
    static constexpr int ExpectedObservationLength = 342; // 9 creatures * 38 features each

    bool done;
    double* observation;
    int observation_length;
    double* rewards;
    int rewards_length;
    bool ownMemory; // Indicates if StepResult owns the memory and should free it

    // Existing constructors, destructors, and methods unchanged

    // Consider adding a method or check to verify the observation length matches expectations
    bool isObservationShapeCorrect() const {
        return observation_length == ExpectedObservationLength;
    }

    // Constructors
    StepResult() : done(false), observation(nullptr), observation_length(0), rewards(nullptr), rewards_length(0), ownMemory(true) {}

    ~StepResult() {
        if (ownMemory) {
            delete[] observation;
            delete[] rewards;
        }
    }

    // Disable copying to prevent accidental double frees
    StepResult(const StepResult&) = delete;
    StepResult& operator=(const StepResult&) = delete;

    // Move constructor for transferring ownership
    StepResult(StepResult&& other) noexcept
    : done(other.done), observation(other.observation), observation_length(other.observation_length), rewards(other.rewards), rewards_length(other.rewards_length), ownMemory(other.ownMemory) {
        other.ownMemory = false; // Prevent the moved-from object from freeing the memory
    }

    // Move assignment operator
    StepResult& operator=(StepResult&& other) noexcept {
        if (this != &other) {
            // Free existing resources if any
            if (ownMemory) {
                delete[] observation;
                delete[] rewards;
            }

            // Transfer ownership
            done = other.done;
            observation = other.observation;
            observation_length = other.observation_length;
            rewards = other.rewards;
            rewards_length = other.rewards_length;
            ownMemory = other.ownMemory;

            // Prevent the moved-from object from freeing the memory
            other.ownMemory = false;
        }
        return *this;
    }
};

#endif // STEP_RESULT_H
