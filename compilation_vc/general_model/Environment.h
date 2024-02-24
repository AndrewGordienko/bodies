#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include <string>
#include <vector>
#include "CustomAntEnv.h" // Custom environment class for MuJoCo simulation
#include <mujoco.h>
#include <GLFW/glfw3.h>
#include <Eigen/Dense>
#include "StepResult.h"

// Struct to encapsulate step results for easier interfacing

class Environment {
public:
    Environment(CustomAntEnv* customEnv, const std::string& modelFilePath, const std::vector<std::vector<int>>& legInfo, int maxSteps, int numCreatures);

    ~Environment();

    void initialize(); // Initialize the simulation environment
    void render_environment();
    void loadNewModel(const std::string& xml_file);

    // int getHitCounter() const {
    //     return env->hitCounter;
    // }

    // Inside the Environment class in Environment.h
    StepResult stepHelper(double* data, int rows, int cols);

    StepResult step(const Eigen::MatrixXd& actions); // Perform one simulation step with the given actions
    void reset(); // Reset the simulation environment
    int getActionSize() const; // Get the size of the action space

    int getNumRewards() const;
    void getRewards(double* outRewards) const;

    // Forwarding methods to CustomAntEnv
    int getObservationSize() const {
        return env->getObservationSize();
    }

    double* getObservationData() const {
        return env->getObservationData();
    }

    int getRewardsSize() const {
        return env->getRewardsSize();
    }

    double* getRewardsData() const {
        return env->getRewardsData();
    }

private:
    mjModel* m = nullptr; // MuJoCo model
    mjData* d = nullptr; // MuJoCo simulation data
    mjvCamera cam; // Visualization camera
    mjvOption opt; // Visualization options
    mjvScene scn; // Visualization scene
    mjrContext con; // Rendering context

    GLFWwindow* window = nullptr; // GLFW window for rendering

    CustomAntEnv* env = nullptr; // Custom environment instance

    std::string modelFilePath; // Path to the MuJoCo model file
    std::vector<std::vector<int>> legInfo; // Configuration info for the simulation
    int maxSteps; // Maximum number of steps per episode
    int numCreatures; // Number of creatures in the environment
};

#endif // ENVIRONMENT_H
