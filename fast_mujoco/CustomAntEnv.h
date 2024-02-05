#ifndef CUSTOM_ANT_ENV_H
#define CUSTOM_ANT_ENV_H

#include <string>
#include <vector>
#include <Eigen/Dense>
#include <mujoco.h>
#include "StepResult.h"
#include <GLFW/glfw3.h>

class CustomAntEnv {
public:
    CustomAntEnv(const std::string& xml_file, const std::vector<std::vector<int>>& leg_info, int max_steps = 2500, int num_creatures = 1);
    ~CustomAntEnv();
    
    void reset();
    void render_environment();
    StepResult step(const Eigen::MatrixXd& actions);
    
    int getNumCreatures() const;
    int getActionSize() const;
    void mainLoop(); // Ensure this is only declared once

private:
    void initializeGLFW();
    void initializeMuJoCo(const std::string& xml_file);
    void deinitialize();
    void setupCallbacks();

    mjModel* model = nullptr;
    mjData* data = nullptr;
    mjvScene scn;
    mjvCamera cam;
    mjvOption opt;
    GLFWwindow* window = nullptr;

    int max_steps;
    int step_count;
    int num_creatures;
    std::vector<std::vector<int>> leg_info;

    mjrContext con; // Declare the MuJoCo rendering context

    // Declare missing methods
    void setControls(const Eigen::MatrixXd& actions);
    Eigen::VectorXd getObservation();
    double calculateReward();
    bool isDone();
};

#endif // CUSTOM_ANT_ENV_H
