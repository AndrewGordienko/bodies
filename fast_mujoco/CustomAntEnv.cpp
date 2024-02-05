#include "CustomAntEnv.h"
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <cmath>

// GLFW callback functions
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods) {
    if (act == GLFW_PRESS && key == GLFW_KEY_BACKSPACE) {
        CustomAntEnv* env = static_cast<CustomAntEnv*>(glfwGetWindowUserPointer(window));
        if (env) {
            env->reset();
        }
    }
}

void mouse_button(GLFWwindow* window, int button, int act, int mods) {
    CustomAntEnv* env = static_cast<CustomAntEnv*>(glfwGetWindowUserPointer(window));
    if (!env) return;

    // Update button states in env or handle as needed
}

void mouse_move(GLFWwindow* window, double xpos, double ypos) {
    // Handle mouse move if necessary
}

void scroll(GLFWwindow* window, double xoffset, double yoffset) {
    // Handle scroll if necessary
}

// CustomAntEnv class methods
CustomAntEnv::CustomAntEnv(const std::string& xml_file, const std::vector<std::vector<int>>& leg_info, int max_steps, int num_creatures)
    : max_steps(max_steps), step_count(0), num_creatures(num_creatures), leg_info(leg_info) {
    initializeGLFW();
    initializeMuJoCo(xml_file);
    setupCallbacks();
}

CustomAntEnv::~CustomAntEnv() {
    deinitialize();
}

void CustomAntEnv::initializeGLFW() {
    if (!glfwInit()) {
        throw std::runtime_error("Could not initialize GLFW.");
    }

    window = glfwCreateWindow(1200, 900, "MuJoCo Environment", NULL, NULL);
    if (!window) {
        glfwTerminate();
        throw std::runtime_error("Could not create GLFW window.");
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // Set user pointer to allow callbacks to access class instance
    glfwSetWindowUserPointer(window, this);

    // It's crucial to process events, even before the main loop starts
    glfwPollEvents();
}


void CustomAntEnv::initializeMuJoCo(const std::string& xml_file) {
    char error[1000] = "";
    model = mj_loadXML(xml_file.c_str(), nullptr, error, 1000);
    if (!model) {
        std::cerr << "Load model error: " << error << std::endl;
        throw std::runtime_error("Load model error.");
    }

    data = mj_makeData(model);
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);

    // Adjust the camera distance here
    cam.distance = 10.0; // Adjust this value to set how far out you want the camera to start

    mjv_makeScene(model, &scn, 2000);
    mjr_makeContext(model, &con, mjFONTSCALE_150);
}


void CustomAntEnv::setupCallbacks() {
    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);
}

void CustomAntEnv::mainLoop() {
    while (!glfwWindowShouldClose(window)) {
        // Simulation logic, e.g., step simulation
        render_environment();

        // Handle GLFW events
        glfwPollEvents();
    }
}

void CustomAntEnv::render_environment() {
    if (!model || !data) {
        std::cerr << "Simulation components not initialized.\n";
        return;
    }

    // Update scene and render
    mjv_updateScene(model, data, &opt, nullptr, &cam, mjCAT_ALL, &scn);
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    mjrRect viewport = {0, 0, width, height};
    mjr_render(viewport, &scn, &con);

    glfwSwapBuffers(window);
    // Polling GLFW events after rendering ensures the window remains responsive
    glfwPollEvents();
}

void CustomAntEnv::reset() {
    mj_resetData(model, data);
    step_count = 0;
}

void CustomAntEnv::deinitialize() {
    mjv_freeScene(&scn);
    mjr_freeContext(&con);
    if (data) mj_deleteData(data);
    if (model) mj_deleteModel(model);
    if (window) {
        glfwDestroyWindow(window);
        glfwTerminate();
    }
}

// Additional methods (e.g., setControls, getObservation, calculateReward, isDone) should be implemented here.


bool CustomAntEnv::isDone() {
    // Implementation for checking if the simulation is done
    return step_count >= max_steps;
}

StepResult CustomAntEnv::step(const Eigen::MatrixXd& actions) {
    setControls(actions);
    mj_step(model, data);
    step_count++;
    Eigen::VectorXd observation = getObservation();
    double reward = calculateReward();
    bool done = isDone(); // Correct usage of isDone()
    return {done, observation, reward}; // Assuming StepResult can hold these values
}


void CustomAntEnv::setControls(const Eigen::MatrixXd& actions) {
    if (actions.cols() != model->nu || actions.rows() != num_creatures) {
        throw std::runtime_error("Actions matrix dimension mismatch.");
    }

    for (int creatureIdx = 0; creatureIdx < actions.rows(); ++creatureIdx) {
        for (int actionIdx = 0; actionIdx < actions.cols(); ++actionIdx) {
            data->ctrl[creatureIdx * model->nu + actionIdx] = actions(creatureIdx, actionIdx);
        }
    }
}

Eigen::VectorXd CustomAntEnv::getObservation() {
    int obs_size = model->nq + model->nv;
    Eigen::VectorXd observation(obs_size);
    std::memcpy(observation.data(), data->qpos, model->nq * sizeof(mjtNum));
    std::memcpy(observation.data() + model->nq, data->qvel, model->nv * sizeof(mjtNum));
    return observation;
}

double CustomAntEnv::calculateReward() {
    // Implement reward calculation
    return data->qvel[0]; // Example reward based on velocity in the x-direction
}

int CustomAntEnv::getNumCreatures() const {
    return num_creatures;
}

int CustomAntEnv::getActionSize() const {
    return model ? model->nu : 0;
}
