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

CustomAntEnv::CustomAntEnv(const std::string& xml_file, const std::vector<std::vector<int>>& leg_info, int max_steps, int num_creatures)
    : max_steps(max_steps), step_count(0), num_creatures(num_creatures), leg_info(leg_info) {
    initializeGLFW();
    initializeMuJoCo(xml_file);
    setupCallbacks();

    // Initialize flag positions here
    flag_positions = Eigen::MatrixXd(num_creatures, 3);
    for (int i = 0; i < num_creatures; ++i) {
        // Example: Initialize each flag position for creature i
        flag_positions.row(i) = Eigen::Vector3d(5.0 * i, 0.0, 0.1); // Adjust based on your scenario
    }

    rewards = Eigen::VectorXd::Zero(num_creatures);
    observation = Eigen::VectorXd::Zero(CREATURE_STATE_SIZE * num_creatures);
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
    if (window) { // Check if the window exists before trying to destroy it
        glfwDestroyWindow(window);
        window = nullptr; // Nullify the pointer to prevent double destruction
    }
    if (data) {
        mj_deleteData(data);
        data = nullptr; // Prevent use after free
    }
    if (model) {
        mj_deleteModel(model);
        model = nullptr; // Prevent use after free
    }

    // Free MuJoCo visualization resources
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    glfwTerminate(); // Terminate GLFW after all GLFW resources are freed
}


// Additional methods (e.g., setControls, getObservation, calculateReward, isDone) should be implemented here.

// CustomAntEnv.cpp
int CustomAntEnv::getNumRewards() const {
    return this->rewards.size(); // Assuming 'rewards' is an Eigen::VectorXd
}

void CustomAntEnv::getRewards(double* outRewards) const {
    Eigen::VectorXd::Map(outRewards, this->rewards.size()) = this->rewards;
}

bool CustomAntEnv::isDone() {
    // Implementation for checking if the simulation is done
    return step_count >= max_steps;
}

StepResult CustomAntEnv::step(const Eigen::MatrixXd& actions) {
    setControls(actions);
    mj_step(model, data);
    step_count++;

    Eigen::VectorXd observationVec = getObservation();
    Eigen::VectorXd rewardsVec = calculateReward();

    StepResult result;
    result.done = isDone();
    result.observation_length = observationVec.size();
    result.rewards_length = rewardsVec.size();

    result.observation = new double[result.observation_length];
    result.rewards = new double[result.rewards_length];

    for (int i = 0; i < result.observation_length; ++i) {
        result.observation[i] = observationVec(i);
    }
    for (int i = 0; i < result.rewards_length; ++i) {
        result.rewards[i] = rewardsVec(i);
    }

    return result;
}

int CustomAntEnv::calculateControlIndex(int creatureIdx, int legIdx, int partIdx) {
    // Example logic to calculate control index based on creature configuration
    // This should be adjusted based on how your creatures are structured in the MuJoCo model
    int index = 0; // Initialize with the starting index for this creature
    for (int i = 0; i < legIdx; ++i) {
        index += leg_info[creatureIdx][i] * CONTROLS_PER_PART;
    }
    index += partIdx * CONTROLS_PER_PART;
    return index;
}



const int ACTION_DIMS = 12;
const int MAX_OBSERVATION_SIZE = 342;

// In CustomAntEnv.cpp
void CustomAntEnv::setControls(const Eigen::VectorXd& actions) {
    // Assuming actions are flattened: [creature1_leg1_part1, creature1_leg1_part2, ..., creatureN_legM_partP]
    int actionIndex = 0;
    for (int creatureIdx = 0; creatureIdx < num_creatures; ++creatureIdx) {
        for (int legIdx = 0; legIdx < MAX_LEGS; ++legIdx) {
            for (int partIdx = 0; partIdx < leg_info[creatureIdx][legIdx]; ++partIdx) {
                int controlIndex = calculateControlIndex(creatureIdx, legIdx, partIdx);
                if (controlIndex >= 0 && actionIndex < actions.size()) {
                    data->ctrl[controlIndex] = actions[actionIndex++];
                } else {
                    // Skip this action as there's no corresponding joint
                    actionIndex++;
                }
            }
        }
    }
}



Eigen::VectorXd CustomAntEnv::getObservation() {
    int max_obs_size = 342;  // Maximum observation size
    Eigen::VectorXd observation = Eigen::VectorXd::Zero(max_obs_size);  // Initialize with zeros

    int stateSizePerCreature = 38; // Number of observations per creature

    for (int creatureIdx = 0; creatureIdx < num_creatures; ++creatureIdx) {
        int startIndex = creatureIdx * stateSizePerCreature; // Start index for the current creature's observations

        // Copy qpos data into observation
        int qpos_size = model->nq * sizeof(mjtNum);
        if (qpos_size > stateSizePerCreature) {
            qpos_size = stateSizePerCreature;
        }
        std::memcpy(observation.data() + startIndex, data->qpos, qpos_size);

        // Copy qvel data into observation
        int qvel_size = model->nv * sizeof(mjtNum);
        if (qvel_size > stateSizePerCreature - qpos_size) {
            qvel_size = stateSizePerCreature - qpos_size;
        }
        std::memcpy(observation.data() + startIndex + model->nq, data->qvel, qvel_size);
    }

    return observation;
}





Eigen::VectorXd CustomAntEnv::getCreatureState(int creatureIdx) {
    // Assuming 'observation' is already populated with the current state of the simulation
    // and each creature's state occupies a segment of 38 elements.
    const int stateSizePerCreature = 38;

    // Calculate the start index for the current creature's state within the observation vector.
    int startIndex = creatureIdx * stateSizePerCreature;

    // Extract the state segment for the current creature.
    Eigen::VectorXd creatureState = observation.segment(startIndex, stateSizePerCreature);

    return creatureState;
}





// Implement the _get_torso_position method
Eigen::Vector3d CustomAntEnv::_get_torso_position(int creature_id) {
    // Example implementation, adjust index based on your model structure
    int index = creature_id * 3; // Assuming qpos stores positions in a flat array
    return Eigen::Vector3d(data->qpos[index], data->qpos[index + 1], data->qpos[index + 2]);
}

Eigen::VectorXd CustomAntEnv::calculateReward() {
    Eigen::VectorXd rewards(num_creatures); // Vector to hold rewards for all creatures
    for (int creature_id = 0; creature_id < num_creatures; ++creature_id) {
        // Compute reward for each creature
        Eigen::Vector3d torso_position = _get_torso_position(creature_id);
        Eigen::Vector3d flag_pos = flag_positions.row(creature_id); // Make sure this is correct
        double distance = (torso_position - flag_pos).norm();

        double speed_reward_factor = 1.0;
        double speed_reward = speed_reward_factor / (1 + step_count);

        double energy_used = 0.0;
        for (int i = 0; i < model->nu; ++i) {
            energy_used += std::abs(data->ctrl[creature_id * model->nu + i]);
        }
        double energy_penalty = energy_used * 0.00005;

        double flag_reached_reward = (distance < 0.1) ? 10.0 : 0.0;

        rewards(creature_id) = speed_reward + flag_reached_reward - energy_penalty;

        // Diagnostic print statements
        // std::cout << "Creature " << creature_id << ": Distance = " << distance
        //           << ", Speed Reward = " << speed_reward
        //           << ", Energy Penalty = " << energy_penalty
        //           << ", Flag Reached Reward = " << flag_reached_reward
        //           << ", Total Reward = " << rewards(creature_id) << std::endl;
    }
    return rewards;
}



int CustomAntEnv::getNumCreatures() const {
    return num_creatures;
}

int CustomAntEnv::getActionSize() const {
    // Each creature has ACTION_DIMS actions, and there are num_creatures creatures
    return ACTION_DIMS * num_creatures;
}

double* CustomAntEnv::getObservationData() {
    return observation.data();
}

int CustomAntEnv::getObservationSize() const {
    return observation.size();
}

int CustomAntEnv::getRewardsSize() const {
    return rewards.size();
}

double* CustomAntEnv::getRewardsData() const {
    return const_cast<double*>(rewards.data());
}

