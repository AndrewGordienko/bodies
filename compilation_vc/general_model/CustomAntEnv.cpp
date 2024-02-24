#include "CustomAntEnv.h"
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <sstream>
#include "tinyxml2.h"
using namespace tinyxml2; // This allows you to use tinyxml2 classes without the namespace prefix

// Include your XML parser library headers here

void CustomAntEnv::initializeFlagPositionsFromXML() {
    XMLDocument xmlDoc;
    XMLError eResult = xmlDoc.LoadFile("xml_world_episode_0.xml");
    if (eResult != XML_SUCCESS) {
        std::cerr << "Failed to load XML file. Error: " << xmlDoc.ErrorName() << std::endl;
        return;
    }

    // Initialize flag_positions matrix size
    flag_positions = Eigen::MatrixXd(num_creatures, 3);

    // Find all flag elements
    XMLNode* root = xmlDoc.FirstChild();
    if (root == nullptr) return; // Handle error

    int flagIndex = 0;
    for (XMLElement* elem = root->FirstChildElement("geom"); elem != nullptr; elem = elem->NextSiblingElement("geom")) {
        const char* name = elem->Attribute("name");
        if (name != nullptr && std::string(name).find("flag_") == 0) {
            // Parse position attribute
            const char* posAttr = elem->Attribute("pos");
            std::vector<double> posValues = parsePosition(posAttr);

            // Assign to flag_positions matrix
            if (posValues.size() == 3 && flagIndex < num_creatures) {
                flag_positions.row(flagIndex) = Eigen::Vector3d(posValues[0], posValues[1], posValues[2]);
                ++flagIndex;
            }
        }
    }
}

std::vector<double> CustomAntEnv::parsePosition(const char* posAttr) {
    std::vector<double> pos;
    if (posAttr != nullptr) {
        std::stringstream ss(posAttr);
        double val;
        while (ss >> val) {
            pos.push_back(val);
            if (ss.peek() == ' ') ss.ignore();
        }
    }
    return pos;
}

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

// After - Assume hitCounter is declared before step_count in the class definition
CustomAntEnv::CustomAntEnv(const std::string& xml_file, const std::vector<std::vector<int>>& leg_info, int max_steps, int num_creatures)
    : hitCounter(0), max_steps(max_steps), step_count(0), num_creatures(num_creatures), leg_info(leg_info) {
    initializeGLFW();
    initializeMuJoCo(xml_file);
    setupCallbacks();

    // Load flag positions from XML
    initializeFlagPositionsFromXML();

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
    cam.distance = 25.0; // Adjust this value to set how far out you want the camera to start

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
    // Load or reload flag positions from the XML file at the start
    initializeFlagPositionsFromXML();
    
    mj_resetData(model, data);
    step_count = 0;
    
    // Now safe to call calculateIntermediateTargets
    calculateIntermediateTargets();

    // Reset other necessary states
    hitCounter = 0;

    // Debugging print - consider commenting out once confirmed working
    std::cerr << "Reset completed. Flag positions:\n" << flag_positions << std::endl;
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

void CustomAntEnv::loadNewModel(const std::string& xml_file) {
    if (data) {
        mj_deleteData(data);
        data = nullptr;
    }
    if (model) {
        mj_deleteModel(model);
        model = nullptr;
    }

    char error[1000] = "";
    model = mj_loadXML(xml_file.c_str(), nullptr, error, 1000);
    if (!model) {
        std::cerr << "Load model error: " << error << std::endl;
        throw std::runtime_error("Load model error.");
    }

    data = mj_makeData(model);
    reset(); // Reset the environment with the new model
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
    StepResult result; // Declare StepResult instance

    // Assume getObservation now returns a MatrixXd representing a 9x38 matrix
    Eigen::MatrixXd observationMatrix = getObservation();
    Eigen::VectorXd rewardsVec = calculateReward();

    // Flatten the observation matrix to a vector for easy transfer
    Eigen::Map<Eigen::VectorXd> observationVecFlat(observationMatrix.data(), observationMatrix.size());

    if (flag_positions.rows() != num_creatures || flag_positions.cols() != 3) {
        std::cerr << "Flag positions matrix has unexpected dimensions." << std::endl;
        return StepResult(); // Return an error state or handle appropriately
    }

    result.done = isDone();
    result.observation_length = observationVecFlat.size(); // Should be 9 * 38 = 342 for a 9x38 matrix
    result.rewards_length = rewardsVec.size();

    // Allocate memory for flattened observations and rewards
    result.observation = new double[result.observation_length];
    result.rewards = new double[result.rewards_length];

    // Copy data into the allocated arrays
    std::memcpy(result.observation, observationVecFlat.data(), result.observation_length * sizeof(double));
    std::memcpy(result.rewards, rewardsVec.data(), result.rewards_length * sizeof(double));

    for (int i = 0; i < 9; ++i) {
        Eigen::Vector3d creaturePos3D = _get_torso_position(i);
        Eigen::Vector3d flagPos3D = flag_positions.row(i);
        
        // Create 2D vectors by excluding the z-coordinate
        Eigen::Vector2d creaturePos = creaturePos3D.head<2>();
        Eigen::Vector2d flagPos = flagPos3D.head<2>();
        
        // Calculate the horizontal distance between the creature and the flag
        double distance = (creaturePos - flagPos).norm();

        std::cout << "top bread " << std::endl;
        std::cout << "Creature Position: " << creaturePos.transpose() << std::endl;
        std::cout << "Flag Position: " << flagPos.transpose() << std::endl;
        std::cout << "Horizontal Distance: " << distance << std::endl;
        std::cout << "bottom bread " << std::endl;

        // Assuming a hit is counted if the creature is within 0.5 units of the flag horizontally
        if (distance < 0.1) { // Adjust this threshold as necessary
            // std::cout << "hit " << std::endl;
            hitCounter++;
        }
    }

    return result; // Return the populated result
}


int CustomAntEnv::getHitCounter() const {
    return hitCounter;
}

int CustomAntEnv::calculateControlIndex(int creatureIdx, int legIdx, int partIdx) {
    int index = 0; // Initialize with the starting index for this creature

    // First, add offset for all previous creatures
    for (int prevCreature = 0; prevCreature < creatureIdx; ++prevCreature) {
        // Assuming each creature has the same number of motors, which is MAX_LEGS * MAX_PARTS_PER_LEG
        index += MAX_LEGS * MAX_PARTS_PER_LEG * CONTROLS_PER_PART;
    }

    // Then, calculate index for current creature
    for (int i = 0; i < legIdx; ++i) {
        index += leg_info[creatureIdx][i] * CONTROLS_PER_PART;
    }
    index += partIdx * CONTROLS_PER_PART;

    return index;
}




const int ACTION_DIMS = 12;
const int MAX_OBSERVATION_SIZE = 342;

void CustomAntEnv::setControls(const Eigen::MatrixXd& actions) {
    //std::cout << "Setting controls for actions matrix of size: " << actions.rows() << "x" << actions.cols() << std::endl;

    // Assuming actions is a 9x12 Eigen::MatrixXd
    for (int creatureIdx = 0; creatureIdx < num_creatures; ++creatureIdx) {
        //std::cout << "Creature " << creatureIdx << ":" << std::endl;
        for (int legIdx = 0; legIdx < MAX_LEGS; ++legIdx) {
            for (int partIdx = 0; partIdx < MAX_PARTS_PER_LEG; ++partIdx) {
                int actionIndex = legIdx * MAX_PARTS_PER_LEG + partIdx;
                if (actionIndex < actions.cols()) {
                    int controlIndex = calculateControlIndex(creatureIdx, legIdx, partIdx);
                    if (controlIndex >= 0) { // Check if the motor exists
                        double actionValue = actions(creatureIdx, actionIndex);
                        //std::cout << " - Leg " << legIdx << ", Part " << partIdx << ", Action Index: " << actionIndex << ", Control Index: " << controlIndex << ", Action Value: " << actionValue << std::endl;
                        data->ctrl[controlIndex] = actionValue;
                    } else {
                        //std::cout << " - Leg " << legIdx << ", Part " << partIdx << ", Action Index: " << actionIndex << " has no corresponding control index." << std::endl;
                    }
                }
            }
        }
    }
}




Eigen::VectorXd CustomAntEnv::getCreatureState(int creatureIdx) {
    if (creatureIdx < 0 || creatureIdx >= num_creatures) {
        throw std::out_of_range("Creature index out of range");
    }
    // Return the specified row as a vector
    return observation.row(creatureIdx);
}




Eigen::MatrixXd CustomAntEnv::getObservation() {
    // Constants for the observation dimensions
    const int OBSERVATION_DIMS_PER_CREATURE = MAX_LEGS * MAX_PARTS_PER_LEG * DATA_POINTS_PER_SUBPART + DISTANCE_TO_TARGET_DIMS + 3; // Adjust these constants as per your model
    
    // Initialize the observation matrix for all creatures
    Eigen::MatrixXd observations = Eigen::MatrixXd::Zero(num_creatures, OBSERVATION_DIMS_PER_CREATURE);

    for (int creatureIdx = 0; creatureIdx < num_creatures; ++creatureIdx) {
        int observationIndex = 0; // Reset for each creature

        // Process leg and part data
        for (int legIdx = 0; legIdx < MAX_LEGS; ++legIdx) {
            for (int partIdx = 0; partIdx < MAX_PARTS_PER_LEG; ++partIdx) {
                // Placeholder logic for physics index calculation
                int physicsIdx = calculatePhysicsIndex(creatureIdx, legIdx, partIdx); // Implement this function based on your simulation setup

                // Example placeholders for retrieving sensor data
                double angle = data->qpos[physicsIdx];
                double velocity = data->qvel[physicsIdx];
                double acceleration = data->sensordata[physicsIdx]; // Assuming acceleration data is stored similarly
                
                // Populate observations with the retrieved data
                observations(creatureIdx, observationIndex++) = angle;
                observations(creatureIdx, observationIndex++) = velocity;
                observations(creatureIdx, observationIndex++) = acceleration;
            }
        }

        // Adding distance to target data
        Eigen::Vector2d distanceToTarget = calculateDistanceToTarget(creatureIdx); // Implement this function based on your simulation setup
        observations(creatureIdx, observationIndex++) = distanceToTarget.x();
        observations(creatureIdx, observationIndex++) = distanceToTarget.y();
        
        // Gyroscope data integration
        int gyroIndex = 2 * creatureIdx + 1; // Gyroscope data index calculation
        double gyro_x = data->sensordata[gyroIndex * 3];     // Gyro X
        double gyro_y = data->sensordata[gyroIndex * 3 + 1]; // Gyro Y
        double gyro_z = data->sensordata[gyroIndex * 3 + 2]; // Gyro Z

        observations(creatureIdx, observationIndex++) = gyro_x;
        observations(creatureIdx, observationIndex++) = gyro_y;
        observations(creatureIdx, observationIndex++) = gyro_z;
    }
    return observations;
}

int CustomAntEnv::calculatePhysicsIndex(int creatureIdx, int legIdx, int partIdx) {
    // This function needs to calculate the index in qpos and qvel arrays based on creature, leg, and part indices
    // The specific calculation will depend on how your model is structured in MuJoCo
    // For example, this is a placeholder calculation
    return (creatureIdx * MAX_LEGS * MAX_PARTS_PER_LEG) + (legIdx * MAX_PARTS_PER_LEG) + partIdx;
}

Eigen::Vector3d CustomAntEnv::_get_torso_position(int creatureIdx) {
    if (creatureIdx < 0 || creatureIdx >= num_creatures || creatureIdx * 3 + 2 >= model->nq) {
        std::cerr << "Invalid access in _get_torso_position for creature index: " << creatureIdx << std::endl;
        return Eigen::Vector3d::Zero();
    }

    // Assuming each creature's torso position is stored at a certain index in qpos
    int baseIndex = creatureIdx * 3; // Example index, adjust based on your model

    // Validate baseIndex is within qpos bounds
    if (baseIndex < 0 || baseIndex + 2 >= model->nq) {
        std::cerr << "Base index for qpos is out of bounds." << std::endl;
        return Eigen::Vector3d::Zero();
    }

    return Eigen::Vector3d(data->qpos[baseIndex], data->qpos[baseIndex + 1], data->qpos[baseIndex + 2]);
}




// Method to calculate distance to the target
Eigen::Vector2d CustomAntEnv::calculateDistanceToTarget(int creatureIdx) {
    Eigen::Vector3d torsoPosition = _get_torso_position(creatureIdx);
    Eigen::Vector3d flagPosition = flag_positions.row(creatureIdx);
    Eigen::Vector2d distance = (flagPosition.head<2>() - torsoPosition.head<2>());
    return distance;
}


Eigen::VectorXd CustomAntEnv::calculateReward() {
    Eigen::VectorXd rewards = Eigen::VectorXd::Zero(num_creatures);

    for (int creatureIdx = 0; creatureIdx < num_creatures; ++creatureIdx) {
        // Obtain the torso position and the flag position for the current creature
        Eigen::Vector3d torso_position = _get_torso_position(creatureIdx);
        Eigen::Vector3d flag_pos = flag_positions.row(creatureIdx);
        double distanceToFinalFlag = (torso_position - flag_pos).norm();

        // Calculate the speed reward
        double speed_reward_factor = 1.0;
        double speed_reward = speed_reward_factor / (1 + step_count);

        // Calculate the energy penalty
        double energy_used = 0.0;
        for (int i = 0; i < model->nu; ++i) {
            energy_used += std::abs(data->ctrl[creatureIdx * model->nu + i]);
        }
        double energy_penalty = energy_used * 0.0005;

        // Calculate the flag reached reward
        double flag_reached_reward = (distanceToFinalFlag < 0.1) ? 10.0 : 0.0;

        // Calculate the intermediate reward
        auto [closestIndex, distanceToClosestIntermediate] = getClosestTargetIndexAndDistance(creatureIdx, torso_position);
        double intermediate_reward = 0.0;
        if (closestIndex != -1 && distanceToClosestIntermediate < 0.5) {
            intermediate_reward = (closestIndex + 1) * 10.0;
            intermediate_targets[creatureIdx].erase(intermediate_targets[creatureIdx].begin() + closestIndex);
        }

        // Calculate gyroscope-based stability reward
        int gyroIndex = 2 * creatureIdx + 1; // Assuming a method to calculate the correct index for gyroscope data
        double gyro_x = data->sensordata[gyroIndex * 3];
        double gyro_y = data->sensordata[gyroIndex * 3 + 1];
        double gyro_z = data->sensordata[gyroIndex * 3 + 2];
        double gyro_stability_reward = -0.1 * (std::abs(gyro_x) + std::abs(gyro_y) + std::abs(gyro_z));

        // Combine all reward components
        rewards(creatureIdx) = speed_reward + flag_reached_reward - energy_penalty + intermediate_reward + gyro_stability_reward;
    }

    return rewards;
}


void CustomAntEnv::calculateIntermediateTargets() {
    // Clear existing intermediate targets
    intermediate_targets.clear();
    
    // Ensure that flag_positions has been correctly initialized
    if (flag_positions.size() <= 0) {
        std::cerr << "Flag positions not initialized.\n";
        return;
    }

    for (int i = 0; i < num_creatures; ++i) {
        // Make sure to access valid indices within flag_positions
        if (i >= flag_positions.rows()) {
            std::cerr << "Index out of bounds for flag_positions matrix.\n";
            continue;
        }

        Eigen::Vector3d spawn_pos = _get_torso_position(i);
        Eigen::Vector3d target_pos = flag_positions.row(i);

        std::vector<Eigen::Vector3d> creatureTargets;
        for (int j = 1; j <= 5; ++j) {
            double fraction = static_cast<double>(j) / 6.0;
            Eigen::Vector3d intermediateTarget = spawn_pos + fraction * (target_pos - spawn_pos);
            creatureTargets.push_back(intermediateTarget);
        }
        intermediate_targets.push_back(creatureTargets);
    }
}





std::pair<int, double> CustomAntEnv::getClosestTargetIndexAndDistance(int creature_id, const Eigen::Vector3d& position) {
    int closestIndex = -1;
    double closestDistance = std::numeric_limits<double>::max();

    for (size_t i = 0; i < intermediate_targets[creature_id].size(); ++i) {
        double distance = (intermediate_targets[creature_id][i] - position).norm();
        if (distance < closestDistance) {
            closestDistance = distance;
            closestIndex = static_cast<int>(i);
        }
    }
    return {closestIndex, closestDistance};
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

GLFWwindow* CustomAntEnv::getWindow() const {
    return window;
}
