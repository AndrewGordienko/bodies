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
    CustomAntEnv(const std::string& xml_file, const std::vector<std::vector<int>>& leg_info, int max_steps = 1000, int num_creatures = 1);
    ~CustomAntEnv();
    
    void reset();
    void render_environment();
    StepResult step(const Eigen::MatrixXd& actions);
    
    int getNumCreatures() const;
    int getActionSize() const;
    Eigen::VectorXd getObservation();
    Eigen::VectorXd calculateReward();
    bool isDone();
    int getNumRewards() const;
    void getRewards(double* outRewards) const;
    double* getObservationData();
    int getObservationSize() const;
    int getRewardsSize() const;
    double* getRewardsData() const;

    Eigen::VectorXd getCreatureState(int creatureIdx);

    static const int ACTION_DIMS = 12; // Action dimensions per creature
    static const int CREATURE_STATE_SIZE = 342; // Observation size per creature

    // In CustomAntEnv.h, within the CustomAntEnv class or as global constants
    static constexpr int MAX_LEGS = 4;
    static constexpr int MAX_PARTS_PER_LEG = 3;
    static constexpr int CONTROLS_PER_PART = 1; // Adjust based on your model's specifics


private:
    void initializeGLFW();
    void initializeMuJoCo(const std::string& xml_file);
    void setupCallbacks();
    void deinitialize();
    void setControls(const Eigen::VectorXd& actions);
    Eigen::Vector3d _get_torso_position(int creature_id);
    int calculateControlIndex(int creatureIdx, int legIdx, int partIdx);


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
    mjrContext con;
    Eigen::MatrixXd flag_positions;
    Eigen::VectorXd rewards;
    Eigen::VectorXd observation; // Ensures observation vector is defined once
};

#endif // CUSTOM_ANT_ENV_H
