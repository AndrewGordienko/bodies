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
    Eigen::MatrixXd getObservation();
    Eigen::VectorXd calculateReward();
    bool isDone();
    int getNumRewards() const;
    void getRewards(double* outRewards) const;
    double* getObservationData();
    int getObservationSize() const;
    int getRewardsSize() const;
    double* getRewardsData() const;
    void loadNewModel(const std::string& xml_file);
    GLFWwindow* getWindow() const;

    int getHitCounter() const;
    int hitCounter; // Initialize the hit counter

    // Change the method to work with matrices
    Eigen::VectorXd getCreatureState(int creatureIdx);

    static const int ACTION_DIMS = 12;
    static const int CREATURE_STATE_SIZE = 342;

    static constexpr int MAX_LEGS = 4;
    static constexpr int MAX_PARTS_PER_LEG = 3;
    static constexpr int CONTROLS_PER_PART = 1;

    static constexpr int DATA_POINTS_PER_SUBPART = 3;
    static constexpr int DISTANCE_TO_TARGET_DIMS = 2;

    Eigen::Vector2d calculateDistanceToTarget(int creatureIdx);
    Eigen::Vector3d getTorsoPosition(int creatureIdx);
    int calculatePhysicsIndex(int creatureIdx, int legIdx, int partIdx);
    void initializeFlagPositionsFromXML();

private:
    void initializeGLFW();
    void initializeMuJoCo(const std::string& xml_file);
    void setupCallbacks();
    void deinitialize();
    void setControls(const Eigen::MatrixXd& actions);

    Eigen::Vector3d _get_torso_position(int creature_id);
    int calculateControlIndex(int creatureIdx, int legIdx, int partIdx);
    std::vector<double> parsePosition(const char* posAttr);

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
    Eigen::VectorXd rewards;
    Eigen::MatrixXd observation; // Use Eigen::MatrixXd for observation

    std::vector<std::vector<Eigen::Vector3d>> intermediate_targets;
    Eigen::MatrixXd flag_positions; 
    void calculateIntermediateTargets();
    std::pair<int, double> getClosestTargetIndexAndDistance(int creature_id, const Eigen::Vector3d& position);


};

#endif // CUSTOM_ANT_ENV_H
