#include "Environment.h"
#include <mujoco.h>
#include <Eigen/Dense>

// Constructor
Environment::Environment(CustomAntEnv* customEnv, const std::string& modelFilePath, const std::vector<std::vector<int>>& legInfo, int maxSteps, int numCreatures)
: env(customEnv), modelFilePath(modelFilePath), legInfo(legInfo), maxSteps(maxSteps), numCreatures(numCreatures) {
    if (!env || !env->getWindow()) {
        throw std::runtime_error("CustomAntEnv instance or GLFW window not found.");
    }
    initialize();
}


StepResult Environment::stepHelper(double* data, int rows, int cols) {
    Eigen::Map<Eigen::MatrixXd> actions(data, rows, cols);
    return this->step(actions);
}

// Destructor
Environment::~Environment() {
    if (env != nullptr) {
        delete env;
        env = nullptr;
    }
    if (d != nullptr) {
        mj_deleteData(d);
        d = nullptr;
    }
    if (m != nullptr) {
        mj_deleteModel(m);
        m = nullptr;
    }
    if (window != nullptr) {
        glfwDestroyWindow(window);
        glfwTerminate();
    }
}


// // Initialize method
// void Environment::initialize() {
//     char error[1000];
//     m = mj_loadXML(modelFilePath.c_str(), nullptr, error, 1000);
//     if (!m) {
//         throw std::runtime_error("Load model error: " + std::string(error));
//     }

//     d = mj_makeData(m);

//     if (!glfwInit()) {
//         mju_error("Could not initialize GLFW");
//     }

//     window = glfwCreateWindow(1200, 900, "MuJoCo Environment", nullptr, nullptr);
//     if (!window) {
//         glfwTerminate();
//         throw std::runtime_error("Could not create GLFW window");
//     }
//     glfwMakeContextCurrent(window);
//     glfwSwapInterval(1);

//     mjv_defaultCamera(&cam);
//     mjv_defaultOption(&opt);
//     mjv_defaultScene(&scn);
//     mjr_defaultContext(&con);

//     mjv_makeScene(m, &scn, 2000);
//     mjr_makeContext(m, &con, mjFONTSCALE_150);

//     env = new CustomAntEnv(modelFilePath, legInfo, maxSteps, numCreatures);
// }



void Environment::initialize() {
    char error[1000];
    m = mj_loadXML(modelFilePath.c_str(), nullptr, error, 1000);
    if (!m) {
        throw std::runtime_error("Load model error: " + std::string(error));
    }

    d = mj_makeData(m);

    // Use CustomAntEnv's window
    if (env) {
        window = env->getWindow();
    } else {
        throw std::runtime_error("CustomAntEnv instance not found.");
    }

    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);

    mjv_makeScene(m, &scn, 2000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);
}


// Corrected Step method to match Environment.h
StepResult Environment::step(const Eigen::MatrixXd& actions) {
    if (env == nullptr) {
        throw std::runtime_error("Environment not properly initialized.");
    }
    return env->step(actions);
}

// Reset method
void Environment::reset() {
    if (env == nullptr) {
        throw std::runtime_error("Environment not properly initialized.");
    }
    env->reset();
}

void Environment::render_environment() {
    env->render_environment();
}

// getActionSize method
int Environment::getActionSize() const {
    return env->getActionSize();
}

int Environment::getNumRewards() const {
    return env->getNumRewards();
}

void Environment::getRewards(double* outRewards) const {
    env->getRewards(outRewards);
}

void Environment::loadNewModel(const std::string& xml_file) {
    if (this->env) {
        this->env->loadNewModel(xml_file);
    }
}

// int getHitCounter() const {
//         return env->hitCounter;
// }