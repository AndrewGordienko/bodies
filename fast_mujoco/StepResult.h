#ifndef STEP_RESULT_H
#define STEP_RESULT_H

#include <Eigen/Dense>

struct StepResult {
    bool done;
    Eigen::VectorXd observation;
    double reward;
};

#endif // STEP_RESULT_H
