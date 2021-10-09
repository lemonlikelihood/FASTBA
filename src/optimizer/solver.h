//
// Created by lemon on 2021/3/4.
//

#ifndef FASTBA_SOLVER_H
#define FASTBA_SOLVER_H

#include "datatype.h"
#include "problem.h"

namespace fast_ba {
class Solver {
public:
    Solver(Problem *problem_);
    void schur_complement();
    void compute_delta();
    void param_plus_delta();
    void param_update();
    void param_recover();
    void compute_model_cost_change();
    void compute_gradient_norm();
    void compute_cost(double &cost);
    void compute_cost_change();
    void solve_linear_problem();
    void compute_relative_decrease();
    void compute_step_norm_x_norm(double &step_norm, double &x_norm);
    void step_accepted(double step_quality);
    void step_rejected(double step_quality);

    std::vector<Ftf_factor> ftfs;
    std::vector<Ftb_factor> ftbs;
    std::vector<Etf_factor> etfs;
    std::vector<Ete_factor> etes;
    std::vector<Etb_factor> etbs;
    std::vector<Ete_factor> etes_inverse;

    Eigen::MatrixXd lhs;
    Eigen::VectorXd rhs;
    Eigen::VectorXd x_f;
    Eigen::VectorXd x_e;


    Problem *problem;
    double current_cost = 0;
    double candidate_cost = 0;
    double cost_change = 0;
    double model_cost_change = 0;
    double radius = 1.0;
    double max_radius = 1e16;
    double min_radius = 1e-32;
    double relative_decrease = 1e-3;
    double decrease_factor = 2.0;
    double gradient_max_norm = 0;
};
} // namespace fast_ba

#endif //FASTBA_SOLVER_H
