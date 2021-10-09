//
// Created by lemon on 2021/1/21.
//

#ifndef FASTBA_PROBLEM_H
#define FASTBA_PROBLEM_H

#include "map.h"
#include "eigen_ba.h"
#include "projection_factor.h"
#include "parameter.h"

namespace fast_ba {
    class Problem {
    public:

        Problem(Map* _map);

        void schur_complement();

        void compute_delta();

        void param_plus_delta();

        void param_update();

        void compute_model_cost_change();

        bool evaluate();

        bool evaluate_candidate();

        void compute_relative_decrease();

        void compute_step_norm_x_norm(double &step_norm, double &x_norm);

        void compute_cost(double &cost);

//    private:

        int num_residual;
        int num_camera;
        int num_point;

//        Eigen::VectorXd x_f;
//        Eigen::VectorXd x_e;
//        std::vector<Jac_camera_block> Jac_camera_factors;
//        std::vector<Jac_point_block> Jac_point_factors;
//        std::vector<Residual_block> residuals;
//        std::vector<Ftf_block> Ftf_factors;
//        std::vector<Ftb_block> Ftb_factors;
//        std::vector<Ete_block> ete_factors;
//        std::vector<Etb_block> etb_factors;
        std::vector<ResidualBlock> residual_blocks;
        std::vector<ParamBlock> param_blocks;

        std::vector<std::vector<int>> camera_to_observation_id_vec;
        std::vector<std::vector<int>> point_to_observation_id_vec;

//        Eigen::MatrixXd lhs;
//        Eigen::VectorXd rhs;
//        Eigen::VectorXd x_f;
//        Eigen::VectorXd x_e;

        Map* map;

    };
}
#endif //FASTBA_PROBLEM_H
