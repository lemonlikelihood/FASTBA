//
// Created by lemon on 2021/1/21.
//

#ifndef FASTBA_PROBLEM_H
#define FASTBA_PROBLEM_H

#include "eigen_ba.h"
#include "map.h"
#include "parameter.h"
#include "projection_factor.h"

namespace fast_ba {

class Problem {

public:
    Problem(Map *map_, PROJECTION_TYPE projection_type_);
    ~Problem();

    void add_camera_param();
    void add_point_param();
    void add_residual_param();

    void print_param_blocks();

    void recover_param();
    void recover_carema_param();
    void recover_point_param();

    bool evaluate();

    bool evaluate_candidate();

    int num_residual;
    int num_camera;
    int num_point;

    std::vector<ResidualBlock> residual_blocks;
    std::vector<ParamBlock> param_blocks;

    std::vector<double *> param_vec;
    // std::vector<std::shared_ptr<double>> param_vec;
    // std::vector<std::vector<double>> param_vec;

    std::vector<std::vector<int>> camera_to_observation_id_vec;
    std::vector<std::vector<int>> point_to_observation_id_vec;

    Map *map;
    PROJECTION_TYPE projection_type;
};
} // namespace fast_ba
#endif //FASTBA_PROBLEM_H
