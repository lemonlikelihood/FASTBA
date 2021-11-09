//
// Created by lemon on 2021/2/5.
//

#ifndef FASTBA_PARAMETER_H
#define FASTBA_PARAMETER_H

#include "eigen_ba.h"
#include "local_parameterization.h"
#include "projection_factor.h"
namespace fast_ba {
struct ParamBlock {
    ParamBlock(double *_ptr, int _size, LocalParameterization *_p) {
        param_ptr = _ptr;
        size = _size;
        parameterization = _p;
        param_new.resize(_size);
        for (size_t i = 0; i < size; ++i)
            param_new[i] = param_ptr[i];
    }

    ParamBlock(double *_ptr, int _size) {
        param_ptr = _ptr;
        size = _size;
        parameterization = nullptr;
        param_new.resize(_size);
        for (size_t i = 0; i < size; ++i)
            param_new[i] = param_ptr[i];
    }

    void set_constant() {
        constant = true;
        for (size_t i = 0; i < size; ++i)
            param_new[i] = param_ptr[i];
    }

    void plus_delta(double *param_ptr_delta) {
        double *param_ptr_new = param_new.data();
        if (parameterization) {
            parameterization->Plus(param_ptr, param_ptr_delta, param_ptr_new);
        } else {
            for (size_t i = 0; i < size; ++i)
                param_ptr_new[i] = param_ptr[i] + param_ptr_delta[i];
        }
    }

    void update() {
        for (size_t i = 0; i < size; ++i)
            param_ptr[i] = param_new[i];
    }

    void gradient_decrease() {
        for (size_t i = 0; i < size; ++i)
            param_new[i] = param_ptr[i] - param_new[i];
    }

    double step_square_norm() {
        double sn = 0;
        for (size_t i = 0; i < size; ++i)
            sn += (param_new[i] - param_ptr[i]) * (param_new[i] - param_ptr[i]);
        return sn;
    }

    double x_square_norm() {
        double sn = 0;
        for (size_t i = 0; i < size; ++i)
            sn += param_ptr[i] * param_ptr[i];
        return sn;
    }

    int size;
    double *param_ptr;             // x y z w
    std::vector<double> param_new; // x y z w
    LocalParameterization *parameterization;
    bool constant;
};

class ResidualBlock {
public:
    ResidualBlock() {};

    ~ResidualBlock() {};

    ResidualBlock(int residual_id_, int camera_id_, int point_id_)
        : residual_id(residual_id_), camera_id(camera_id_), point_id(point_id_) {
        // factor = std::make_unique<ClassicalProjectFactor>();
        jacobian_ptr = {jac_camera_factor.data(), jac_point_factor.data()};
        factor_ptr = {
            ftf_factor.data(), ftb_factor.data(), etf_factor.data(), ete_factor.data(),
            etb_factor.data()};
    };

    void init(PROJECTION_TYPE &projection_type, Eigen::Vector2d &measurement) {
        jacobian_ptr = {jac_camera_factor.data(), jac_point_factor.data()};
        factor_ptr = {
            ftf_factor.data(), ftb_factor.data(), etf_factor.data(), ete_factor.data(),
            etb_factor.data()};
        if (projection_type == PROJECTION_TYPE::CLASSICAL_PROJECTION)
            factor = std::make_unique<ClassicalProjectFactor>(measurement);
        else if (projection_type == PROJECTION_TYPE::SYM_PROJECTION) {
            factor = std::make_unique<SymProjectFactor>(measurement);
        }
    }

    int residual_id;
    int camera_id;
    int point_id;
    //        Eigen::Vector2d measurement;

    Jac_camera_factor jac_camera_factor;
    Jac_point_factor jac_point_factor;
    Residual_factor residual_factor;

    Ftf_factor ftf_factor;
    Ftb_factor ftb_factor;
    Etf_factor etf_factor;
    Ete_factor ete_factor;
    Etb_factor etb_factor;

    std::array<double *, 2> jacobian_ptr;
    std::array<double *, 5> factor_ptr;

    std::vector<double *> param_blocks;
    std::vector<double *> param_blocks_candidate;
    //        CostFunction *cost_fun;
    std::unique_ptr<ProjectionFactor> factor;

    void evaluate() {
        //            factor
        factor->Evaluate(
            param_blocks.data(), residual_factor.data(), jacobian_ptr.data(), factor_ptr.data());
    }

    void evaluate_candidate() {
        factor->Evaluate(
            param_blocks_candidate.data(), residual_factor.data(), jacobian_ptr.data(),
            factor_ptr.data());
    }

    void add_paramblock(ParamBlock &pb) {
        param_blocks.emplace_back(pb.param_ptr);
        param_blocks_candidate.emplace_back(pb.param_new.data());
    }
};

class X_f {
    int camera_id;
    std::vector<int> residual_ids;
    std::vector<double *> param_blocks;
    std::vector<double *> param_blocks_candidate;
};

class X_e {
    int track_id;
    std::vector<int> residual_ids;
    std::vector<double *> param_blocks;
    std::vector<double *> param_blocks_candidate;
};
} // namespace fast_ba
#endif //FASTBA_PARAMETER_H
