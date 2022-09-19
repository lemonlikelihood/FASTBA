#pragma once

#include "../../geometry/lie_algebra.h"
#include "../../map/feature.h"
#include "../../map/frame.h"
#include "../../utils/common.h"
#include "../factor.h"
#include <ceres/ceres.h>

/*
https://blog.csdn.net/weixin_41394379/article/details/89975386
*/

class MarginalizationErrorCost : public Factor::FactorCostFunction, public ceres::CostFunction {
public:
    MarginalizationErrorCost(
        const Eigen::MatrixXd &sqrt_inv_cov, const Eigen::VectorXd &info_vec,
        std::vector<Frame *> &&frames)
        : sqrt_inv_cov(sqrt_inv_cov), info_vec(info_vec), frames(std::move(frames)) {
        set_num_residuals((int)this->frames.size() * ES_SIZE);
        pose_0.resize(this->frames.size());
        motion_0.resize(this->frames.size());

        mutable_parameter_block_sizes()->clear();
        for (size_t i = 0; i < this->frames.size(); ++i) {
            mutable_parameter_block_sizes()->push_back(4); // q
            mutable_parameter_block_sizes()->push_back(3); // p
            mutable_parameter_block_sizes()->push_back(3); // v
            mutable_parameter_block_sizes()->push_back(3); // bg
            mutable_parameter_block_sizes()->push_back(3); // ba
            pose_0[i] = this->frames[i]->pose;
            motion_0[i] = this->frames[i]->motion;
        }
        // log_info("[MarginalizationErrorCost create]: sqrt_inv_cov\n{}", sqrt_inv_cov);
        // log_info("[MarginalizationErrorCost create]: info_vec\n{}", info_vec.transpose());
    }

    void update() override {}

    bool Evaluate(
        const double *const *parameters, double *residuals, double **jacobians) const override {
        for (size_t i = 0; i < frames.size(); ++i) {
            Eigen::Map<const Eigen::Quaterniond> q(parameters[5 * i + 0]);
            Eigen::Map<const Eigen::Vector3d> p(parameters[5 * i + 1]);
            Eigen::Map<const Eigen::Vector3d> v(parameters[5 * i + 2]);
            Eigen::Map<const Eigen::Vector3d> bg(parameters[5 * i + 3]);
            Eigen::Map<const Eigen::Vector3d> ba(parameters[5 * i + 4]);

            Eigen::Map<Eigen::Vector3d> dq(&residuals[ES_SIZE * i + ES_Q]);
            Eigen::Map<Eigen::Vector3d> dp(&residuals[ES_SIZE * i + ES_P]);
            Eigen::Map<Eigen::Vector3d> dv(&residuals[ES_SIZE * i + ES_V]);
            Eigen::Map<Eigen::Vector3d> dbg(&residuals[ES_SIZE * i + ES_BG]);
            Eigen::Map<Eigen::Vector3d> dba(&residuals[ES_SIZE * i + ES_BA]);

            dq = logmap(pose_0[i].q.conjugate() * q);
            dp = p - pose_0[i].p;
            dv = v - motion_0[i].v;
            dbg = bg - motion_0[i].bg;
            dba = ba - motion_0[i].ba;
            // log_info("[MarginalizationErrorCost]: dq {}", dq.transpose());
            // log_info("[MarginalizationErrorCost]: dp {}", dp.transpose());
            // log_info("[MarginalizationErrorCost]: dv {}", dv.transpose());
            // log_info("[MarginalizationErrorCost]: dbg {}", dbg.transpose());
            // log_info("[MarginalizationErrorCost]: dba {}", dba.transpose());
        }

        // log_info("[MarginalizationErrorCost]: sqrt_inv_cov \n{}", sqrt_inv_cov);
        // log_info("[MarginalizationErrorCost]: info_vec \n{}", info_vec.transpose());

        if (jacobians) {
            for (size_t i = 0; i < frames.size(); ++i) {
                if (jacobians[5 * i + 0]) {
                    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 4, true>> dr_dq(
                        jacobians[5 * i + 0], frames.size() * ES_SIZE, 4);
                    Eigen::Map<Eigen::Vector3d> dq(&residuals[ES_SIZE * i + ES_Q]);
                    dr_dq.setZero();
                    dr_dq.block<3, 3>(ES_SIZE * i + ES_Q, 0) = right_jacobian_inv(dq);
                    dr_dq = sqrt_inv_cov * dr_dq;
                }
                for (size_t k = 1; k < 5; ++k) {
                    if (jacobians[5 * i + k]) {
                        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3, true>> dr_dk(
                            jacobians[5 * i + k], frames.size() * ES_SIZE, 3);
                        dr_dk.setZero();
                        dr_dk.block<3, 3>(ES_SIZE * i + k * 3, 0).setIdentity();
                        dr_dk = sqrt_inv_cov * dr_dk;
                    }
                }
            }
        }
        Eigen::Map<Eigen::VectorXd> full_residual(residuals, frames.size() * ES_SIZE);
        full_residual = sqrt_inv_cov * full_residual + info_vec;

        return true;
    }


    const std::vector<Frame *> &related_frames() const { return frames; }

private:
    std::vector<Pose> pose_0;
    std::vector<MotionState> motion_0;
    Eigen::MatrixXd sqrt_inv_cov;
    Eigen::VectorXd info_vec;
    std::vector<Frame *> frames;
};