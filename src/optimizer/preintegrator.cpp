/**************************************************************************
* This file is part of PVIO
*
* Copyright (c) ZJU-SenseTime Joint Lab of 3D Vision. All Rights Reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
**************************************************************************/
#include "preintegrator.h"
#include "../../dataset/dataset.h"
#include "../geometry/lie_algebra.h"
#include "../map/frame.h"

void PreIntegrator::reset() {
    delta.t = 0;
    delta.q.setIdentity();
    delta.p.setZero();
    delta.v.setZero();
    delta.cov.setZero();
    delta.sqrt_inv_cov.setZero();

    jacobian.dq_dbg.setZero();
    jacobian.dp_dbg.setZero();
    jacobian.dp_dba.setZero();
    jacobian.dv_dbg.setZero();
    jacobian.dv_dba.setZero();
}

void PreIntegrator::increment(
    double dt, const IMUData &data, const Eigen::Vector3d &bg, const Eigen::Vector3d &ba,
    bool compute_jacobian, bool compute_covariance) {
    // runtime_assert(dt >= 0, "dt cannot be negative.");

    Eigen::Vector3d w = data.w - bg;
    Eigen::Vector3d a = data.a - ba;

    if (compute_covariance) {
        Eigen::Matrix<double, 9, 9> A;
        A.setIdentity();
        A.block<3, 3>(ES_Q, ES_Q) = expmap(w * dt).conjugate().matrix();
        A.block<3, 3>(ES_V, ES_Q) = -dt * delta.q.matrix() * hat(a);
        A.block<3, 3>(ES_P, ES_Q) = -0.5 * dt * dt * delta.q.matrix() * hat(a);
        A.block<3, 3>(ES_P, ES_V) = dt * Eigen::Matrix3d::Identity();

        Eigen::Matrix<double, 9, 6> B;
        B.setZero();
        B.block<3, 3>(ES_Q, ES_BG - ES_BG) = dt * right_jacobian(w * dt);
        B.block<3, 3>(ES_V, ES_BA - ES_BG) = dt * delta.q.matrix();
        B.block<3, 3>(ES_P, ES_BA - ES_BG) = 0.5 * dt * dt * delta.q.matrix();

        Eigen::Matrix<double, 6, 6> white_noise_cov;
        double inv_dt = 1.0 / std::max(dt, 1.0e-7);
        white_noise_cov.setZero();
        white_noise_cov.block<3, 3>(ES_BG - ES_BG, ES_BG - ES_BG) = cov_w * inv_dt;
        white_noise_cov.block<3, 3>(ES_BA - ES_BG, ES_BA - ES_BG) = cov_a * inv_dt;

        delta.cov.block<9, 9>(ES_Q, ES_Q) =
            A * delta.cov.block<9, 9>(0, 0) * A.transpose() + B * white_noise_cov * B.transpose();
        delta.cov.block<3, 3>(ES_BG, ES_BG) += cov_bg * dt;
        delta.cov.block<3, 3>(ES_BA, ES_BA) += cov_ba * dt;
    }

    if (compute_jacobian) {
        jacobian.dp_dbg +=
            dt * jacobian.dv_dbg - 0.5 * dt * dt * delta.q.matrix() * hat(a) * jacobian.dq_dbg;
        jacobian.dp_dba += dt * jacobian.dv_dba - 0.5 * dt * dt * delta.q.matrix();
        jacobian.dv_dbg -= dt * delta.q.matrix() * hat(a) * jacobian.dq_dbg;
        jacobian.dv_dba -= dt * delta.q.matrix();
        jacobian.dq_dbg =
            expmap(w * dt).conjugate().matrix() * jacobian.dq_dbg - right_jacobian(w * dt) * dt;
    }

    delta.t = delta.t + dt;
    delta.p = delta.p + dt * delta.v + 0.5 * dt * dt * (delta.q * a);
    delta.v = delta.v + dt * (delta.q * a);
    delta.q = (delta.q * expmap(w * dt)).normalized();
}

bool PreIntegrator::integrate(
    double t, const Eigen::Vector3d &bg, const Eigen::Vector3d &ba, bool compute_jacobian,
    bool compute_covariance) {
    if (data.size() == 0)
        return false;
    reset();
    for (size_t i = 0; i + 1 < data.size(); ++i) {
        const IMUData &d = data[i];
        // log_info("[integrate]: i: {},t: {},bg: {},ba: {}", i, d.t, bg.transpose(), ba.transpose());
        increment(data[i + 1].t - d.t, d, bg, ba, compute_jacobian, compute_covariance);
    }
    increment(t - data.back().t, data.back(), bg, ba, compute_jacobian, compute_covariance);
    // log_info(
    // "[integrate]: i: {},t: {},bg: {},ba: {}", data.size() - 1, data.back().t, bg.transpose(),
    // ba.transpose());
    if (compute_covariance) {
        compute_sqrt_inv_cov();
    }
    return true;
}

void PreIntegrator::compute_sqrt_inv_cov() {
    delta.sqrt_inv_cov =
        Eigen::LLT<Eigen::Matrix<double, 15, 15>>(delta.cov.inverse()).matrixL().transpose();
}

void PreIntegrator::predict(const Frame *old_frame, Frame *new_frame) {
    static const Eigen::Vector3d gravity = {0, 0, -9.80665};
    new_frame->motion.bg = old_frame->motion.bg;
    new_frame->motion.ba = old_frame->motion.ba;
    new_frame->motion.v = old_frame->motion.v + gravity * delta.t + old_frame->pose.q * delta.v;
    new_frame->pose.p = old_frame->pose.p + 0.5 * gravity * delta.t * delta.t
                        + old_frame->motion.v * delta.t + old_frame->pose.q * delta.p;
    new_frame->pose.q = old_frame->pose.q * delta.q;
}
