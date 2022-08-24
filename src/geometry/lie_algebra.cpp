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
#include "lie_algebra.h"

Eigen::Matrix3d right_jacobian(const Eigen::Vector3d &w) {
    static const double root2_eps = sqrt(std::numeric_limits<double>::epsilon());
    static const double root4_eps = sqrt(root2_eps);
    static const double qdrt720 = sqrt(sqrt(720.0));
    static const double qdrt5040 = sqrt(sqrt(5040.0));
    static const double sqrt24 = sqrt(24.0);
    static const double sqrt120 = sqrt(120.0);

    double angle = w.norm();
    double cangle = cos(angle);
    double sangle = sin(angle);
    double angle2 = angle * angle;

    double cos_term;
    // compute (1-cos(x))/x^2, its taylor expansion around 0 is 1/2-x^2/24+x^4/720+o(x^6)
    if (angle > root4_eps * qdrt720) {
        cos_term = (1 - cangle) / angle2;
    } else { // use taylor expansion to avoid singularity
        cos_term = 0.5;
        if (angle > root2_eps * sqrt24) { // we have to include x^2 term
            cos_term -= angle2 / 24.0;
        }
    }

    double sin_term;
    // compute (x-sin(x))/x^3, its taylor expansion around 0 is 1/6-x^2/120+x^4/5040+o(x^6)
    if (angle > root4_eps * qdrt5040) {
        sin_term = (angle - sangle) / (angle * angle2);
    } else {
        sin_term = 1.0 / 6.0;
        if (angle > root2_eps * sqrt120) { // we have to include x^2 term
            sin_term -= angle2 / 120.0;
        }
    }

    Eigen::Matrix3d hat_w = hat(w);
    return Eigen::Matrix3d::Identity() - cos_term * hat_w + sin_term * hat_w * hat_w;
}

Eigen::Matrix3d right_jacobian_inv(const Eigen::Vector3d &w) {
    // return right_jacobian(w).inverse();
    static const double root2_eps = sqrt(std::numeric_limits<double>::epsilon());
    static const double root4_eps = sqrt(root2_eps);
    static const double qdrt720 = sqrt(sqrt(720.0));
    static const double qdrt5040 = sqrt(sqrt(5040.0));
    static const double sqrt24 = sqrt(24.0);
    static const double sqrt120 = sqrt(120.0);

    double angle = w.norm();
    double cangle = cos(angle);
    double sangle = sin(angle);
    double angle2 = angle * angle;

    double third_term = 0;
    if (angle > root4_eps * qdrt5040) {
        third_term = (2 * sangle - angle * (1 + cangle)) / (2 * sangle * angle2);
    } else {
        third_term = 1.0 / 12.0;
    }

    Eigen::Matrix3d hat_w = hat(w);
    return Eigen::Matrix3d::Identity() + 0.5 * hat_w + third_term * hat_w * hat_w;
}

Eigen::Matrix<double, 3, 2> s2_tangential_basis(const Eigen::Vector3d &x) {
    int d = 0;
    for (int i = 1; i < 3; ++i) {
        if (abs(x[i]) > abs(x[d]))
            d = i;
    }
    Eigen::Vector3d b1 = x.cross(Eigen::Vector3d::Unit((d + 1) % 3)).normalized();
    Eigen::Vector3d b2 = x.cross(b1).normalized();
    return (Eigen::Matrix<double, 3, 2>() << b1, b2).finished();
}

Eigen::Matrix<double, 3, 2> s2_tangential_basis_barrel(const Eigen::Vector3d &x) {
    Eigen::Vector3d b1 =
        x.cross(abs(x.z()) < 0.866 ? Eigen::Vector3d::UnitZ() : Eigen::Vector3d::UnitY())
            .normalized();
    Eigen::Vector3d b2 = x.cross(b1).normalized();
    return (Eigen::Matrix<double, 3, 2>() << b1, b2).finished();
}
