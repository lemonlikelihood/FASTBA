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
#pragma once

#include <Eigen/Eigen>

inline Eigen::Matrix3d hat(const Eigen::Vector3d &w) {
    return (Eigen::Matrix3d() << 0, -w.z(), w.y(), w.z(), 0, -w.x(), -w.y(), w.x(), 0).finished();
}

inline Eigen::Quaterniond expmap(const Eigen::Vector3d &w) {
    Eigen::AngleAxisd aa(w.norm(), w.stableNormalized());
    Eigen::Quaterniond q;
    q = aa;
    return q;
}

inline Eigen::Vector3d logmap(const Eigen::Quaterniond &q) {
    Eigen::AngleAxisd aa(q);
    return aa.angle() * aa.axis();
}

Eigen::Matrix3d right_jacobian(const Eigen::Vector3d &w);
Eigen::Matrix3d right_jacobian_inv(const Eigen::Vector3d &w);
// matrix<3, 2> s2_tangential_basis(const Eigen::Vector3d &x);
// matrix<3, 2> s2_tangential_basis_barrel(const Eigen::Vector3d &x);
