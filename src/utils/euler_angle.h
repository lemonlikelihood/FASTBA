#pragma once

#include "common.h"

Eigen::Vector3d R2ypr(const Eigen::Matrix3d &R);
Eigen::Matrix3d ypr2R(const Eigen::Vector3d &ypr);
Eigen::Matrix3d g2R(const Eigen::Vector3d &g);
Eigen::Matrix3d Rxy(const Eigen::Matrix3d &R);
Eigen::Matrix3d Rz(const Eigen::Matrix3d &R);

Eigen::Vector2d get_gravity_direction(const Eigen::Matrix3d &R);
// void compute_euler(const std::vector<Pose> &pose_vec, std::vector<YprData> &ypr_vec);
// void compute_yaw_gravity(const std::vector<Pose> &pose_vec, std::vector<YprData> &ypr_vec);
