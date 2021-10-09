//
// Created by lemon on 2021/1/21.
//

#ifndef FASTBA_EIGEN_BA_H
#define FASTBA_EIGEN_BA_H

#include "local_parameterization.h"
#include <Eigen/Eigen>

namespace fast_ba {
const int camera_real_size = 7;
const int quaternion_size = 4;
const int translation_size = 3;
const int camera_size = 6;
const int point_size = 3;
const int residual_size = 2;
typedef Eigen::Matrix<double, residual_size, camera_size> Jac_camera_factor;
typedef Eigen::Matrix<double, residual_size, point_size> Jac_point_factor;
typedef Eigen::Matrix<double, residual_size, 1> Residual_factor;
typedef Eigen::Matrix<double, camera_size, camera_size> Ftf_factor;
typedef Eigen::Matrix<double, point_size, point_size> Ete_factor;
typedef Eigen::Matrix<double, point_size, 1> Etb_factor;
typedef Eigen::Matrix<double, camera_size, 1> Ftb_factor;
typedef Eigen::Matrix<double, point_size, camera_size> Etf_factor;
} // namespace fast_ba

#endif //FASTBA_EIGEN_BA_H
