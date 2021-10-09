//
// Created by lemon on 2021/2/4.
//

#ifndef FASTBA_PROJECTION_FACTOR_H
#define FASTBA_PROJECTION_FACTOR_H

#include "../utils/matrix_math.h"
#include "eigen_ba.h"
#include <Eigen/Eigen>

namespace fast_ba {
class ProjectionFactor {
public:
    ProjectionFactor() {};
    ProjectionFactor(const Eigen::Vector2d &measurement_) : measurement(measurement_) {}

    bool Evaluate(
        double const *const *parameters, double *residuals, double **jacobians,
        double **factors) const {
        Eigen::Map<const Eigen::Quaterniond> q_cw(parameters[0]);
        Eigen::Map<const Eigen::Vector3d> p_cw(parameters[1]);
        Eigen::Map<const Eigen::Vector3d> x_w(parameters[2]);
        Eigen::Map<Eigen::Vector2d> residual(residuals);

        Eigen::Matrix3d R_cw = q_cw.toRotationMatrix();
        Eigen::Vector3d a = R_cw * x_w + p_cw;
        residual = a.head<2>() / a.z() - measurement;

        // std::cout << "residual0:  " << residual << std::endl;
        Eigen::Matrix<double, 2, 3> dr_da;
        Eigen::Matrix3d da_dR;

        dr_da << 1.0 / a.z(), 0.0, -a.x() / (a.z() * a.z()), 0.0, 1.0 / a.z(),
            -a.y() / (a.z() * a.z());
        da_dR = -R_cw * skewSymmetric(x_w);
        Eigen::Matrix<double, 2, 3> dr_dR = dr_da * da_dR;

        Eigen::Map<Jac_camera_factor> jacobian_camera(jacobians[0]);
        Eigen::Map<Jac_point_factor> jacobian_point(jacobians[1]);

        jacobian_camera.leftCols<3>() = dr_dR;
        jacobian_camera.rightCols<3>() = dr_da;
        jacobian_point = dr_da * R_cw;

        Eigen::Map<Ftf_factor> ftf_factor(factors[0]);
        ftf_factor = jacobian_camera.transpose() * jacobian_camera;

        Eigen::Map<Ftb_factor> ftb_factor(factors[1]);
        ftb_factor = jacobian_camera.transpose() * residual;

        Eigen::Map<Etf_factor> etf_factor(factors[2]);
        etf_factor = jacobian_point.transpose() * jacobian_camera;

        Eigen::Map<Ete_factor> ete_factor(factors[3]);
        ete_factor = jacobian_point.transpose() * jacobian_point;

        Eigen::Map<Etb_factor> etb_factor(factors[4]);
        etb_factor = jacobian_point.transpose() * residual;

        return true;
    }

    Eigen::Vector2d measurement;
};
};     // namespace fast_ba
#endif //FASTBA_PROJECTION_FACTOR_H
