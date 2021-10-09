//
// Created by SENSETIME\yezhichao1 on 2020/4/23.
//
#pragma once

#include <Eigen/Eigen>

//constexprdouble SLIGHT_ANGLE = 0.008726646;  // 0.5°
constexpr double SLIGHT_ANGLE = 1.745329252e-7; // 1.0e-5°

inline Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d &q)
{
    Eigen::Matrix3d ans;
    ans << 0, -q(2), q(1),
        q(2), 0, -q(0),
        -q(1), q(0), 0;
    return ans;
}

template <typename Derived>
inline Eigen::Quaternion<typename Derived::Scalar> delta_q(const Eigen::MatrixBase<Derived> &theta)
{
    typedef typename Derived::Scalar Scalar_t;
    Eigen::Quaternion<Scalar_t> dq;
    Eigen::Matrix<Scalar_t, 3, 1> half_theta = theta;
    half_theta = half_theta / static_cast<Scalar_t>(2.0);
    double norm = half_theta.norm();

    if (norm > SLIGHT_ANGLE)
    {
        dq.w() = static_cast<Scalar_t>(cos(norm));
        dq.x() = sin(norm) * half_theta.x() / norm;
        dq.y() = sin(norm) * half_theta.y() / norm;
        dq.z() = sin(norm) * half_theta.z() / norm;
    }
    else
    {
        dq.w() = static_cast<Scalar_t>(1.0);
        dq.x() = half_theta.x();
        dq.y() = half_theta.y();
        dq.z() = half_theta.z();
    }
    return dq;
}

template <typename Derived>
inline Eigen::Quaternion<typename Derived::Scalar> delta_q_ceres(const Eigen::MatrixBase<Derived> &theta)
{
    typedef typename Derived::Scalar Scalar_t;
    Eigen::Quaternion<Scalar_t> dq;
    double norm = theta.norm();
    if (norm > 0.0)
    {
        const double sin_delta_by_delta = (sin(norm) / norm);
        dq.w() = static_cast<Scalar_t>(cos(norm));
        dq.x() = sin_delta_by_delta * theta.x();
        dq.y() = sin_delta_by_delta * theta.y();
        dq.z() = sin_delta_by_delta * theta.z();
    }
    else
    {
        dq.w() = static_cast<Scalar_t>(1.0);
        dq.x() = theta.x();
        dq.y() = theta.y();
        dq.z() = theta.z();
    }
    return dq;
}

inline Eigen::Quaterniond expmap(const Eigen::Vector3d &w)
{
    Eigen::AngleAxisd aa(w.norm(), w.stableNormalized());
    Eigen::Quaterniond q;
    q = aa;
    return q;
}

inline Eigen::Matrix3d jri(Eigen::Vector3d theta)
{
    double norm = theta.norm();
    Eigen::Matrix3d jri;
    if (norm < SLIGHT_ANGLE)
    {
        jri = Eigen::Matrix3d::Identity() +
              1.0 / 2.0 * skewSymmetric(theta) +
              1.0 / 4.0 * skewSymmetric(theta) * skewSymmetric(theta);
    }
    else
    {
        jri = Eigen::Matrix3d::Identity() +
              1.0 / 2.0 * skewSymmetric(theta) +
              ((1.0) / (norm * norm) - (1.0 + cos(norm)) / (2 * norm * sin(norm))) *
                  skewSymmetric(theta) * skewSymmetric(theta);
    }
    return jri;
}

inline Eigen::Matrix<double, 3, 2> s2_tangential_basis(const Eigen::Vector3d &x)
{
    int d = 0;
    for (int i = 1; i < 3; ++i)
    {
        if (abs(x[i]) > abs(x[d]))
            d = i;
    }
    Eigen::Vector3d b1 = x.cross(Eigen::Vector3d::Unit((d + 1) % 3)).normalized();
    Eigen::Vector3d b2 = x.cross(b1).normalized();
    return (Eigen::Matrix<double, 3, 2>() << b1, b2).finished();
}

template <typename D>
inline Eigen::Matrix<typename D::Scalar, 3, 1> log(const Eigen::QuaternionBase<D> &q)
{
    using Scalar = typename D::Scalar;
    Eigen::AngleAxis<Scalar> as(q);
    return as.angle() * as.axis();
}

inline Eigen::Matrix3d Jr(const Eigen::Vector3d &theta)
{
    double norm = theta.norm();
    Eigen::Matrix3d jr;
    if (norm < SLIGHT_ANGLE)
    {
        jr = Eigen::Matrix3d::Identity() -
             1.0 / 2.0 * skewSymmetric(theta) +
             1.0 / 6.0 * skewSymmetric(theta) * skewSymmetric(theta);
    }
    else
    {
        jr = Eigen::Matrix3d::Identity() -
             (1.0 - cos(norm)) / (norm * norm) * skewSymmetric(theta) +
             (norm - sin(norm)) / (norm * norm * norm) * skewSymmetric(theta) * skewSymmetric(theta);
    }
    return jr;
}