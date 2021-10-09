#ifndef LOCAL_PARAMETERIZATION_H_
#define LOCAL_PARAMETERIZATION_H_
#include <array>
#include <memory>
#include <vector>
#include <Eigen/Eigen>
#include <iostream>
#include "../utils/matrix_math.h"
namespace fast_ba
{
    class LocalParameterization
    {
    public:
        virtual ~LocalParameterization();

        virtual bool Plus(const double *x,
                          const double *delta,
                          double *x_plus_delta) const = 0;

        virtual bool ComputeJacobian(const double *x, double *jacobian) const = 0;

        virtual bool MultiplyByJacobian(const double *x,
                                        const int num_rows,
                                        const double *global_matrix,
                                        double *local_matrix) const;

        // Size of x.
        virtual int GlobalSize() const = 0;

        // Size of delta.
        virtual int LocalSize() const = 0;
    };

    class QuatParam : public LocalParameterization
    {
    public:
        bool Plus(const double *x, const double *delta, double *x_plus_delta) const override
        {
            // const Eigen::Quaterniond _q(x[0], x[1], x[2], x[3]);
            Eigen::Map<const Eigen::Quaterniond> _q(x);
            // Eigen::Quaterniond q;
            Eigen::Map<Eigen::Quaterniond> q(x_plus_delta);
            Eigen::Map<const Eigen::Vector3d> theta(delta);

            Eigen::Quaterniond dq = expmap(theta); //delta_q(theta);
            q = (_q * dq).normalized();
            // x_plus_delta[0] = q.w();
            // x_plus_delta[1] = q.x();
            // x_plus_delta[2] = q.y();
            // x_plus_delta[3] = q.z();

            return true;
        }

        bool ComputeJacobian(const double *x, double *jacobian) const override
        {
            Eigen::Map<Eigen::Matrix<double, 4, 3, Eigen::RowMajor>> j(jacobian);
            j.setIdentity();
            return true;
        }

        int GlobalSize() const override { return 4; };

        int LocalSize() const override { return 3; };
    };
}
#endif