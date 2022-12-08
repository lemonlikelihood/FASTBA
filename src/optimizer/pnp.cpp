#include "pnp.h"
#include <ceres/ceres.h>

#include "../map/feature.h"
#include "../map/frame.h"
#include "../map/map.h"
#include "ceres/lie_algebra_eigen_quaternion_parameterization.h"
#include "ceres/preintegration_error_cost.h"
#include "ceres/reprojection_error_cost.h"

using namespace ceres;

void visual_inertial_pnp(
    Map *map, Frame *frame, bool use_inertial, size_t max_iter, const double &max_time) {
    LocalParameterization *eigen_quaternion = new LieAlgebraEigenQuaternionParamatrization;
    LossFunction *cauchy_loss = new CauchyLoss(1.0);

    Problem problem;

    problem.AddParameterBlock(frame->pose.q.coeffs().data(), 4, eigen_quaternion);
    problem.AddParameterBlock(frame->pose.p.data(), 3);

    Frame *last_frame = map->get_last_frame();

    if (use_inertial) {
        problem.AddResidualBlock(
            new PreIntegrationPriorCost(last_frame, frame), nullptr, frame->pose.q.coeffs().data(),
            frame->pose.p.data(), frame->motion.v.data(), frame->motion.bg.data(),
            frame->motion.ba.data());
    }

    int valid_num = 0;
    for (size_t i = 0; i < frame->keypoint_num(); ++i) {
        Feature *feature = frame->get_feature(i);
        if (!feature)
            continue;
        if (!feature->has_observation(last_frame))
            continue;
        if (feature->flag(FeatureFlag::FF_VALID)) {
            valid_num++;
            problem.AddResidualBlock(
                new PoseOnlyReprojectionErrorCost(frame, i), cauchy_loss,
                frame->pose.q.coeffs().data(), frame->pose.p.data());
        }
    }

    log_info("[pnp]: valid num: {}", valid_num);
    if (valid_num < 30) {
        getchar();
    }

    Solver::Options solver_options;
    solver_options.linear_solver_type = ceres::DENSE_SCHUR;
    solver_options.trust_region_strategy_type = ceres::DOGLEG;
    solver_options.use_explicit_schur_complement = true;
    solver_options.minimizer_progress_to_stdout = false;
    solver_options.logging_type = ceres::SILENT;
    solver_options.max_num_iterations = (int)max_iter;
    solver_options.max_solver_time_in_seconds = max_time;
    solver_options.num_threads = 1;
    solver_options.num_linear_solver_threads = 1;

    Solver::Summary solver_summary;
    Solve(solver_options, &problem, &solver_summary);
}