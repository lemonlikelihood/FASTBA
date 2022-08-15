#include "bundle_adjustor.h"
#include "../geometry/stereo.h"
#include "../map/feature.h"
#include "../map/frame.h"
#include "../map/sliding_window.h"
#include "ceres/lie_algebra_eigen_quaternion_parameterization.h"
#include "ceres/preintegration_error_cost.h"
#include "ceres/reprojection_error_cost.h"
#include "factor.h"

#include <ceres/ceres.h>

using namespace Eigen;
using namespace ceres;

struct BundleAdjustor::BundleAdjustorSolver {
    BundleAdjustorSolver() {
        cauchy_loss = std::make_unique<CauchyLoss>(1.0);
        quaternion_parameterization = std::make_unique<LieAlgebraEigenQuaternionParamatrization>();
    }
    std::unique_ptr<LossFunction> cauchy_loss;
    std::unique_ptr<LocalParameterization> quaternion_parameterization;
};

BundleAdjustor::BundleAdjustor() {
    solver = std::make_unique<BundleAdjustorSolver>();
}

BundleAdjustor::~BundleAdjustor() = default;

bool BundleAdjustor::solve(
    SlidingWindow *map, bool use_inertial, size_t max_iter, const double &max_time) {
    Problem::Options problem_options;
    problem_options.cost_function_ownership = DO_NOT_TAKE_OWNERSHIP;
    problem_options.loss_function_ownership = DO_NOT_TAKE_OWNERSHIP;
    problem_options.local_parameterization_ownership = DO_NOT_TAKE_OWNERSHIP;

    Problem problem(problem_options);

    for (size_t i = 0; i < map->frame_num(); ++i) {
        Frame *frame = map->get_frame(i);
        problem.AddParameterBlock(
            frame->pose.q.coeffs().data(), 4, solver->quaternion_parameterization.get());
        problem.AddParameterBlock(frame->pose.p.data(), 3);
        if (frame->flag(FrameFlag::FF_FIX_POSE)) {
            problem.SetParameterBlockConstant(frame->pose.q.coeffs().data());
            problem.SetParameterBlockConstant(frame->pose.p.data());
        }
        if (use_inertial) {
            problem.AddParameterBlock(frame->motion.v.data(), 3);
            problem.AddParameterBlock(frame->motion.bg.data(), 3);
            problem.AddParameterBlock(frame->motion.ba.data(), 3);
        }
    }

    std::unordered_set<Feature *> visited_features;
    for (size_t i = 0; i < map->frame_num(); ++i) {
        Frame *frame = map->get_frame(i);
        for (size_t j = 0; j < frame->keypoint_num(); ++j) {
            Feature *feature = frame->get_feature(j);
            if (!feature)
                continue;
            if (!feature->flag(FeatureFlag::FF_VALID)) // 判断有效的三维点，成功三角化过，有初始值
                continue;
            if (visited_features.count(feature) > 0)
                continue;
            visited_features.insert(feature);
            problem.AddParameterBlock(feature->p_in_G.data(), 3);
        }
    }

    for (size_t i = 0; i < map->frame_num(); ++i) {
        Frame *frame = map->get_frame(i);
        for (size_t j = 0; j < frame->keypoint_num(); ++j) {
            Feature *feature = frame->get_feature(j);
            if (!feature)
                continue;
            if (!feature->flag(FeatureFlag::FF_VALID))
                continue;
            problem.AddResidualBlock(
                frame->get_reprojection_factor(j)->get_cost_function<ReprojectionErrorCost>(),
                solver->cauchy_loss.get(), frame->pose.q.coeffs().data(), frame->pose.p.data(),
                feature->p_in_G.data());
        }
    }

    if (use_inertial) {
        for (size_t j = 1; j < map->frame_num(); ++j) {
            Frame *frame_i = map->get_frame(j - 1);
            Frame *frame_j = map->get_frame(j);
            if (frame_j->preintegration.integrate(
                    frame_j->image->t, frame_i->motion.bg, frame_i->motion.ba, true, true)) {
                problem.AddResidualBlock(
                    frame_j->get_preintegration_factor()
                        ->get_cost_function<PreIntegrationErrorCost>(),
                    nullptr, frame_i->pose.q.coeffs().data(), frame_i->pose.p.data(),
                    frame_i->motion.v.data(), frame_i->motion.bg.data(), frame_i->motion.ba.data(),
                    frame_j->pose.q.coeffs().data(), frame_j->pose.p.data(),
                    frame_j->motion.v.data(), frame_j->motion.bg.data(), frame_j->motion.ba.data());
            }
        }
    }

    Solver::Options solver_options;
    solver_options.linear_solver_type = SPARSE_SCHUR;
    solver_options.trust_region_strategy_type = DOGLEG;
    solver_options.use_explicit_schur_complement = true;
    solver_options.minimizer_progress_to_stdout = false;
    solver_options.logging_type = SILENT;
    solver_options.max_num_iterations = (int)max_iter;
    solver_options.max_solver_time_in_seconds = max_time;
    solver_options.num_threads = 1;
    solver_options.num_linear_solver_threads = 1;

    Solver::Summary solver_summary;
    ceres::Solve(solver_options, &problem, &solver_summary);

    for (size_t i = 0; i < map->feature_num(); ++i) {
        Feature *feature = map->get_feature(i);
        if (!feature->flag(FeatureFlag::FF_VALID))
            continue;
        const Vector3d &x = feature->p_in_G;
        double quality = 0.0;
        double quality_num = 0.0;
        for (const auto &k : feature->observation_map()) {
            Frame *frame = k.first;
            size_t keypoint_id = k.second;
            Pose pose = frame->get_camera_pose();
            Vector3d y = pose.q.conjugate() * (x - pose.p);
            if (y.z() <= 1.0e-3 || y.z() > 50) {
                feature->flag(FeatureFlag::FF_VALID) = false;
                break;
            }
            quality += (frame->apply_k(y.hnormalized()) - frame->get_keypoint(keypoint_id)).norm();
            quality_num += 1.0;
        }
        if (!feature->flag(FeatureFlag::FF_VALID))
            continue;
        feature->reprojection_error = quality / std::max(quality_num, 1.0);
    }

    return solver_summary.IsSolutionUsable();
}

struct LandmarkInfo {
    LandmarkInfo() {
        mat.setZero();
        vec.setZero();
    }
    Eigen::Matrix3d mat;
    Eigen::Vector3d vec;
    std::unordered_map<size_t, Eigen::Matrix<double, 3, 6>> h;
};
