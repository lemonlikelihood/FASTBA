#include "bundle_adjustor.h"
#include "../geometry/stereo.h"
#include "../map/feature.h"
#include "../map/frame.h"
#include "../map/map.h"
// #include "../map/sliding_window.h"
#include "ceres/lie_algebra_eigen_quaternion_parameterization.h"
#include "ceres/marginalization_error_cost.h"
#include "ceres/preintegration_error_cost.h"
#include "ceres/reprojection_error_cost.h"
#include "factor.h"

#include <ceres/ceres.h>

using namespace ceres;

struct BundleAdjustorIterationCallback : public ceres::IterationCallback {
    std::vector<Factor::FactorCostFunction *> cost_functions;

    ceres::CallbackReturnType operator()(const ceres::IterationSummary &summary) override {
        for (Factor::FactorCostFunction *cost : cost_functions) {
            cost->update();
        }
        return ceres::SOLVER_CONTINUE;
    }
};

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

bool BundleAdjustor::solve(Map *map, bool use_inertial, size_t max_iter, const double &max_time) {
    Problem::Options problem_options;
    problem_options.cost_function_ownership = DO_NOT_TAKE_OWNERSHIP;
    problem_options.loss_function_ownership = DO_NOT_TAKE_OWNERSHIP;
    problem_options.local_parameterization_ownership = DO_NOT_TAKE_OWNERSHIP;
    Problem problem(problem_options);
    BundleAdjustorIterationCallback iteration_callback;

    // add parameter frame
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

    // add parameter point
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
            if (feature->flag(FeatureFlag::FF_FIXED)) {
                problem.SetParameterBlockConstant(feature->p_in_G.data());
            }
        }
    }


    // add marginalization cost
    if (map->get_marginalization_factor()) {
        // log_info("[BundleAdjustor] get_marginalization_factor ");
        MarginalizationErrorCost *marcost =
            map->get_marginalization_factor()->get_cost_function<MarginalizationErrorCost>();
        std::vector<double *> params;
        for (size_t i = 0; i < marcost->related_frames().size(); ++i) {
            Frame *frame = marcost->related_frames()[i];
            params.emplace_back(frame->pose.q.coeffs().data());
            params.emplace_back(frame->pose.p.data());
            params.emplace_back(frame->motion.v.data());
            params.emplace_back(frame->motion.bg.data());
            params.emplace_back(frame->motion.ba.data());
        }
        problem.AddResidualBlock(marcost, nullptr, params);
        iteration_callback.cost_functions.push_back(marcost);
    }

    // add reprojection error
    for (size_t i = 0; i < map->frame_num(); ++i) {
        Frame *frame = map->get_frame(i);
        for (size_t j = 0; j < frame->keypoint_num(); ++j) {
            Feature *feature = frame->get_feature(j);
            if (!feature)
                continue;
            if (!feature->flag(FeatureFlag::FF_VALID))
                continue;
            ReprojectionErrorCost *rpcost =
                frame->get_reprojection_factor(j)->get_cost_function<ReprojectionErrorCost>();
            problem.AddResidualBlock(
                rpcost, solver->cauchy_loss.get(), frame->pose.q.coeffs().data(),
                frame->pose.p.data(), feature->p_in_G.data());
            iteration_callback.cost_functions.push_back(rpcost);
        }
    }

    if (use_inertial) {
        for (size_t j = 1; j < map->frame_num(); ++j) {
            Frame *frame_i = map->get_frame(j - 1);
            Frame *frame_j = map->get_frame(j);
            // log_info("[BundleAdjustor] preintegration begin ");
            if (frame_j->preintegration.integrate(
                    frame_j->image->t, frame_i->motion.bg, frame_i->motion.ba, true, true)) {
                PreIntegrationErrorCost *picost =
                    frame_j->get_preintegration_factor()
                        ->get_cost_function<PreIntegrationErrorCost>();
                problem.AddResidualBlock(
                    picost, nullptr, frame_i->pose.q.coeffs().data(), frame_i->pose.p.data(),
                    frame_i->motion.v.data(), frame_i->motion.bg.data(), frame_i->motion.ba.data(),
                    frame_j->pose.q.coeffs().data(), frame_j->pose.p.data(),
                    frame_j->motion.v.data(), frame_j->motion.bg.data(), frame_j->motion.ba.data());
                iteration_callback.cost_functions.push_back(picost);
            }
            // log_info("[BundleAdjustor] preintegration over ");
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
        const Eigen::Vector3d &x = feature->p_in_G;
        double quality = 0.0;
        double quality_num = 0.0;
        for (const auto &k : feature->observation_map()) {
            Frame *frame = k.first;
            size_t keypoint_id = k.second;
            Pose pose = frame->get_camera_pose();
            Eigen::Vector3d y = pose.q.conjugate() * (x - pose.p);
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

void BundleAdjustor::marginalize_frame(Map *map, size_t index) {
    log_info("[BundleAdjustor]: marginalize_frame {}", index);
    Eigen::MatrixXd pose_motion_info_mat;
    Eigen::VectorXd pose_motion_info_vec;
    std::map<Feature *, LandmarkInfo> landmark_info;

    pose_motion_info_mat.resize(map->frame_num() * ES_SIZE, map->frame_num() * ES_SIZE);
    pose_motion_info_vec.resize(map->frame_num() * ES_SIZE);
    pose_motion_info_mat.setZero();
    pose_motion_info_vec.setZero();

    std::unordered_map<Frame *, size_t> frame_indices;
    for (size_t i = 0; i < map->frame_num(); ++i) {
        frame_indices[map->get_frame(i)] = i;
    }

    /* scope : marginalization factor*/
    if (map->get_marginalization_factor()) {
        MarginalizationErrorCost *marcost =
            map->get_marginalization_factor()->get_cost_function<MarginalizationErrorCost>();
        const std::vector<Frame *> frames = marcost->related_frames();
        std::vector<const double *> marparameters(frames.size() * 5);
        std::vector<double *> marjacobians_ptrs(frames.size() * 5);
        std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, true>> marjacobians(
            frames.size() * 5);
        for (size_t i = 0; i < frames.size(); ++i) {
            // log_info("[raw marginalization factor]: i {}: {}", i, frames[i]->id());
            marparameters[5 * i + 0] = frames[i]->pose.q.coeffs().data();
            marparameters[5 * i + 1] = frames[i]->pose.p.data();
            marparameters[5 * i + 2] = frames[i]->motion.v.data();
            marparameters[5 * i + 3] = frames[i]->motion.bg.data();
            marparameters[5 * i + 4] = frames[i]->motion.ba.data();

            marjacobians[5 * i + 0].resize(frames.size() * ES_SIZE, 4);
            marjacobians[5 * i + 1].resize(frames.size() * ES_SIZE, 3);
            marjacobians[5 * i + 2].resize(frames.size() * ES_SIZE, 3);
            marjacobians[5 * i + 3].resize(frames.size() * ES_SIZE, 3);
            marjacobians[5 * i + 4].resize(frames.size() * ES_SIZE, 3);

            marjacobians_ptrs[5 * i + 0] = marjacobians[5 * i + 0].data();
            marjacobians_ptrs[5 * i + 1] = marjacobians[5 * i + 1].data();
            marjacobians_ptrs[5 * i + 2] = marjacobians[5 * i + 2].data();
            marjacobians_ptrs[5 * i + 3] = marjacobians[5 * i + 3].data();
            marjacobians_ptrs[5 * i + 4] = marjacobians[5 * i + 4].data();
        }

        Eigen::VectorXd marresidual;
        marresidual.resize(frames.size() * ES_SIZE);
        marcost->Evaluate(marparameters.data(), marresidual.data(), marjacobians_ptrs.data());
        // for (size_t i = 0; i < marparameters.size(); ++i) {
        //     log_info("marparameters[{}]: {}", i, *marparameters[i]);
        // }

        // log_info("marresidual: {}", marresidual);

        // for (size_t i = 0; i < marjacobians_ptrs.size(); ++i) {
        //     log_info("marjacobians_ptrs[{}]: {}", i, *marjacobians_ptrs[i]);
        // }

        std::vector<Eigen::MatrixXd> state_jacobians(frames.size());
        for (size_t i = 0; i < frames.size(); ++i) {
            size_t frame_index = frame_indices[frames[i]];
            Eigen::MatrixXd &dr_ds = state_jacobians[frame_index];
            dr_ds.resize(frames.size() * ES_SIZE, ES_SIZE);
            dr_ds.block(0, ES_Q, frames.size() * ES_SIZE, 3) = marjacobians[5 * i + 0].leftCols(3);
            dr_ds.block(0, ES_P, frames.size() * ES_SIZE, 3) = marjacobians[5 * i + 1];
            dr_ds.block(0, ES_V, frames.size() * ES_SIZE, 3) = marjacobians[5 * i + 2];
            dr_ds.block(0, ES_BG, frames.size() * ES_SIZE, 3) = marjacobians[5 * i + 3];
            dr_ds.block(0, ES_BA, frames.size() * ES_SIZE, 3) = marjacobians[5 * i + 4];
        }

        for (size_t i = 0; i < frames.size(); ++i) {
            for (size_t j = 0; j < frames.size(); ++j) {
                pose_motion_info_mat.block<ES_SIZE, ES_SIZE>(ES_SIZE * i, ES_SIZE * j) +=
                    state_jacobians[i].transpose() * state_jacobians[j];
            }
            pose_motion_info_vec.segment<ES_SIZE>(ES_SIZE * i) +=
                state_jacobians[i].transpose() * marresidual;
        }
        // log_info("[margin_frame]: get_marginalization_factor");
    }


    /*scope: preintegration factor*/
    for (size_t j = index; j <= index + 1; ++j) {
        if (j == 0)
            continue;
        if (j >= map->feature_num())
            continue;
        size_t i = j - 1;
        Frame *frame_i = map->get_frame(i);
        Frame *frame_j = map->get_frame(j);

        if (!frame_j->get_preintegration_factor())
            continue;
        PreIntegrationErrorCost *picost =
            frame_j->get_preintegration_factor()->get_cost_function<PreIntegrationErrorCost>();
        std::array<const double *, 10> piparameters = {
            frame_i->pose.q.coeffs().data(), frame_i->pose.p.data(),
            frame_i->motion.v.data(),        frame_i->motion.bg.data(),
            frame_i->motion.ba.data(),       frame_j->pose.q.coeffs().data(),
            frame_j->pose.p.data(),          frame_j->motion.v.data(),
            frame_j->motion.bg.data(),       frame_j->motion.ba.data()};
        Eigen::VectorXd piresidual(ES_SIZE);
        Eigen::Matrix<double, ES_SIZE, 4, true> dr_dqi, dr_dqj;
        Eigen::Matrix<double, ES_SIZE, 3, true> dr_dpi, dr_dpj;
        Eigen::Matrix<double, ES_SIZE, 3, true> dr_dvi, dr_dvj;
        Eigen::Matrix<double, ES_SIZE, 3, true> dr_dbgi, dr_dbgj;
        Eigen::Matrix<double, ES_SIZE, 3, true> dr_dbai, dr_dbaj;

        std::array<double *, 10> pijacobians = {
            dr_dqi.data(), dr_dpi.data(), dr_dvi.data(), dr_dbgi.data(), dr_dbai.data(),
            dr_dqj.data(), dr_dpj.data(), dr_dvj.data(), dr_dbgj.data(), dr_dbaj.data(),
        };

        picost->Evaluate(piparameters.data(), piresidual.data(), pijacobians.data());

        Eigen::Matrix<double, ES_SIZE, ES_SIZE * 2> dr_dstates;

        dr_dstates.block<ES_SIZE, 3>(0, ES_SIZE * 0 + ES_Q) = dr_dqi.block<ES_SIZE, 3>(0, 0);
        dr_dstates.block<ES_SIZE, 3>(0, ES_SIZE * 0 + ES_P) = dr_dpi;
        dr_dstates.block<ES_SIZE, 3>(0, ES_SIZE * 0 + ES_V) = dr_dvi;
        dr_dstates.block<ES_SIZE, 3>(0, ES_SIZE * 0 + ES_BG) = dr_dbgi;
        dr_dstates.block<ES_SIZE, 3>(0, ES_SIZE * 0 + ES_BA) = dr_dbai;

        dr_dstates.block<ES_SIZE, 3>(0, ES_SIZE * 1 + ES_Q) = dr_dqj.block<ES_SIZE, 3>(0, 0);
        dr_dstates.block<ES_SIZE, 3>(0, ES_SIZE * 1 + ES_P) = dr_dpj;
        dr_dstates.block<ES_SIZE, 3>(0, ES_SIZE * 1 + ES_V) = dr_dvj;
        dr_dstates.block<ES_SIZE, 3>(0, ES_SIZE * 1 + ES_BG) = dr_dbgj;
        dr_dstates.block<ES_SIZE, 3>(0, ES_SIZE * 1 + ES_BA) = dr_dbaj;

        pose_motion_info_mat.block<ES_SIZE * 2, ES_SIZE * 2>(ES_SIZE * i, ES_SIZE * i) +=
            dr_dstates.transpose() * dr_dstates;
        pose_motion_info_vec.segment<ES_SIZE * 2>(ES_SIZE * i) +=
            dr_dstates.transpose() * piresidual;
    }

    log_info("[margin_frame]: get preintegration factor");

    /* scope: reprojection error factor */
    Frame *frame_victim = map->get_frame(index);
    for (size_t j = 0; j < frame_victim->keypoint_num(); ++j) {
        Feature *feature = frame_victim->get_feature(j);
        if (!feature || !feature->flag(FeatureFlag::FF_VALID))
            continue;
        Frame *frame_ref = feature->first_frame();
        // size_t frame_index_ref = frame_indices.at(frame_ref);
        for (const auto &[frame_tgt, keypoint_index] : feature->observation_map()) {
            if (frame_tgt == frame_ref)
                continue;
            if (frame_indices.count(frame_tgt) == 0)
                continue;
            size_t frame_index_tgt = frame_indices.at(frame_tgt);
            ReprojectionErrorCost *rpcost = frame_tgt->get_reprojection_factor(keypoint_index)
                                                ->get_cost_function<ReprojectionErrorCost>();
            std::array<const double *, 3> rpparameters = {
                frame_tgt->pose.q.coeffs().data(), frame_tgt->pose.p.data(),
                feature->p_in_G.data()};
            Eigen::Vector2d rpresidual;
            Eigen::Matrix<double, 2, 4> dr_dq_tgt;
            Eigen::Matrix<double, 2, 3> dr_dp_tgt;
            Eigen::Matrix<double, 2, 3> dr_dx;
            std::array<double *, 3> rpjacobians = {
                dr_dq_tgt.data(), dr_dp_tgt.data(), dr_dx.data()};

            rpcost->Evaluate(rpparameters.data(), rpresidual.data(), rpjacobians.data());

            Eigen::Matrix<double, 2, 3> dr_dq_tgt_local = dr_dq_tgt.block<2, 3>(0, 0);

            pose_motion_info_mat.block<3, 3>(
                ES_SIZE * frame_index_tgt + ES_Q, ES_SIZE * frame_index_tgt + ES_Q) +=
                dr_dq_tgt_local.transpose() * dr_dq_tgt_local;
            pose_motion_info_mat.block<3, 3>(
                ES_SIZE * frame_index_tgt + ES_P, ES_SIZE * frame_index_tgt + ES_P) +=
                dr_dp_tgt.transpose() * dr_dp_tgt;

            pose_motion_info_mat.block<3, 3>(
                ES_SIZE * frame_index_tgt + ES_Q, ES_SIZE * frame_index_tgt + ES_P) +=
                dr_dq_tgt_local.transpose() * dr_dp_tgt;
            pose_motion_info_mat.block<3, 3>(
                ES_SIZE * frame_index_tgt + ES_P, ES_SIZE * frame_index_tgt + ES_Q) +=
                dr_dp_tgt.transpose() * dr_dq_tgt_local;

            pose_motion_info_vec.segment<3>(ES_SIZE * frame_index_tgt + ES_Q) +=
                dr_dq_tgt_local.transpose() * rpresidual;
            pose_motion_info_vec.segment<3>(ES_SIZE * frame_index_tgt + ES_P) +=
                dr_dp_tgt.transpose() * rpresidual;

            LandmarkInfo &linfo = landmark_info[feature];

            linfo.mat += dr_dx.transpose() * dr_dx;
            linfo.vec += dr_dx.transpose() * rpresidual;
            if (linfo.h.count(frame_index_tgt) == 0) {
                linfo.h[frame_index_tgt].setZero();
            }
            /* scope */ {
                Eigen::Matrix<double, 3, 6> &h = linfo.h.at(frame_index_tgt);
                // h.block<3, 3>(ES_Q - ES_Q, ES_Q) += dr_dx.transpose() * dr_dq_tgt_local;
                // h.block<3, 3>(ES_P - ES_Q, ES_Q) += dr_dx.transpose() * dr_dp_tgt;
                h.block<3, 3>(0, 0) += dr_dx.transpose() * dr_dq_tgt_local;
                h.block<3, 3>(0, 3) += dr_dx.transpose() * dr_dp_tgt;
            }
        }
    }

    // log_info("[margin_frame]: get reprojection factor");

    /* scope: marginalize landmarks */
    for (const auto &[track, info] : landmark_info) {
        if (std::abs(info.mat.determinant()) < 1e-9) {
            continue;
        }
        Eigen::Matrix3d inv_infomat = info.mat.inverse();
        if (!std::isfinite(inv_infomat.determinant())) {
            // log_error("[finite] : \n{}", info.mat);
            continue;
        }
        for (const auto &[frame_index_i, h_i] : info.h) {
            for (const auto &[frame_index_j, h_j] : info.h) {
                pose_motion_info_mat.block<6, 6>(
                    ES_SIZE * frame_index_i + ES_Q, ES_SIZE * frame_index_j + ES_Q) -=
                    h_i.transpose() * inv_infomat * h_j;
            }
            pose_motion_info_vec.segment<6>(ES_SIZE * frame_index_i + ES_Q) -=
                h_i.transpose() * inv_infomat * info.vec;
        }
    }

    // log_info("[margin_frame]: get landmark factor");

    /* scope: marginalize the corresponding frame */ {
        Eigen::Matrix<double, 15, 15> inv_infomat =
            pose_motion_info_mat.block<ES_SIZE, ES_SIZE>(ES_SIZE * index, ES_SIZE * index)
                .inverse();
        Eigen::MatrixXd complement_infomat;
        Eigen::VectorXd complement_infovec;
        complement_infomat.resize(
            ES_SIZE * (map->frame_num() - 1), ES_SIZE * (map->frame_num() - 1));
        complement_infovec.resize(ES_SIZE * (map->frame_num() - 1));
        complement_infomat.setZero();
        complement_infovec.setZero();

        if (index > 0) {
            Eigen::MatrixXd complement_infomat_block =
                pose_motion_info_mat.block(0, 0, ES_SIZE * index, ES_SIZE * index);
            Eigen::VectorXd complement_infovec_segment =
                pose_motion_info_vec.segment(0, ES_SIZE * index);
            complement_infomat_block -=
                pose_motion_info_mat.block(0, ES_SIZE * index, ES_SIZE * index, ES_SIZE)
                * inv_infomat
                * pose_motion_info_mat.block(ES_SIZE * index, 0, ES_SIZE, ES_SIZE * index);
            complement_infovec_segment -=
                pose_motion_info_mat.block(0, ES_SIZE * index, ES_SIZE * index, ES_SIZE)
                * inv_infomat * pose_motion_info_vec.segment(ES_SIZE * index, ES_SIZE);
            complement_infomat.block(0, 0, ES_SIZE * index, ES_SIZE * index) =
                complement_infomat_block;
            complement_infovec.segment(0, ES_SIZE * index) = complement_infovec_segment;
        }
        if (index < map->frame_num() - 1) {
            Eigen::MatrixXd complement_infomat_block = pose_motion_info_mat.block(
                ES_SIZE * (index + 1), ES_SIZE * (index + 1),
                ES_SIZE * (map->frame_num() - 1 - index), ES_SIZE * (map->frame_num() - 1 - index));
            Eigen::VectorXd complement_infovec_segment = pose_motion_info_vec.segment(
                ES_SIZE * (index + 1), ES_SIZE * (map->frame_num() - 1 - index));
            complement_infomat_block -= pose_motion_info_mat.block(
                                            ES_SIZE * (index + 1), ES_SIZE * index,
                                            ES_SIZE * (map->frame_num() - 1 - index), ES_SIZE)
                                        * inv_infomat
                                        * pose_motion_info_mat.block(
                                            ES_SIZE * index, ES_SIZE * (index + 1), ES_SIZE,
                                            ES_SIZE * (map->frame_num() - 1 - index));
            complement_infovec_segment -= pose_motion_info_mat.block(
                                              ES_SIZE * (index + 1), ES_SIZE * index,
                                              ES_SIZE * (map->frame_num() - 1 - index), ES_SIZE)
                                          * inv_infomat
                                          * pose_motion_info_vec.segment(ES_SIZE * index, ES_SIZE);
            complement_infomat.block(
                ES_SIZE * index, ES_SIZE * index, ES_SIZE * (map->frame_num() - 1 - index),
                ES_SIZE * (map->frame_num() - 1 - index)) = complement_infomat_block;
            complement_infovec.segment(ES_SIZE * index, ES_SIZE * (map->frame_num() - 1 - index)) =
                complement_infovec_segment;
        }
        if (index > 0 && index < map->frame_num() - 1) {
            Eigen::MatrixXd complement_infomat_block = pose_motion_info_mat.block(
                0, ES_SIZE * (index + 1), ES_SIZE * index,
                ES_SIZE * (map->frame_num() - 1 - index));
            complement_infomat_block -=
                pose_motion_info_mat.block(0, ES_SIZE * index, ES_SIZE * index, ES_SIZE)
                * inv_infomat
                * pose_motion_info_mat.block(
                    ES_SIZE * index, ES_SIZE * (index + 1), ES_SIZE,
                    ES_SIZE * (map->frame_num() - 1 - index));
            complement_infomat.block(
                0, ES_SIZE * index, ES_SIZE * index, ES_SIZE * (map->frame_num() - 1 - index)) =
                complement_infomat_block;
            complement_infomat.block(
                ES_SIZE * index, 0, ES_SIZE * (map->frame_num() - 1 - index), ES_SIZE * index) =
                complement_infomat_block.transpose();
        }

        pose_motion_info_mat = complement_infomat;
        pose_motion_info_vec = complement_infovec;
    }

    // log_info("[margin_frame]: marginalize the corresponding frame");

    /* scope: create marginalization factor */ {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saesolver(pose_motion_info_mat);
        // log_info("[margin_frame]: pose_motion_info_mat \n{} ", pose_motion_info_mat);

        Eigen::VectorXd lambdas =
            (saesolver.eigenvalues().array() > 1.0e-8).select(saesolver.eigenvalues(), 0);
        Eigen::VectorXd lambdas_inv = (saesolver.eigenvalues().array() > 1.0e-8)
                                          .select(saesolver.eigenvalues().cwiseInverse(), 0);

        Eigen::MatrixXd sqrt_infomat =
            lambdas.cwiseSqrt().asDiagonal() * saesolver.eigenvectors().transpose();
        Eigen::VectorXd sqrt_infovec = lambdas_inv.cwiseSqrt().asDiagonal()
                                       * saesolver.eigenvectors().transpose()
                                       * pose_motion_info_vec;

        std::vector<Frame *> remaining_frames;
        for (size_t i = 0; i < map->frame_num(); ++i) {
            if (i == index)
                continue;
            remaining_frames.emplace_back(map->get_frame(i));
        }
        map->set_marginalization_factor(Factor::create_marginalization_error(
            sqrt_infomat, sqrt_infovec, std::move(remaining_frames)));
    }
    // log_info("[margin_frame]: create marginalization factor ");
    // getchar();
}