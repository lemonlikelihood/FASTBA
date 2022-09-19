#include "factor.h"
#include "../map/frame.h"
#include "./ceres/marginalization_error_cost.h"
#include "./ceres/preintegration_error_cost.h"
#include "./ceres/reprojection_error_cost.h"

std::unique_ptr<Factor> Factor::create_reprojection_error(Frame *frame, size_t keypoint_id) {
    return std::make_unique<Factor>(
        std::make_unique<ReprojectionErrorCost>(frame, keypoint_id), factor_construct_t());
}

std::unique_ptr<Factor> Factor::create_preintegration_error(Frame *frame_i, Frame *frame_j) {
    return std::make_unique<Factor>(
        std::make_unique<PreIntegrationErrorCost>(frame_i, frame_j), factor_construct_t());
}

std::unique_ptr<Factor> Factor::create_marginalization_error(
    const Eigen::MatrixXd &sqrt_inv_cov, const Eigen::VectorXd &info_vec,
    std::vector<Frame *> &&frames) {
    return std::make_unique<Factor>(
        std::make_unique<MarginalizationErrorCost>(sqrt_inv_cov, info_vec, std::move(frames)),
        factor_construct_t());
}

Factor::Factor(std::unique_ptr<FactorCostFunction> cost_function, const factor_construct_t &)
    : cost_function(std::move(cost_function)) {}

Factor::~Factor() = default;
