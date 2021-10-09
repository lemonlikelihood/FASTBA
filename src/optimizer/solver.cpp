//
// Created by lemon on 2021/3/4.
//

#include "solver.h"

namespace fast_ba {
Solver::Solver(fast_ba::Problem *problem_) {
    problem = problem_;
    x_f.resize(camera_size * problem->num_camera);
    x_e.resize(point_size * problem->num_point);

    lhs.resize(camera_size * problem->num_camera, camera_size * problem->num_camera);
    rhs.resize(camera_size * problem->num_camera);
    lhs.setZero();
    rhs.setZero();

    etes.resize(problem->num_point);
    ftfs.resize(problem->num_camera);
    ftbs.resize(problem->num_camera);
    etes_inverse.resize(problem->num_point);
    etbs.resize(problem->num_point);
    etfs.resize(problem->num_residual);
}

void Solver::schur_complement() {
    // Ete_factor ete_diag = Ete_factor::Zero();
    // ete_diag.diagonal().setConstant(1.0 / radius);

    // for (int point_index_in_x_e = 0; point_index_in_x_e < problem->num_point;
    //      ++point_index_in_x_e) {
    //     int track_id = problem->map->index_to_track_id_map[point_index_in_x_e];
    //     auto &track_curr_observation_vec = problem->map->track_to_observation_map[track_id];
    //     int track_curr_observation_vec_size = track_curr_observation_vec.size();
    //     Ete_factor ete = Ete_factor::Zero();
    //     Etb_factor etb = Etb_factor::Zero();
    //     for (int j = 0; j < track_curr_observation_vec_size; ++j) {
    //         int residual_id = track_curr_observation_vec[j];
    //         ete.noalias() += problem->residual_blocks[residual_id].ete_factor;
    //         etb.noalias() += problem->residual_blocks[residual_id].etb_factor;
    //     }
    //     etes[point_index_in_x_e] = ete + ete_diag;
    //     etbs[point_index_in_x_e] = etb;
    // }

    // Ftf_factor ftf_diag = Ftf_factor::Zero();
    // ftf_diag.diagonal().setConstant(1.0 / radius);

    // for (int camera_index_in_x_f = 1; camera_index_in_x_f < problem->num_camera;
    //      ++camera_index_in_x_f) {
    //     Ftf_factor ftf = Ftf_factor::Zero();
    //     Ftb_factor ftb = Ftb_factor::Zero();
    //     int camera_id = camera_index_in_x_f;
    //     auto &frame_curr_observation_vec = problem->map->frame_to_observation_map[camera_id];
    //     int frame_curr_observation_vec_size = frame_curr_observation_vec.size();
    //     for (int j = 0; j < frame_curr_observation_vec_size; ++j) {
    //         int residual_id = frame_curr_observation_vec[j];
    //         ftf.noalias() += problem->residual_blocks[residual_id].ftf_factor;
    //         ftb.noalias() += problem->residual_blocks[residual_id].ftb_factor;
    //     }
    //     ftfs[camera_index_in_x_f] = ftf + ftf_diag;
    //     ftbs[camera_index_in_x_f] = ftb;
    // }

    for (int point_index_in_x_e = 0; point_index_in_x_e < problem->num_point;
         ++point_index_in_x_e) {
        etes_inverse[point_index_in_x_e].noalias() = etes[point_index_in_x_e].inverse();
    }

    for (int camera1_index_in_x_f = 0; camera1_index_in_x_f < problem->num_camera;
         ++camera1_index_in_x_f) {
        auto &frame_curr_observation_vec =
            problem->map->frame_to_observation_map[camera1_index_in_x_f];
        Ftb_factor tmpv = Ftb_factor::Zero();
        for (auto residual_id : frame_curr_observation_vec) {
            auto &residual = problem->residual_blocks[residual_id];
            int point_index_in_x_e = problem->map->track_id_to_index_map[residual.point_id];
            tmpv.noalias() -= residual.etf_factor.transpose() * etes_inverse[point_index_in_x_e]
                              * etbs[point_index_in_x_e];
        }
        rhs.segment<6>(camera1_index_in_x_f * 6).noalias() = tmpv;

        auto &cov_frame_map = problem->map->frame_to_other_frame[camera1_index_in_x_f];
        for (auto &frame : cov_frame_map) {
            int camera2_index_in_x_f = frame.first;
            auto &cov_track_vec = cov_frame_map[camera2_index_in_x_f];
            Ftf_factor tmp = Ftf_factor::Zero();
            for (int j = 0; j < cov_track_vec.size(); ++j) {
                int residual_id1 = cov_track_vec[j].first;
                int residual_id2 = cov_track_vec[j].second;
                auto &residual_first = problem->residual_blocks[residual_id1];
                auto &residual_second = problem->residual_blocks[residual_id2];
                int point_index_in_x_e =
                    problem->map->track_id_to_index_map[residual_first.point_id];
                tmp.noalias() -= residual_first.etf_factor.transpose()
                                 * etes_inverse[point_index_in_x_e] * residual_second.etf_factor;
            }
            lhs.block<6, 6>(camera1_index_in_x_f * 6, camera2_index_in_x_f * 6).noalias() = tmp;
        }
    }

    for (int camera_index_in_x_f = 0; camera_index_in_x_f < problem->num_camera;
         ++camera_index_in_x_f) {
        lhs.block<6, 6>(camera_index_in_x_f * 6, camera_index_in_x_f * 6).noalias() +=
            ftfs[camera_index_in_x_f];
        rhs.segment<6>(camera_index_in_x_f * 6).noalias() += ftbs[camera_index_in_x_f];
    }
}

void Solver::compute_delta() {
    // std::cout << "compute_delta ..." << std::endl;
    x_f.setZero();
    x_e.setZero();

    int num_block_effective = 6 * (problem->num_camera - 1);
    Eigen::MatrixXd lhs_effective = lhs.bottomRightCorner(num_block_effective, num_block_effective);
    Eigen::MatrixXd rhs_effective = rhs.tail(num_block_effective);
    Eigen::LLT<Eigen::MatrixXd, Eigen::Upper> llt =
        lhs_effective.selfadjointView<Eigen::Upper>().llt();
    x_f.tail(num_block_effective) = llt.solve(rhs_effective);

    for (int point_index_in_x_e = 0; point_index_in_x_e < problem->num_point;
         ++point_index_in_x_e) {
        Eigen::Vector3d tmp = etbs[point_index_in_x_e];
        int point_id = problem->map->index_to_track_id_map[point_index_in_x_e];
        auto &track_curr_observation_vec = problem->map->track_to_observation_map[point_id];
        for (auto residual_id : track_curr_observation_vec) {
            auto &residual = problem->residual_blocks[residual_id];
            int camera_index_in_x_f = residual.camera_id;
            if (camera_index_in_x_f != 0)
                tmp -= residual.etf_factor * x_f.block<6, 1>(6 * camera_index_in_x_f, 0);
        }
        x_e.block<3, 1>(3 * point_index_in_x_e, 0) = etes_inverse[point_index_in_x_e] * tmp;
    }
    x_f = -x_f;
    x_e = -x_e;
}

void Solver::param_plus_delta() {
    for (int camera_index_in_x_f = 0; camera_index_in_x_f < problem->num_camera;
         ++camera_index_in_x_f) { //fix camera 0
        if (camera_index_in_x_f == 0) {
            continue;
        }
        double *delta_x = x_f.block<6, 1>(6 * camera_index_in_x_f, 0).data();
        problem->param_blocks[camera_index_in_x_f * 2].plus_delta(delta_x);
        problem->param_blocks[camera_index_in_x_f * 2 + 1].plus_delta(delta_x + 3);
    }
    for (int point_index_in_x_e = 0; point_index_in_x_e < problem->num_point;
         ++point_index_in_x_e) {
        double *delta_x = x_e.block<3, 1>(3 * point_index_in_x_e, 0).data();
        problem->param_blocks[problem->num_camera * 2 + point_index_in_x_e].plus_delta(delta_x);
    }
}

void Solver::param_update() {
    for (int camera_index_in_x_f = 0; camera_index_in_x_f < problem->num_camera;
         ++camera_index_in_x_f) {
        if (camera_index_in_x_f == 0)
            continue; //fix camera 0
        problem->param_blocks[camera_index_in_x_f * 2].update();
        problem->param_blocks[camera_index_in_x_f * 2 + 1].update();
    }
    for (int point_index_in_x_e = 0; point_index_in_x_e < problem->num_point;
         ++point_index_in_x_e) {
        problem->param_blocks[problem->num_camera * 2 + point_index_in_x_e].update();
    }
}

void Solver::param_recover() {
    // std::cout << "param_recover ..." << std::endl;
    for (int camera_index_in_x_f = 0; camera_index_in_x_f < problem->num_camera;
         ++camera_index_in_x_f) {
        int camera_id = camera_index_in_x_f;
        Eigen::Map<Eigen::Quaterniond> qcw(
            problem->param_blocks[camera_index_in_x_f * 2].param_ptr);
        problem->map->frame_map[camera_id]->qcw = qcw;
        Eigen::Map<Eigen::Vector3d> pcw(
            problem->param_blocks[camera_index_in_x_f * 2 + 1].param_ptr);
        problem->map->frame_map[camera_id]->pcw = pcw;
    }
    for (int point_index_in_x_e = 0; point_index_in_x_e < problem->num_point;
         ++point_index_in_x_e) {
        int track_id = problem->map->index_to_track_id_map[point_index_in_x_e];
        Eigen::Map<Eigen::Vector3d> x_w(
            problem->param_blocks[problem->num_camera * 2 + point_index_in_x_e].param_ptr);
        problem->map->track_map[track_id]->point_3D = x_w;
    }
    // std::cout << "param_recover end " << std::endl;
}

void Solver::compute_model_cost_change() {
    // new_model_cost
    //  = 1/2 [f + J * step]^2
    //  = 1/2 [ f'f + 2f'J * step + step' * J' * J * step ]
    // model_cost_change
    //  = cost - new_model_cost
    //  = f'f/2  - 1/2 [ f'f + 2f'J * step + step' * J' * J * step]
    //  = -f'J * step - step' * J' * J * step / 2
    //  = -(J * step)'(f + J * step / 2)
    Eigen::VectorXd model_residuals(residual_size * problem->num_residual);
    Eigen::VectorXd tmp(residual_size * problem->num_residual);
    for (int residual_id = 0; residual_id < problem->num_residual; ++residual_id) {
        auto &residual = problem->residual_blocks[residual_id];
        int camera_index_in_x_f = residual.camera_id;
        int point_index_in_x_e = problem->map->track_id_to_index_map[residual.point_id];
        model_residuals.segment<residual_size>(residual_id * residual_size) =
            residual.jac_point_factor * x_e.segment<3>(3 * point_index_in_x_e);
        if (camera_index_in_x_f != 0)
            model_residuals.segment<residual_size>(residual_id * residual_size) +=
                residual.jac_camera_factor * x_f.segment<6>(6 * camera_index_in_x_f);
        tmp.segment<residual_size>(residual_id * residual_size) =
            model_residuals.segment<residual_size>(residual_id * residual_size) / 2
            + residual.residual_factor;
    }

    // std::cout<<"model: "<<model_residuals<<std::endl;
    model_cost_change = -model_residuals.dot(tmp);
    // std::cout << "model cost change: " << model_cost_change << std::endl;
}

void Solver::compute_cost(double &cost) {
    cost = 0;
    for (int residual_id = 0; residual_id < problem->num_residual; ++residual_id) {
        cost += problem->residual_blocks[residual_id].residual_factor.squaredNorm();
    }
    cost /= 2;
}

void Solver::compute_gradient_norm() {
    Ete_factor ete_diag = Ete_factor::Zero();
    ete_diag.diagonal().setConstant(1.0 / radius);

    for (int point_index_in_x_e = 0; point_index_in_x_e < problem->num_point;
         ++point_index_in_x_e) {
        int track_id = problem->map->index_to_track_id_map[point_index_in_x_e];
        auto &track_curr_observation_vec = problem->map->track_to_observation_map[track_id];
        int track_curr_observation_vec_size = track_curr_observation_vec.size();
        Ete_factor ete = Ete_factor::Zero();
        Etb_factor etb = Etb_factor::Zero();
        for (int j = 0; j < track_curr_observation_vec_size; ++j) {
            int residual_id = track_curr_observation_vec[j];
            ete.noalias() += problem->residual_blocks[residual_id].ete_factor;
            etb.noalias() += problem->residual_blocks[residual_id].etb_factor;
        }
        etes[point_index_in_x_e] = ete + ete_diag;
        etbs[point_index_in_x_e] = etb;
    }

    Ftf_factor ftf_diag = Ftf_factor::Zero();
    ftf_diag.diagonal().setConstant(1.0 / radius);

    for (int camera_index_in_x_f = 1; camera_index_in_x_f < problem->num_camera;
         ++camera_index_in_x_f) {
        Ftf_factor ftf = Ftf_factor::Zero();
        Ftb_factor ftb = Ftb_factor::Zero();
        int camera_id = camera_index_in_x_f;
        auto &frame_curr_observation_vec = problem->map->frame_to_observation_map[camera_id];
        int frame_curr_observation_vec_size = frame_curr_observation_vec.size();
        for (int j = 0; j < frame_curr_observation_vec_size; ++j) {
            int residual_id = frame_curr_observation_vec[j];
            ftf.noalias() += problem->residual_blocks[residual_id].ftf_factor;
            ftb.noalias() += problem->residual_blocks[residual_id].ftb_factor;
        }
        ftfs[camera_index_in_x_f] = ftf + ftf_diag;
        ftbs[camera_index_in_x_f] = ftb;
    }

    Eigen::VectorXd gradient;
    gradient.resize(camera_real_size * problem->num_camera + point_size * problem->num_point);
    gradient.setZero();

    for (int camera_index_in_x_f = 0; camera_index_in_x_f < problem->num_camera;
         ++camera_index_in_x_f) { //fix camera 0
        if (camera_index_in_x_f == 0) {
            continue;
        }
        Ftb_factor negative_delta = -ftbs[camera_index_in_x_f];
        double *delta_x = negative_delta.data();
        problem->param_blocks[camera_index_in_x_f * 2].plus_delta(delta_x);
        problem->param_blocks[camera_index_in_x_f * 2].gradient_decrease();
        for (int i = 0; i < quaternion_size; i++) {
            gradient[camera_real_size * camera_index_in_x_f + i] =
                problem->param_blocks[camera_index_in_x_f * 2].param_new[i];
        }
        problem->param_blocks[camera_index_in_x_f * 2 + 1].plus_delta(delta_x + 3);
        problem->param_blocks[camera_index_in_x_f * 2 + 1].gradient_decrease();
        for (int i = 0; i < translation_size; i++) {
            gradient[camera_real_size * camera_index_in_x_f + quaternion_size + i] =
                problem->param_blocks[camera_index_in_x_f * 2 + 1].param_new[i];
        }
    }

    for (int point_index_in_x_e = 0; point_index_in_x_e < problem->num_point;
         ++point_index_in_x_e) {
        Etb_factor negative_delta = -etbs[point_index_in_x_e];
        double *delta_x = negative_delta.data();
        problem->param_blocks[problem->num_camera * 2 + point_index_in_x_e].plus_delta(delta_x);
        problem->param_blocks[problem->num_camera * 2 + point_index_in_x_e].gradient_decrease();
        for (int i = 0; i < point_size; i++) {
            gradient[camera_real_size * problem->num_camera + point_index_in_x_e * point_size + i] =
                problem->param_blocks[problem->num_camera * 2 + point_index_in_x_e].param_new[i];
        }
    }
    // std::cout << "gradient: " << gradient[3] << std::endl;

    gradient_max_norm = gradient.lpNorm<Eigen::Infinity>();
}

void Solver::compute_cost_change() {
    // std::cout << "compute_cost_change ..." << std::endl;
    compute_cost(current_cost);
    // std::cout << "current_cost: " << current_cost << std::endl;
    param_plus_delta(); // compute candidate param
    problem->evaluate_candidate();
    compute_cost(candidate_cost);
    // std::cout << "candidate_cost: " << candidate_cost << std::endl;
    cost_change = current_cost - candidate_cost;
    // std::cout << "compute_cost_change end" << std::endl;
}

void Solver::compute_relative_decrease() {
    // std::cout << "compute_relative_decrease ..." << std::endl;
    // compute relative_decrease
    //        compute_cost(current_cost);
    //        param_plus_delta();// compute candidate param
    //        problem->evaluate_candidate();
    //        compute_cost(candidate_cost);
    // compute_gradient_norm();
    compute_model_cost_change();
    compute_cost_change();
    relative_decrease = cost_change / model_cost_change; //TODO reference_cost
    // std::cout << "cost change: " << cost_change << std::endl;
    // std::cout << "model_cost_change: " << model_cost_change << std::endl;
    // std::cout << "relative_decrease: " << relative_decrease << std::endl;
    // std::cout << "compute_relative_decrease end" << std::endl;
}

void Solver::compute_step_norm_x_norm(double &step_norm, double &x_norm) {
    step_norm = x_norm = 0;
    for (int camera_index_in_x_f = 1; camera_index_in_x_f < problem->num_camera;
         ++camera_index_in_x_f) { //fix camera0
        step_norm += problem->param_blocks[camera_index_in_x_f * 2].step_square_norm();
        step_norm += problem->param_blocks[camera_index_in_x_f * 2 + 1].step_square_norm();
        x_norm += problem->param_blocks[camera_index_in_x_f * 2].x_square_norm();
        x_norm += problem->param_blocks[camera_index_in_x_f * 2 + 1].x_square_norm();
    }
    for (int point_index_in_x_e = 0; point_index_in_x_e < problem->num_point;
         ++point_index_in_x_e) {
        step_norm +=
            problem->param_blocks[problem->num_camera * 2 + point_index_in_x_e].step_square_norm();
        x_norm +=
            problem->param_blocks[problem->num_camera * 2 + point_index_in_x_e].x_square_norm();
    }
    step_norm = sqrt(step_norm);
    x_norm = sqrt(x_norm);
}

void Solver::step_accepted(double step_quality) {
    radius = radius / std::max(1.0 / 3.0, 1.0 - std::pow(2.0 * step_quality - 1.0, 3));
    radius = std::min(max_radius, radius);
    // radius = std::max(min_radius, radius);
    decrease_factor = 2.0;
}

void Solver::step_rejected(double step_quality) {
    radius = radius / decrease_factor;
    // radius = std::min(max_radius, radius);
    // radius = std::max(min_radius, radius);
    decrease_factor *= 2.0;
}

void Solver::solve_linear_problem() {
    schur_complement();
    compute_delta();
}
} // namespace fast_ba