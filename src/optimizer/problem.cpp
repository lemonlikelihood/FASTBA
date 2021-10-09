//
// Created by lemon on 2021/1/21.
//

#include "problem.h"
#include "eigen_ba.h"
#include "projection_factor.h"
#include "parameter.h"

namespace fast_ba {
    Problem::Problem(Map *_map) : residual_blocks(_map->observation_num) {
        map = _map;
        num_camera = map->frame_num;
        num_residual = map->observation_num;
        num_point = map->index_to_track_id_map.size();

//        camera_to_observation_id_vec.resize(num_camera);
//        point_to_observation_id_vec.resize(num_point);

//        x_f.resize(camera_size * num_camera);
//        x_e.resize(point_size * num_point);
//
//        lhs.resize(num_camera * camera_size, num_camera * camera_size);
//        rhs.resize(num_camera * camera_size);
//        lhs.setZero();
//        rhs.setZero();

        for (int camera_index_in_x_f = 0; camera_index_in_x_f < num_camera; ++camera_index_in_x_f) {
            int frame_id = camera_index_in_x_f;
            param_blocks.emplace_back(ParamBlock(map->frame_map[frame_id]->qcw.coeffs().data(), 4, new QuatParam));   // x,y,z,w
            param_blocks.emplace_back(ParamBlock(map->frame_map[frame_id]->pcw.data(), 3));
        }

        for (int point_index_in_x_e = 0; point_index_in_x_e < num_point; ++point_index_in_x_e) {
            int track_id = map->index_to_track_id_map[point_index_in_x_e];
            param_blocks.emplace_back(ParamBlock(map->track_map[track_id]->point_3D.data(), 3));
        }

        for (int residual_id = 0; residual_id < num_residual; ++residual_id) {

            residual_blocks[residual_id].residual_id = map->observation_map[residual_id]->obs_id;
            residual_blocks[residual_id].camera_id = map->observation_map[residual_id]->frame_id;
            residual_blocks[residual_id].point_id = map->observation_map[residual_id]->track_id;
//            residual_blocks[residual_id] = ResidualBlock(map->observation_map[residual_id]->obs_id,
//                    map->observation_map[residual_id]->frame_id,map->observation_map[residual_id]->track_id);
            residual_blocks[residual_id].factor = std::make_unique<ProjectionFactor>(
                    map->observation_map[residual_id]->feature->normalized);
            residual_blocks[residual_id].factor->measurement = map->observation_map[residual_id]->feature->normalized;

            int camera_index_in_x_f = map->observation_map[residual_id]->frame_id;
            int point_index_in_x_e = map->track_id_to_index_map[residual_blocks[residual_id].point_id];

            residual_blocks[residual_id].add_paramblock(param_blocks[camera_index_in_x_f * 2]);
            residual_blocks[residual_id].add_paramblock(param_blocks[camera_index_in_x_f * 2 + 1]);
            residual_blocks[residual_id].add_paramblock(param_blocks[num_camera * 2 + point_index_in_x_e]);

            residual_blocks[residual_id].init();
        }

    }


    bool Problem::evaluate() {
        // std::cout<<"problem evaluate ... "<<std::endl;
        for (int residual_id = 0; residual_id < num_residual; ++residual_id) {
            residual_blocks[residual_id].evaluate();
        }
        // std::cout<<"problem evaluate end ... "<<std::endl;
        return true;
    }

    bool Problem::evaluate_candidate() {
        for (int residual_id = 0; residual_id < num_residual; ++residual_id) {
            residual_blocks[residual_id].evaluate_candidate();
        }
        return true;
    }

//    void Problem::schur_complement() {
//        Ete_factor ete_diag = Ete_factor::Zero();
//        ete_diag.diagonal().setConstant(1.0 / radius);
//        for (int point_index_in_x_e = 0; point_index_in_x_e < num_point; ++point_index_in_x_e) {
//            int track_id = map->index_to_track_id_map[point_index_in_x_e];
//            Ete_factor ete = Ete_factor::Zero();
//            Etb_factor etb = Etb_factor::Zero();
//            auto &track_curr_observation_vec = map->track_to_observation_map[track_id];
//            int track_curr_observation_vec_size = track_curr_observation_vec.size();
//            for (int j = 0; j < track_curr_observation_vec_size; ++j) {
//                int residual_id = track_curr_observation_vec[j];
//                ete.noalias() += residual_blocks[residual_id].ete_factor;
//                etb.noalias() += residual_blocks[residual_id].etb_factor;
//            }
//            etes[point_index_in_x_e] = ete + ete_diag;
//            etbs[point_index_in_x_e] = etb;
//        }
//
//        Ftf_factor ftf_diag = Ftf_factor::Zero();
//        ftf_diag.diagonal().setConstant(1.0 / radius);
//        for (int camera_index_in_x_f = 0; camera_index_in_x_f < num_camera; ++camera_index_in_x_f) {
//            Ftf_factor ftf = Ftf_factor::Zero();
//            Ftb_factor ftb = Ftb_factor::Zero();
//            auto &frame_curr_observation_vec = map->frame_to_observation_map[camera_index_in_x_f];
//            int frame_curr_observation_vec_size = frame_curr_observation_vec.size();
//            for (int j = 0; j < frame_curr_observation_vec_size; ++j) {
//                int residual_id = frame_curr_observation_vec[j];
//                ftf.noalias() += residual_blocks[residual_id].ftf_factor;
//                ftb.noalias() += residual_blocks[residual_id].ftb_factor;
//            }
//            ftfs[camera_index_in_x_f] = ftf + ftf_diag;
//            ftbs[camera_index_in_x_f] = ftb;
//        }
//
//        for (int point_index_in_x_e = 0; point_index_in_x_e < num_point; ++point_index_in_x_e) {
//            etes_inverse[point_index_in_x_e].noalias() = etes[point_index_in_x_e].inverse();
//        }
//
//        for (int camera1_index_in_x_f = 0; camera1_index_in_x_f < num_camera; ++camera1_index_in_x_f) {
//            auto &cov_frame_map = map->frame_to_other_frame[camera1_index_in_x_f];
//            auto &frame_curr_observation_vec = map->frame_to_observation_map[camera1_index_in_x_f];
//            Ftb_factor tmpv = Ftb_factor::Zero();
//            for (auto residual_id:frame_curr_observation_vec) {
//                auto &residual = residual_blocks[residual_id];
//                int point_index_in_x_e = map->track_id_to_index_map[residual.point_id];
//                tmpv.noalias() +=
//                        residual.etf_factor.transpose() * etes_inverse[point_index_in_x_e] * etbs[point_index_in_x_e];
//            }
//            rhs.segment<6>(camera1_index_in_x_f * 6).noalias() = tmpv;
//
//            for (auto &frame:cov_frame_map) {
//                int camera2_index_in_x_f= frame.first;
//                auto &cov_track_vec = cov_frame_map[camera2_index_in_x_f];
//                Ftf_factor tmp = Ftf_factor::Zero();
//                for (int j = 0; j < cov_track_vec.size(); ++j) {
//                    int residual_id1 = cov_track_vec[j].first;
//                    int residual_id2 = cov_track_vec[j].second;
//                    auto &residual_first = residual_blocks[residual_id1];
//                    auto &residual_second = residual_blocks[residual_id2];
//                    int point_index_in_x_e1 = map->track_id_to_index_map[residual_first.point_id];
//                    int point_index_in_x_e2 = map->track_id_to_index_map[residual_second.point_id];
//                    tmp.noalias() += residual_first.etf_factor.transpose() * etes_inverse[point_index_in_x_e1] *
//                                     residual_second.etf_factor;
//                }
//                lhs.block<6, 6>(camera1_index_in_x_f * 6, camera2_index_in_x_f * 6).noalias() = tmp;
//            }
//        }
//
//        for (int camera_index_in_x_f = 0; camera_index_in_x_f < num_camera; ++camera_index_in_x_f) {
//            lhs.block<6, 6>(camera_index_in_x_f * 6, camera_index_in_x_f * 6).noalias() =
//                    ftfs[camera_index_in_x_f] - lhs.block<6, 6>(camera_index_in_x_f * 6, camera_index_in_x_f * 6);
//            rhs.segment<6>(camera_index_in_x_f * 6).noalias() = ftbs[camera_index_in_x_f] - rhs.segment<6>(camera_index_in_x_f * 6);
//        }
//    }

//    void Problem::compute_delta() {
//        x_f.setZero();
//        x_e.setZero();
//
//        int num_block_effective = 6 * (num_camera - 1);
//        Eigen::MatrixXd lhs_effective = lhs.bottomRightCorner(num_block_effective, num_block_effective);
//        Eigen::MatrixXd rhs_effective = rhs.tail(num_block_effective);
//        Eigen::LLT<Eigen::MatrixXd, Eigen::Upper> llt = lhs_effective.selfadjointView<Eigen::Upper>().llt();
//        x_f.tail(num_block_effective) = llt.solve(rhs_effective);
//
//        for (int point_index_in_x_e = 0; point_index_in_x_e < num_point; ++point_index_in_x_e) {
//            Eigen::Vector3d tmp = etbs[point_index_in_x_e];
//            int point_id = map->index_to_track_id_map[point_index_in_x_e];
//            auto &track_curr_observation_vec = map->track_to_observation_map[point_id];
//            for (auto residual_id:track_curr_observation_vec) {
//                auto &residual = residual_blocks[residual_id];
//                int camera_index_in_x_f = residual.camera_id;
//                tmp -= residual.etf_factor * x_f.block<6, 1>(6 * camera_index_in_x_f, 0);
//            }
//            x_e.block<3, 1>(3 * point_index_in_x_e, 0) = etes_inverse[point_index_in_x_e] * tmp;
//        }
//        x_f = -x_f;
//        x_e = -x_e;
//    }

//    void Problem::param_plus_delta() {
//        for (int camera_index_in_x_f = 1; camera_index_in_x_f < num_camera; ++camera_index_in_x_f) {//fix camera 0
//            double *delta_x = x_f.block<6, 1>(6 * camera_index_in_x_f, 0).data();
//            param_blocks[camera_index_in_x_f * 2].plus_delta(delta_x);
//            param_blocks[camera_index_in_x_f * 2 + 1].plus_delta(delta_x + 3);
//        }
//        for (int point_index_in_x_e = 0; point_index_in_x_e < num_point; ++point_index_in_x_e) {
//            double *delta_x = x_e.block<3, 1>(3 * point_index_in_x_e, 0).data();
//            param_blocks[num_camera * 2 + point_index_in_x_e].plus_delta(delta_x);
//        }
//    }

//    void Problem::param_update() {
//        for (int camera_index_in_x_f = 0; camera_index_in_x_f < num_camera; ++camera_index_in_x_f) {
//            if (camera_index_in_x_f == 0)continue;//fix camera 0
//            param_blocks[camera_index_in_x_f * 2].update();
//            param_blocks[camera_index_in_x_f * 2 + 1].update();
//        }
//        for (int point_index_in_x_e = 0; point_index_in_x_e < num_point; ++point_index_in_x_e) {
//            param_blocks[num_camera * 2 + point_index_in_x_e].update();
//        }
//    }

//    void Problem::compute_model_cost_change() {
//        // new_model_cost
//        //  = 1/2 [f + J * step]^2
//        //  = 1/2 [ f'f + 2f'J * step + step' * J' * J * step ]
//        // model_cost_change
//        //  = cost - new_model_cost
//        //  = f'f/2  - 1/2 [ f'f + 2f'J * step + step' * J' * J * step]
//        //  = -f'J * step - step' * J' * J * step / 2
//        //  = -(J * step)'(f + J * step / 2)
//        Eigen::VectorXd model_residuals(residual_size * num_residual);
//        Eigen::VectorXd tmp(residual_size * num_residual);
//        for (int residual_id = 0; residual_id < num_residual; ++residual_id) {
//            auto &residual = residual_blocks[residual_id];
//            int camera_index_in_x_f = residual.camera_id;
//            int point_index_in_x_e = map->track_id_to_index_map[residual.point_id];
//            model_residuals.segment<residual_size>(residual_id * residual_size) =
//                    residual.jac_camera_factor * x_f.segment<6>(6 * camera_index_in_x_f) +
//                    residual.jac_point_factor * x_e.segment<3>(3 * point_index_in_x_e);
//            tmp.segment<residual_size>(residual_id * residual_size) =
//                    model_residuals.segment<residual_size>(residual_id * residual_size) / 2 + residual.residual_factor;
//        }
//
//        model_cost_change = -model_residuals.dot(tmp);
//        std::cout << "model cost change: " << model_cost_change << std::endl;
//    }

//    void Problem::compute_cost(double &cost) {
//        cost = 0;
//        for (int residual_id = 0; residual_id < num_residual; ++residual_id) {
//            cost += residual_blocks[residual_id].residual_factor.squaredNorm();
//        }
//        cost /= 2;
//    }

//    void Problem::compute_relative_decrease() {
//        // compute relative_decrease
//        compute_cost(current_cost);
//        param_plus_delta();// compute candidate param
//        evaluate_candidate();
//        compute_cost(candidate_cost);
//        relative_decrease = (current_cost - candidate_cost) / model_cost_change;//TODO reference_cost
//    }
//
//    void Problem::compute_step_norm_x_norm(double &step_norm, double &x_norm) {
//        step_norm = x_norm = 0;
//        for (int camera_index_in_x_f = 1; camera_index_in_x_f < num_camera; ++camera_index_in_x_f) {//fix camera0
//            step_norm += param_blocks[camera_index_in_x_f * 2].step_square_norm();
//            step_norm += param_blocks[camera_index_in_x_f * 2 + 1].step_square_norm();
//            x_norm += param_blocks[camera_index_in_x_f * 2].x_square_norm();
//            x_norm += param_blocks[camera_index_in_x_f * 2 + 1].x_square_norm();
//        }
//        step_norm = sqrt(step_norm);
//        x_norm = sqrt(x_norm);
//    }
}