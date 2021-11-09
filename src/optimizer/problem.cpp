//
// Created by lemon on 2021/1/21.
//

#include "problem.h"
#include "eigen_ba.h"
#include "parameter.h"
#include "projection_factor.h"

namespace fast_ba {
Problem::Problem(Map *map_, PROJECTION_TYPE projection_type_)
    : residual_blocks(map_->observation_num) {
    map = map_;
    projection_type = projection_type_;
    num_camera = map->frame_num;
    num_residual = map->observation_num;
    num_point = map->index_to_track_id_map.size();

    add_camera_param();
    add_point_param();
    add_residual_param();

    print_param_blocks();
}

Problem::~Problem() {
    for (int i = 0; i < param_vec.size(); i++) {
        delete param_vec[i];
    }
}


void Problem::print_param_blocks() {
    std::cout << std::endl;
    for (int camera_index_in_x_f = 0; camera_index_in_x_f < num_camera; ++camera_index_in_x_f) {
        int camera_id = camera_index_in_x_f;
        Eigen::Map<Eigen::Quaterniond> qcw(param_blocks[camera_index_in_x_f * 2].param_ptr);
        Eigen::Map<Eigen::Vector3d> pcw(param_blocks[camera_index_in_x_f * 2 + 1].param_ptr);
        std::cout << "camera_id: " << camera_id << ", qcw: " << qcw.coeffs().transpose()
                  << ", pcw: " << pcw.transpose() << std::endl;
    }
    std::cout << std::endl;
}

void Problem::add_camera_param() {
    for (int camera_index_in_x_f = 0; camera_index_in_x_f < num_camera; ++camera_index_in_x_f) {
        int frame_id = camera_index_in_x_f;
        Eigen::Quaterniond q = map->frame_map[frame_id]->qcw;
        double *param_q = new double[4] {q.x(), q.y(), q.z(), q.w()};
        param_vec.push_back(param_q);
        param_blocks.emplace_back(ParamBlock(param_vec.back(), 4, new QuatParam)); // x,y,z,w
        if (projection_type == PROJECTION_TYPE::CLASSICAL_PROJECTION) {
            Eigen::Vector3d p = map->frame_map[frame_id]->pcw;
            double *param_p = new double[3] {p.x(), p.y(), p.z()};
            param_vec.push_back(param_p);
            param_blocks.emplace_back(ParamBlock(param_vec.back(), 3));
        } else if (projection_type == PROJECTION_TYPE::SYM_PROJECTION) {
            Eigen::Vector3d p = map->frame_map[frame_id]->pcw;
            p = -(q.conjugate() * p);
            double *param_p = new double[3] {p.x(), p.y(), p.z()};
            param_vec.push_back(param_p);
            param_blocks.emplace_back(ParamBlock(param_vec.back(), 3));
        }
    }
}

void Problem::add_point_param() {
    for (int point_index_in_x_e = 0; point_index_in_x_e < num_point; ++point_index_in_x_e) {
        int track_id = map->index_to_track_id_map[point_index_in_x_e];
        Eigen::Vector3d point = map->track_map[track_id]->point_3D;
        double *param_point = new double[3] {point.x(), point.y(), point.z()};
        param_vec.push_back(param_point);
        param_blocks.emplace_back(ParamBlock(param_vec.back(), 3));
    }
}

void Problem::add_residual_param() {
    for (int residual_id = 0; residual_id < num_residual; ++residual_id) {
        residual_blocks[residual_id].residual_id = map->observation_map[residual_id]->obs_id;
        residual_blocks[residual_id].camera_id = map->observation_map[residual_id]->frame_id;
        residual_blocks[residual_id].point_id = map->observation_map[residual_id]->track_id;

        int camera_index_in_x_f = map->observation_map[residual_id]->frame_id;
        int point_index_in_x_e = map->track_id_to_index_map[residual_blocks[residual_id].point_id];

        residual_blocks[residual_id].add_paramblock(param_blocks[camera_index_in_x_f * 2]);
        residual_blocks[residual_id].add_paramblock(param_blocks[camera_index_in_x_f * 2 + 1]);
        residual_blocks[residual_id].add_paramblock(
            param_blocks[num_camera * 2 + point_index_in_x_e]);

        // residual_blocks[residual_id].factor = std::make_unique<ClassicalProjectFactor>(
        //     map->observation_map[residual_id]->feature->normalized);
        residual_blocks[residual_id].init(
            projection_type, map->observation_map[residual_id]->feature->normalized);
    }
}

void Problem::recover_param() {
    recover_carema_param();
    recover_point_param();
    print_param_blocks();
}

void Problem::recover_carema_param() {
    for (int camera_index_in_x_f = 0; camera_index_in_x_f < num_camera; ++camera_index_in_x_f) {
        int camera_id = camera_index_in_x_f;
        Eigen::Map<Eigen::Quaterniond> qcw(param_blocks[camera_index_in_x_f * 2].param_ptr);
        map->frame_map[camera_id]->qcw = qcw;
        Eigen::Map<Eigen::Vector3d> pcw(param_blocks[camera_index_in_x_f * 2 + 1].param_ptr);
        if (projection_type == PROJECTION_TYPE::CLASSICAL_PROJECTION) {
            map->frame_map[camera_id]->pcw = pcw;
        } else if (projection_type == PROJECTION_TYPE::SYM_PROJECTION) {
            pcw = -(qcw * pcw);
            map->frame_map[camera_id]->pcw = pcw;
        }
    }
}

void Problem::recover_point_param() {
    for (int point_index_in_x_e = 0; point_index_in_x_e < num_point; ++point_index_in_x_e) {
        int track_id = map->index_to_track_id_map[point_index_in_x_e];
        Eigen::Map<Eigen::Vector3d> x_w(
            param_blocks[num_camera * 2 + point_index_in_x_e].param_ptr);
        map->track_map[track_id]->point_3D = x_w;
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


} // namespace fast_ba