//
// Created by lemon on 2021/1/21.
//


#include "../utils/read_file.h"
#include "read_bal.h"
#include <fstream>
#include "map.h"
#include <ceres/rotation.h>

BalReader::BalReader(std::string &path) {
    this->path = path;
    map = std::make_unique<Map>();
}

BalReader::BalReader() = default;

BalReader::~BalReader() = default;

bool BalReader::read_map() {
    std::cout << "read_map()..." << std::endl;
    std::ifstream fin(path);
    if (!fin.is_open()) {
        std::cout << path << " is opened!!! " << std::endl;
        return false;
    }

    int frame_num;
    int track_num;
    int observation_num;

    bool valid;
    std::stringstream ss = get_line_ss(fin,valid);
    ss >> frame_num >> track_num >> observation_num;
    map->frame_num = frame_num;
    map->track_num = track_num;
    map->observation_num = observation_num;
//    map->observation_vec.resize(observation_num);
    std::cout << "map->frame_num: " << map->frame_num << " map->track_num: " << map->track_num
              << " map->observation_num: " << map->observation_num << std::endl;
    for (int i = 0; i < observation_num; ++i) {
        ss = get_line_ss(fin,valid);
        int frame_id;
        int track_id;
        double x;
        double y;
        ss >> frame_id >> track_id >> x >> y;
        std::cout << i << " " << frame_id << " " << track_id << " " << x << " " << y << std::endl;
//        map->observation_vec[i] = std::make_unique<Observation>(i,frame_id,track_id,x,y);
        Observation *observation_curr = nullptr;
        Frame *frame_curr = nullptr;
        Track *track_curr = nullptr;
        if (map->observation_map.find(i) == map->observation_map.end()) {
            map->observation_map[i] = std::make_unique<Observation>(i, frame_id, track_id, x, y);
        }

        if (map->frame_map.find(frame_id) == map->frame_map.end()) {
            map->frame_map[frame_id] = std::make_unique<Frame>(frame_id);
        }
        frame_curr = map->frame_map[frame_id].get();

        if (map->track_map.find(track_id) == map->track_map.end()) {
            map->track_map[track_id] = std::make_unique<Track>(track_id);
        }
        track_curr = map->track_map[track_id].get();

        map->frame_to_observation_map[frame_id].emplace_back(i);
        map->track_to_observation_map[track_id].emplace_back(i);
    }
    std::cout << "observation over" << std::endl;
    for (int i = 0; i < frame_num; ++i) {
        auto *R_vec = new double[3];
        auto *q_vec = new double[4];
        Eigen::Vector3d t;
        Eigen::Vector3d C;
        Eigen::Matrix3d K;
        ss = get_line_ss(fin,valid);
        ss >> R_vec[0] >> R_vec[1] >> R_vec[2] >> t.x() >> t.y() >> t.z() >> C.x() >> C.y() >> C.z();
        map->frame_map[i].get()->f = C.x();
        map->frame_map[i].get()->k1 = C.y();
        map->frame_map[i].get()->k2 = C.z();
        ceres::AngleAxisToQuaternion(R_vec, q_vec);
        Eigen::Quaterniond q(q_vec[0], q_vec[1], q_vec[2], q_vec[3]);
        map->frame_map[i].get()->qwc = q;
        map->frame_map[i].get()->pwc = t;
    }

    for (int i = 0; i < track_num; i++) {
        Eigen::Vector3d point;
        ss = get_line_ss(fin,valid);
        ss >> point[0] >> point[1] >> point[2];
        map->track_map[i].get()->point_3D = point;
    }

    for (int i = 0; i < frame_num; ++i) {
        auto &frame_curr_observation_vec = map->frame_to_observation_map[i];
        int frame_curr_observation_vec_size = frame_curr_observation_vec.size();
        for (int j = 0; j < frame_curr_observation_vec_size; ++j) {
            auto &frame_observation_id = frame_curr_observation_vec[j];
            auto &track_id = map->observation_map[frame_observation_id]->track_id;
            auto &track_curr_observation_vec = map->track_to_observation_map[track_id];
            int track_curr_observation_vec_size = track_curr_observation_vec.size();
            for (int k = 0; k < track_curr_observation_vec_size; ++k) {
                auto &track_observation_id = track_curr_observation_vec[k];
                auto &observation = map->observation_map[track_observation_id];
                std::map<int, std::vector<std::pair<int, int>>> other_frame;
//                    map->frame_to_other_frame[i].clear();
                (map->frame_to_other_frame[i])[observation->frame_id].emplace_back(frame_observation_id, observation->obs_id);
            }
        }
    }

    std::cout << "read_map() over" << std::endl;
    return true;
}