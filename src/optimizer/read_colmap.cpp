//
// Created by lemon on 2021/1/21.
//

#include "read_colmap.h"
#include "../utils/read_file.h"
#include "../utils/tic_toc.h"
#include <iostream>

#include "map.h"
#include <fstream>

ColmapReader::ColmapReader(std::string &path) {
    this->path = path;
    map = std::make_unique<Map>();
}

ColmapReader::~ColmapReader() {}

void ColmapReader::read_cameras(const std::string &cameras_file) {
    std::ifstream fin(cameras_file);
    if (!fin.is_open()) {
        std::cout << "camera.txt is not existed!";
        return;
    }
    std::cout << "Loading " << cameras_file << " ..." << std::endl;
    std::string s;

    std::stringstream ss;
    bool valid;
    int line_num = 0;
    ss = get_line_ss(fin, valid);
    while (valid) {
        if (line_num > 2) {
            int frame_id;
            int width;
            int height;
            std::string camera_model;
            double camera_params[4];
            ss >> frame_id >> camera_model >> width >> height >> camera_params[0]
                >> camera_params[1] >> camera_params[2] >> camera_params[3];
            map->frame_map[frame_id - 1] = std::make_unique<Frame>(frame_id - 1);
            map->frame_map[frame_id - 1]->camera_model = camera_model;
            map->frame_map[frame_id - 1]->width = width;
            map->frame_map[frame_id - 1]->height = height;
            map->frame_map[frame_id - 1]->f = camera_params[0];
            map->frame_map[frame_id - 1]->cx = camera_params[1];
            map->frame_map[frame_id - 1]->cy = camera_params[2];
        }
        line_num++;
        std::cout << line_num << std::endl;
        ss = get_line_ss(fin, valid);
    }
}

void ColmapReader::read_images(const std::string &images_file) {
    std::ifstream fin(images_file);
    if (!fin.is_open()) {
        std::cout << "images.txt is not existed!";
        return;
    }
    std::cout << "Loading " << images_file << " ..." << std::endl;
    std::string s;

    //    Eigen::Quaterniond qcw;
    //    Eigen::Vector3d tcw;
    Eigen::Vector2d keypoint;
    int point3d_index = 0;
    int line_num = 0;
    std::stringstream ss;
    bool valid;
    ss = get_line_ss(fin, valid);
    while (valid) {
        if (line_num > 3) {
            int frame_id;
            ss >> frame_id;
            ss >> map->frame_map[frame_id - 1]->qcw.w() >> map->frame_map[frame_id - 1]->qcw.x()
                >> map->frame_map[frame_id - 1]->qcw.y() >> map->frame_map[frame_id - 1]->qcw.z();

            ss >> map->frame_map[frame_id - 1]->pcw.x() >> map->frame_map[frame_id - 1]->pcw.y()
                >> map->frame_map[frame_id - 1]->pcw.z();
            ss >> map->frame_map[frame_id - 1]->camera_id;
            ss >> map->frame_map[frame_id - 1]->image_name;
            ss = get_line_ss(fin, valid);
            while (ss) {
                ss >> keypoint(0) >> keypoint(1);
                ss >> point3d_index;
                map->frame_map[frame_id - 1]->keypoints.emplace_back(keypoint);
                map->frame_map[frame_id - 1]->track_ids.emplace_back(point3d_index);
            }
        }
        line_num++;
        ss = get_line_ss(fin, valid);
    }
}

void ColmapReader::read_points_3D(const std::string &points3D_file) {
    std::ifstream fin(points3D_file);
    if (!fin.is_open()) {
        std::cout << "points3D.txt is not existed!";
        return;
    }
    std::cout << "Loading " << points3D_file << " ..." << std::endl;
    std::string s;

    int image_id;
    int point2d_index = 0;
    int line_num = 0;
    int point_index_in_map = 0;
    std::stringstream ss;
    bool valid;
    ss = get_line_ss(fin, valid);
    while (valid) {
        if (line_num > 2) {
            int track_id;
            ss >> track_id;
            if (map->track_map.find(track_id) == map->track_map.end()) {
                map->track_map[track_id] = std::make_unique<Track>(track_id);
            }
            ss >> map->track_map[track_id]->point_3D.x() >> map->track_map[track_id]->point_3D.y()
                >> map->track_map[track_id]->point_3D.z();
            ss >> map->track_map[track_id]->color[0] >> map->track_map[track_id]->color[1]
                >> map->track_map[track_id]->color[2];
            ss >> map->track_map[track_id]->error;
            //            map->index_to_track_id_map[point_index_in_map] = track_id;
            //            map->track_id_to_index_map[track_id]= point_index_in_map;
            while (ss) {
                ss >> image_id;
                ss >> point2d_index;
                map->track_map[track_id]->observations[image_id - 1] = point2d_index;
            }
            //            std::cout<<track.id<<std::endl;
        }
        line_num++;
        ss = get_line_ss(fin, valid);
    }
    //    std::cout<<line_num<<std::endl;
}

void ColmapReader::associate_OBS() {
    std::cout << "associate_OBS Begin!" << std::endl;
    int obs_id = 0;
    int point_index = 0;
    for (auto &it : map->frame_map) {
        auto &frame = it.second;
        frame->qcw = frame->qcw.normalized();
        frame->keypoints_normalized.resize(frame->keypoints.size());
        for (int i = 0; i < frame->keypoints.size(); ++i) {
            auto &pt = frame->keypoints[i];
            Eigen::Vector2d point_normalized;
            point_normalized(0) = (pt.x() - frame->cx) / frame->f;
            point_normalized(1) = (pt.y() - frame->cy) / frame->f;
            frame->keypoints_normalized[i] = point_normalized;
        }

        for (int i = 0; i < frame->keypoints.size(); ++i) {
            int track_id = frame->track_ids[i];
            if (track_id == -1)
                continue;
            auto &track_observation_map = map->track_map[track_id]->observations;
            if (track_observation_map.find(frame->frame_id) == track_observation_map.end())
                continue;
            if (track_observation_map.size() < 2)
                continue;
            if (map->track_id_to_index_map.count(track_id) == 0) {
                map->track_id_to_index_map[track_id] = point_index;
                map->index_to_track_id_map[point_index] = track_id;
                point_index++;
            }
            map->frame_to_observation_map[frame->frame_id].emplace_back(obs_id);
            map->track_to_observation_map[track_id].emplace_back(obs_id);
            map->observation_map[obs_id] = std::make_unique<Observation>(
                obs_id, frame->frame_id, track_id, frame->keypoints_normalized[i].x(),
                frame->keypoints_normalized[i].y());
            obs_id++;
        }
    }

    map->frame_num = map->frame_map.size();
    map->track_num = map->track_map.size();
    map->observation_num = map->observation_map.size();

    for (int i = 0; i < map->frame_num; ++i) {
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
                (map->frame_to_other_frame[i])[observation->frame_id].emplace_back(
                    frame_observation_id, observation->obs_id);
            }
        }
    }
    std::cout << "associate_OBS Done!" << std::endl;
}

bool ColmapReader::read_map() {
    read_cameras(path + "/cameras.txt");
    read_images(path + "/images.txt");
    read_points_3D(path + "/points3D.txt");

    for (auto &f : map->frame_map) {
        const int id = f.first; // f_id
        auto &frame = f.second; // frame
        for (int i = 0; i < frame->track_ids.size(); i++) {
            auto &track_id = frame->track_ids[i];
            if (track_id == -1)
                continue;
            if (map->track_map[track_id]->observations[id] != i) {
                frame->track_ids[i] = -1;
            }
        }
    }

    associate_OBS();
    std::cout << "frame num: " << map->frame_num << std::endl;
    std::cout << "track num: " << map->track_num << std::endl;
    std::cout << "point index num: " << map->index_to_track_id_map.size() << std::endl;
    std::cout << "observation num: " << map->observation_num << std::endl;
    std::cout << std::endl;
    //    std::cout<<"init_error: "<<init_error<<std::endl;
    return true;
}
