//
// Created by lemon on 2021/1/21.
//

#ifndef FASTBA_MAP_H
#define FASTBA_MAP_H

#include <Eigen/Eigen>
#include <iostream>
#include <map>
#include <memory>

class Feature {
public:
    Feature();
    ~Feature();
    Feature(double x_, double y_);
    Eigen::Vector2d corner;
    Eigen::Vector2d normalized;
};

class Frame {
public:
    Frame();
    Frame(int frame_id_);
    ~Frame();
    int frame_id;
    int index_in_map;
    double timestamp;
    std::string camera_model;
    std::string image_name;
    int camera_id;

    int width;
    int height;
    double cx;
    double cy;

    double f;
    double k1;
    double k2;

    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> keypoints;
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> keypoints_normalized;
    std::vector<int> track_ids;

    Eigen::Quaterniond qwc;
    Eigen::Vector3d pwc;

    Eigen::Quaterniond qcw;
    Eigen::Vector3d pcw;
};

class Track {
public:
    Track();
    Track(int track_id_);
    ~Track();

    int track_id;
    Eigen::Vector3d point_3D;
    char color[3];
    double error;
    std::map<int, size_t> observations;
};

class Observation {
public:
    Observation();
    ~Observation();
    Observation(int obs_id_, int frame_id_, int track_id_, double x_, double y_);
    int obs_id;
    int frame_id;
    int track_id;
    std::unique_ptr<Feature> feature;
};

class Map {
public:
    Map();
    ~Map();
    void print_map();
    double calculate_reprojection_error(bool bal_flag = false);
    std::map<int, std::unique_ptr<Frame>> frame_map;             // frame_id ->Frame*
    std::map<int, std::unique_ptr<Track>> track_map;             // track_id ->Track*
    std::map<int, std::unique_ptr<Observation>> observation_map; // observation_id -> Observation*
    std::map<int, std::vector<int>> frame_to_observation_map;    // frame_id -> observation_id
    std::map<int, std::vector<int>> track_to_observation_map;    // track_id -> observation_id
    std::map<int, std::map<int, std::vector<std::pair<int, int>>>>
        frame_to_other_frame; // other_frame_id -> {<observation_id1,observation_id2>}
    std::map<int, int> index_to_track_id_map;
    std::map<int, int> track_id_to_index_map;
    int frame_num;
    int track_num;
    int observation_num;
};

#endif //FASTBA_MAP_H
