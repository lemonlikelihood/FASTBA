//
// Created by lemon on 2021/1/21.
//

#include "map.h"
#include <iostream>

Frame::Frame() = default;

Frame::~Frame() = default;

Frame::Frame(int frame_id_) {
    frame_id = frame_id_;
}

Track::Track() = default;

Track::Track(int track_id_) {
    track_id = track_id_;
}

Track::~Track() = default;

Feature::Feature() = default;

Feature::~Feature() = default;

Feature::Feature(double x_, double y_) {
    corner.x() = x_;
    corner.y() = y_;
    normalized.x() = x_;
    normalized.y() = y_;
}

Map::Map() = default;

Map::~Map() = default;

Observation::Observation() = default;

Observation::~Observation() = default;

Observation::Observation(int obs_id_, int frame_id_, int track_id_, double x_, double y_) :
        obs_id(obs_id_), frame_id(frame_id_), track_id(track_id_) {
    feature = std::make_unique<Feature>(x_, y_);
}

void Map::print_map() {
    std::cout << "Frame num: " << frame_num << std::endl;
    std::cout << "Track num: " << track_num << std::endl;
    std::cout << "Observation num: " << observation_num << std::endl;
//    for (int i = 0; i < frame_num; i++) {
//        Frame *frame_cur = frame_map[i].get();
////        auto features_cur = frame_cur->features;
//        auto &frame_curr_observation_vec = frame_to_observation_map[i];
//        std::cout << "frame_id: "<<frame_cur->frame_id << "observation_size: " << frame_curr_observation_vec.size() << std::endl;
//        for (auto &item:frame_curr_observation_vec) {
//            std::cout << observation_map[item]->feature->corner.x() << " "
//                      << observation_map[item]->feature->corner.y() << std::endl;
//        }
//        std::cout << std::endl;
//    }
//    std::cout<<"helo"<<std::endl;

    for (int i = 0; i < 5; i++) {
        Track *track_cur = track_map[i].get();
        auto &track_curr_observation_vec = track_to_observation_map[i];
        std::cout << "track_id: " << track_cur->track_id << " observation_size:" << track_curr_observation_vec.size()
                  << std::endl;
        for (int j = 0; j < track_curr_observation_vec.size(); j++) {
            std::cout << observation_map[track_curr_observation_vec[j]]->obs_id << " "
                      << observation_map[track_curr_observation_vec[j]]->frame_id << " "
                      << observation_map[track_curr_observation_vec[j]]->feature->corner.x() << " "
                      << observation_map[track_curr_observation_vec[j]]->feature->corner.y() << std::endl;
        }
    }

    for (int i = 0; i < frame_to_other_frame[0][1].size(); ++i) {
        std::cout << observation_map[frame_to_other_frame[0][1][i].first]->track_id << " "
                  << observation_map[frame_to_other_frame[0][1][i].first]->frame_id << " "
                  << observation_map[frame_to_other_frame[0][1][i].first]->obs_id << " "
                  << observation_map[frame_to_other_frame[0][1][i].first]->feature->corner.x() << " "
                  << observation_map[frame_to_other_frame[0][1][i].first]->feature->corner.y() << std::endl;
        std::cout << observation_map[frame_to_other_frame[0][1][i].second]->track_id << " "
                  << observation_map[frame_to_other_frame[0][1][i].second]->frame_id << " "
                  << observation_map[frame_to_other_frame[0][1][i].second]->obs_id << " "
                  << observation_map[frame_to_other_frame[0][1][i].second]->feature->corner.x() << " "
                  << observation_map[frame_to_other_frame[0][1][i].second]->feature->corner.y() << std::endl;
        std::cout << std::endl;
    }
}

double Map::calculate_reprojection_error() {
    int count = 0;
    double mse = 0;
    for (auto &obs:observation_map) {
        int obs_id = obs.first;
        int track_id = (obs.second)->track_id;
        int frame_id = (obs.second)->frame_id;
        Eigen::Quaterniond qcw(frame_map[frame_id]->qcw);
        Eigen::Vector3d pcw(frame_map[frame_id]->pcw);
        Eigen::Vector3d pw(track_map[track_id]->point_3D);

        Eigen::Matrix3d Rcw = qcw.toRotationMatrix();
        Eigen::Vector3d Pc = Rcw * pw + pcw;
        Eigen::Vector2d Pn = Pc.head<2>() / Pc.z();
        Eigen::Vector2d res = Pn - (obs.second)->feature->normalized;

        mse += res.norm();
        count++;
    }
    return mse / count;
}
