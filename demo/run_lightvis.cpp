#include <lightvis/lightvis.h>
#include <lightvis/shader.h>
#include <lyra/lyra.hpp>
#include <nuklear.h>
#include <thread>

#include "../dataset/trajectory_reader.h"

class TrajectoryVisualizer : public lightvis::LightVis {
    Eigen::Vector3d velocity_dir;

    Eigen::Quaterniond latest_output_q = Eigen::Quaterniond::Identity();
    Eigen::Vector3d latest_output_p = Eigen::Vector3d::Zero();

    std::vector<double> frame_timestamps;
    std::vector<Pose> frame_poses;
    std::vector<cv::Mat> frame_images;

    std::vector<Eigen::Vector3f> trajectory;
    Eigen::Vector4f trajectory_color;

    Eigen::Matrix3d K;

    std::unique_ptr<TrajectoryReader> trajectory_reader;

    int32_t is_playing = 0;
    int32_t frame_id = 0;
    int32_t target_fps = 30;

    const bool with_images = false;

public:
    bool traj_is_portrait = false;

public:
    TrajectoryVisualizer(const std::string &trajectory_path) : LightVis("LVO", 1600, 900) {
        trajectory_color = {1.0, 0.25, 0.4, 1.0};
        add_trajectory(trajectory, trajectory_color);

        K << 477, 0, 240, 0, 477, 320, 0, 0, 1;

        // read trajectory
        trajectory_reader = std::make_unique<TumTrajectoryReader>(trajectory_path);
        trajectory_reader->read_poses(frame_timestamps, frame_poses);

        //     if (with_images) {

        //         if (data_path.size() > 8
        //             && data_path.substr(data_path.size() - 8, 8) == std::string(".sensors")) {
        //             printf("to load sensors data\n");
        //             auto dataset_reader = DatasetReader::create_reader("sensors", data_path);
        //             std::vector<std::shared_ptr<lvo::Image>> lvo_images;
        //             bool loading_data = true;
        //             while (loading_data) {
        //                 DatasetReader::NextDataType next_data_type;
        //                 while ((next_data_type = dataset_reader->next()) == DatasetReader::AGAIN) {}
        //                 switch (next_data_type) {
        //                     case DatasetReader::AGAIN: { // impossible but we put it here
        //                     } break;
        //                     case DatasetReader::GYROSCOPE: {
        //                         dataset_reader->read_gyroscope();
        //                     } break;
        //                     case DatasetReader::ACCELEROMETER: {
        //                         dataset_reader->read_accelerometer();
        //                     } break;
        //                     case DatasetReader::ATTITUDE: {
        //                         dataset_reader->read_attitude();
        //                     } break;
        //                     case DatasetReader::GRAVITY: {
        //                         dataset_reader->read_gravity();
        //                     } break;
        //                     case DatasetReader::CAMERA: {
        //                         auto image = dataset_reader->read_image();
        //                         lvo_images.push_back(image);
        //                     } break;
        //                     case DatasetReader::END: {
        //                         loading_data = false;
        //                     } break;
        //                 }
        //             }
        //             int32_t i = 0;
        //             int32_t j = 0;
        //             while (i < frame_timestamps.size() && j < lvo_images.size()) {
        //                 if (j == lvo_images.size() - 1
        //                     || std::abs(frame_timestamps[i] - lvo_images[j]->t)
        //                            < std::abs(frame_timestamps[i] - lvo_images[j + 1]->t)) {
        //                     auto img = cv::Mat(
        //                         lvo_images[j]->image.rows, lvo_images[j]->image.cols, CV_8UC1,
        //                         lvo_images[j]->image.data);
        //                     cv::cvtColor(img, frame_images.emplace_back(), cv::COLOR_GRAY2RGBA);
        //                     printf(
        //                         "frame_timestamps[%d]: %lf, lvo_images[%d]->t: %lf\n", i,
        //                         frame_timestamps[i], j, lvo_images[j]->t);
        //                     ++i;
        //                 } else {
        //                     ++j;
        //                 }
        //             }
        //         } else {
        //             CameraCsv cam_csv;
        //             cam_csv.load(data_path + "/camera/data.csv");

        //             int32_t i = 0;
        //             int32_t j = 0;
        //             while (i < frame_timestamps.size() && j < cam_csv.items.size()) {
        //                 if (j == cam_csv.items.size() - 1
        //                     || std::abs(frame_timestamps[i] - cam_csv.items[j].t)
        //                            < std::abs(frame_timestamps[i] - cam_csv.items[j + 1].t)) {
        //                     auto img = cv::imread(
        //                         data_path + "/camera/images/" + cam_csv.items[j].filename,
        //                         cv::IMREAD_GRAYSCALE);
        //                     cv::cvtColor(img, frame_images.emplace_back(), cv::COLOR_GRAY2RGBA);
        //                     printf(
        //                         "frame_timestamps[%d]: %lf, cam_csv.items[%d].t: %lf\n", i,
        //                         frame_timestamps[i], j, cam_csv.items[j].t);
        //                     ++i;
        //                 } else {
        //                     ++j;
        //                 }
        //             }
        //         }
        //         printf(
        //             "frame_timestamps.size(): %zu, frame_images.size(): %zu\n", frame_timestamps.size(),
        //             frame_images.size());
        //     }
        // }
    }

    void load() override {}

    void unload() override {}

    bool step() {
        if (frame_id >= frame_poses.size()) {
            // exit(EXIT_SUCCESS);
            return false;
        }

        double t = time_now();
        double dt = t - last_frame_time;
        if (dt * target_fps <= 1.0) {
            double t_wait = std::max(1.0 / target_fps - dt, 0.001);
            std::this_thread::sleep_for(std::chrono::milliseconds((long long)(t_wait * 1000)));
        }
        last_frame_time = t;

        const double timestamp = frame_timestamps[frame_id];
        Pose pose = frame_poses[frame_id];
        Eigen::Matrix3d rdq;
        rdq << 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0;
        const Eigen::Quaterniond dq = Eigen::Quaterniond(rdq);
        if (traj_is_portrait) {
            pose.q = pose.q * dq;
        }
        latest_output_q = pose.q;
        latest_output_p = pose.p;

        Eigen::Vector3f p = latest_output_p.cast<float>();
        trajectory.push_back(p);
        location() = {p.x(), p.y(), 0.0};

        if (with_images) {
            constexpr int32_t obj_num_per_side = 50;
            constexpr double obj_distance = 10.0;
            constexpr double obj_radius = 0.5;

            cv::Mat img = frame_images[frame_id];

            const int target_image_longer_side_length =
                static_cast<int>(2.0 * std::max(K(0, 2), K(1, 2)));
            const int source_image_longer_side_length = std::max(img.rows, img.cols);

            if (target_image_longer_side_length != source_image_longer_side_length) {
                const double scale_factor = static_cast<double>(target_image_longer_side_length)
                                            / static_cast<double>(source_image_longer_side_length);
                cv::resize(img, img, cv::Size(), scale_factor, scale_factor, cv::INTER_LINEAR);
            }

            for (int32_t x = -obj_num_per_side; x <= obj_num_per_side; ++x) {
                for (int32_t y = -obj_num_per_side; y <= obj_num_per_side; ++y) {
                    const Eigen::Vector3d center(x * obj_distance, y * obj_distance, 0.0);
                    const Eigen::Vector3d center_c = pose.q.conjugate() * (center - pose.p);
                    const Eigen::Vector2d center_px = (K * center_c).hnormalized();
                    if (center_px.x() < -K(0, 2) * 0.5 || center_px.x() > K(0, 2) * 2.5
                        || center_px.y() < -K(1, 2) * 0.5 || center_px.y() > K(1, 2) * 2.5
                        || center_c.z() < 0.0) {
                        continue;
                    }

                    cv::Scalar color = CV_RGB(0, 255, 0);

                    Eigen::Matrix<double, 3, 8> global_vertices;
                    global_vertices.col(0) = Eigen::Vector3d(
                        center.x() + obj_radius, center.y() + obj_radius, center.z() + 0);
                    global_vertices.col(1) = Eigen::Vector3d(
                        center.x() + obj_radius, center.y() - obj_radius, center.z() + 0);
                    global_vertices.col(2) = Eigen::Vector3d(
                        center.x() - obj_radius, center.y() - obj_radius, center.z() + 0);
                    global_vertices.col(3) = Eigen::Vector3d(
                        center.x() - obj_radius, center.y() + obj_radius, center.z() + 0);
                    Eigen::Matrix<double, 3, 8> camera_vertices =
                        pose.q.conjugate().toRotationMatrix()
                        * (global_vertices.colwise() - pose.p);
                    Eigen::Matrix<double, 2, 8> pixel_vertices =
                        (K * camera_vertices).colwise().hnormalized();
                    cv::line(
                        img, cv::Point(pixel_vertices.col(0).x(), pixel_vertices.col(0).y()),
                        cv::Point(pixel_vertices.col(1).x(), pixel_vertices.col(1).y()), color, 2,
                        cv::LINE_AA);
                    cv::line(
                        img, cv::Point(pixel_vertices.col(0).x(), pixel_vertices.col(0).y()),
                        cv::Point(pixel_vertices.col(3).x(), pixel_vertices.col(3).y()), color, 2,
                        cv::LINE_AA);
                    cv::line(
                        img, cv::Point(pixel_vertices.col(2).x(), pixel_vertices.col(2).y()),
                        cv::Point(pixel_vertices.col(1).x(), pixel_vertices.col(1).y()), color, 2,
                        cv::LINE_AA);
                    cv::line(
                        img, cv::Point(pixel_vertices.col(2).x(), pixel_vertices.col(2).y()),
                        cv::Point(pixel_vertices.col(3).x(), pixel_vertices.col(3).y()), color, 2,
                        cv::LINE_AA);
                }
            }
            if (img.rows < img.cols) {
                cv::rotate(img, img, cv::ROTATE_90_CLOCKWISE);
            }
            cv::imshow("AR", img);
            cv::waitKey(1);
            // cv::imwrite("./output/videos/" + std::to_string((unsigned long long)(timestamp * 1e9)) + ".png", img);
        }

        if (frame_id > 0) {
            const Eigen::Vector3d delta_p = frame_poses[frame_id].p - frame_poses[frame_id - 1].p;
            velocity_dir = delta_p.normalized();
        }

        ++frame_id;
        return true;
    }

    void gui(void *ctx, int w, int h) override {
        auto *context = (nk_context *)(ctx);
        context->style.window.spacing = nk_vec2(0, 0);
        context->style.window.padding = nk_vec2(0, 0);
        context->style.window.border = 1.0;
        context->style.window.fixed_background = nk_style_item_color(nk_rgba(48, 48, 48, 128));

        if (nk_input_is_key_pressed(&context->input, NK_KEY_TAB)) {
            is_playing = 1 - is_playing;
        }

        // control panel
        if (nk_begin(context, "Controls", nk_rect(0, h - 40, 320, 40), NK_WINDOW_NO_SCROLLBAR)) {
            nk_layout_row_static(context, 40, 80, 4);
            if (is_playing) {
                is_playing = step();
            }
            if (is_playing) {
                if (nk_button_label(context, "Playing")) {
                    is_playing = false;
                }
            } else {
                if (nk_button_label(context, "Stopped")) {
                    is_playing = true;
                }
                nk_button_push_behavior(context, NK_BUTTON_REPEATER);
                if (nk_button_label(context, "Forward")) {
                    step();
                }
                nk_button_pop_behavior(context);
                nk_button_push_behavior(context, NK_BUTTON_DEFAULT);
                if (nk_button_label(context, "Step")) {
                    step();
                }
                nk_button_pop_behavior(context);
            }
        }
        nk_end(context);

        // fps control panel
        if (nk_begin(
                context, "FPSControl", nk_rect(320, h - 40, 200, 40), NK_WINDOW_NO_SCROLLBAR)) {
            nk_layout_row_begin(context, NK_STATIC, 40, 2);
            {
                nk_layout_row_push(context, 80);
                nk_property_int(context, "fps: ", 1, &target_fps, 60, 2, 1);
                nk_layout_row_push(context, 120);
                nk_slider_int(context, 1, &target_fps, 60, 1);
            }
            nk_layout_row_end(context);
        }
        nk_end(context);

        int32_t current_y = 0;

        context->style.window.fixed_background = nk_style_item_color(nk_rgba(32, 32, 32, 0));

        if (nk_begin(
                context, "Overlays", nk_rect(w - 100, 0, 100, 30),
                NK_WINDOW_NO_SCROLLBAR | NK_WINDOW_NO_INPUT)) {
            char frameid_text[18];
            snprintf(frameid_text, 18, "frame id: %d", frame_id);
            nk_command_buffer *canvas = nk_window_get_canvas(context);
            nk_draw_text(
                canvas, nk_rect(w - 90, 5, 90, 25), frameid_text, strlen(frameid_text),
                context->style.font, nk_rgba(255, 255, 255, 0), nk_rgba(255, 255, 255, 255));
        }
        nk_end(context);
    }

    void draw_camera(
        const Eigen::Vector3d &p, const Eigen::Quaterniond &q, const Eigen::Matrix3d &K,
        const Eigen::Vector4d &color, double size) {
        std::vector<Eigen::Vector3f> draw_point;
        std::vector<Eigen::Vector4f> draw_color;
        draw_point.resize(18);
        draw_color.resize(18);

        std::array<Eigen::Vector3d, 6> cone_points;
        Eigen::Matrix3d dcm = q.matrix();
        cone_points[0] = p;
        cone_points[1] =
            p + size * (dcm.col(2) * K(0, 0) + dcm.col(0) * K(0, 2) + dcm.col(1) * K(1, 2));
        cone_points[2] =
            p + size * (dcm.col(2) * K(0, 0) + dcm.col(0) * K(0, 2) - dcm.col(1) * K(1, 2));
        cone_points[3] =
            p + size * (dcm.col(2) * K(0, 0) - dcm.col(0) * K(0, 2) - dcm.col(1) * K(1, 2));
        cone_points[4] =
            p + size * (dcm.col(2) * K(0, 0) - dcm.col(0) * K(0, 2) + dcm.col(1) * K(1, 2));
        cone_points[5] = p;
        // cone_points[5] = p + velocity_dir * 5.0;

        draw_point[0] = cone_points[0].cast<float>();
        draw_point[1] = cone_points[1].cast<float>();
        draw_point[2] = cone_points[0].cast<float>();
        draw_point[3] = cone_points[2].cast<float>();
        draw_point[4] = cone_points[0].cast<float>();
        draw_point[5] = cone_points[3].cast<float>();
        draw_point[6] = cone_points[0].cast<float>();
        draw_point[7] = cone_points[4].cast<float>();
        draw_point[8] = cone_points[1].cast<float>();
        draw_point[9] = cone_points[2].cast<float>();
        draw_point[10] = cone_points[2].cast<float>();
        draw_point[11] = cone_points[3].cast<float>();
        draw_point[12] = cone_points[3].cast<float>();
        draw_point[13] = cone_points[4].cast<float>();
        draw_point[14] = cone_points[4].cast<float>();
        draw_point[15] = cone_points[1].cast<float>();
        draw_point[16] = cone_points[0].cast<float>();
        draw_point[17] = cone_points[5].cast<float>();

        for (int i = 0; i < 16; ++i) {
            draw_color[i] = color.cast<float>();
        }
        Eigen::Vector4f velocity_color = {1.0, 1.0, 1.0, 0.8};
        draw_color[16] = velocity_color;
        draw_color[17] = velocity_color;

        shader()->bind();
        shader()->set_uniform(
            "ProjMat", Eigen::Matrix4f(projection_matrix() * view_matrix() * model_matrix()));
        shader()->set_uniform("Location", location());
        shader()->set_uniform("Scale", scale());
        shader()->set_attribute("Position", draw_point);
        shader()->set_attribute("Color", draw_color);
        shader()->draw(gl::GL_LINES, 0, draw_point.size());
        shader()->unbind();
    }

    void draw_axis() {
        std::vector<Eigen::Vector3f> draw_point;
        std::vector<Eigen::Vector4f> draw_color;
        draw_point.resize(6);
        draw_color.resize(6);

        std::array<Eigen::Vector3d, 4> cone_points;
        cone_points[0] = {0, 0, 0};
        cone_points[1] = {5, 0, 0};
        cone_points[2] = {0, 5, 0};
        cone_points[3] = {0, 0, 5};

        draw_color[0] = {1, 0, 0, 1};
        draw_color[1] = {1, 0, 0, 1};
        draw_color[2] = {0, 1, 0, 1};
        draw_color[3] = {0, 1, 0, 1};
        draw_color[4] = {0, 0, 1, 1};
        draw_color[5] = {0, 0, 1, 1};

        draw_point[0] = cone_points[0].cast<float>();
        draw_point[1] = cone_points[1].cast<float>();
        draw_point[2] = cone_points[0].cast<float>();
        draw_point[3] = cone_points[2].cast<float>();
        draw_point[4] = cone_points[0].cast<float>();
        draw_point[5] = cone_points[3].cast<float>();

        shader()->bind();
        shader()->set_uniform(
            "ProjMat", Eigen::Matrix4f(projection_matrix() * view_matrix() * model_matrix()));
        shader()->set_uniform("Location", location());
        shader()->set_uniform("Scale", scale());
        shader()->set_attribute("Position", draw_point);
        shader()->set_attribute("Color", draw_color);
        shader()->draw(gl::GL_LINES, 0, draw_point.size());
        shader()->unbind();
    }

    void draw(int w, int h) override {
        auto err = gl::glGetError();
        if (err != gl::GL_NONE) {
            // std::cout << err << std::endl;
            exit(0);
        }
        draw_camera(latest_output_p, latest_output_q, K, {1.0, 1.0, 0.0, 0.8}, 0.001);
        draw_axis();
    }

    static double time_now() {
        return std::chrono::duration_cast<std::chrono::duration<double>>(
                   std::chrono::high_resolution_clock::now().time_since_epoch())
            .count();
    }
    double last_frame_time;
};

int main(int argc, const char *argv[]) {

    // bool show_help = false;

    // std::string euroc_data_path;
    // std::string config_file_path;

    // auto cli =
    //     lyra::cli() | lyra::help(show_help).description("Run FastBA")
    //     | lyra::opt(euroc_data_path, "dataset path")["-d"]["--dataset-path"]("Euroc dataset path")
    //           .required()
    //     | lyra::opt(config_file_path, "config path")["-c"]["--config-path"]("Config file path");


    // auto cli_result = cli.parse({argc, argv});
    // if (!cli_result) {
    //     fmt::print(stderr, "{}\n\n{}\n", cli_result.message(), cli);
    //     return -1;
    // }

    std::string euroc_path = "/Users/lemon/dataset/MH_05";
    bool traj_is_portrait = false;
    TrajectoryVisualizer visualizer(euroc_data_path);
    visualizer.traj_is_portrait = false;
    visualizer.show();
    return lightvis::main();
    return 0;
}