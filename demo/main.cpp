#include <lightvis/lightvis.h>
#include <lightvis/shader.h>
#include <lyra/lyra.hpp>
#include <nuklear.h>
#include <thread>

#include "../dataset/euroc_dataset_reader.h"
#include "../dataset/trajectory_reader.h"
#include "../src/fastba/fastba.h"
#include "../src/geometry/lie_algebra.h"
#include "../src/utils/debug.h"
#include "../src/utils/read_file.h"

#include "config.h"
#include "fastba_player.h"


class FastBAVisualizer : public lightvis::LightVis {
    Eigen::Vector3d velocity_dir;

    Eigen::Quaterniond latest_output_q = Eigen::Quaterniond::Identity();
    Eigen::Vector3d latest_output_p = Eigen::Vector3d::Zero();

    std::vector<double> frame_timestamps;
    std::vector<Pose> frame_poses;
    std::vector<cv::Mat> frame_images;

    std::vector<Eigen::Vector3f> trajectory;
    Eigen::Vector4f trajectory_color;

    std::unique_ptr<TrajectoryReader> trajectory_reader;
    std::unique_ptr<FastBAPlayer> fastba_player;
    std::unique_ptr<TrajectoryWriter> trajectory_writer;


    int32_t is_playing = 0;
    int32_t frame_id = 0;
    int32_t target_fps = 30;
    double real_gui_fps = 30;

    // std::unique_ptr<FastBAPlayer> fastba_player;

    const bool with_images = true;

public:
    bool traj_is_portrait = false;

public:
    FastBAVisualizer(const std::string &euroc_data_path, const std::string &config_path)
        : LightVis("LVO", 1600, 900) {

        fastba_player = std::make_unique<FastBAPlayer>(euroc_data_path, config_path);
        std::string tum_pose = "trajectory.txt";
        trajectory_writer = std::make_unique<TumTrajectoryWriter>(tum_pose);

        trajectory_color = {1.0, 0.25, 0.4, 1.0};
        add_trajectory(trajectory, trajectory_color);

        // K << 477, 0, 240, 0, 477, 320, 0, 0, 1;

        // read trajectory
        // trajectory_reader = std::make_unique<TumTrajectoryReader>(trajectory_path);
        // trajectory_reader->read_poses(frame_timestamps, frame_poses);

        // auto config = std::make_unique<Config>(config_path);
    }

    void load() override {}

    void unload() override {}

    bool step() {
        if (!fastba_player->step()) {
            // exit(EXIT_SUCCESS);
            return false;
        }

        double const t = fastba_player->image->t;
        bool const tracking_state = fastba_player->tracking_state;
        auto const pose = fastba_player->pose;

        frame_id = fastba_player->fid;

        trajectory_writer->write_pose(t, pose);

        Eigen::Vector3f p = latest_output_p.cast<float>();
        trajectory.push_back(p);
        location() = {p.x(), p.y(), 0.0};
        latest_output_q = pose.q;
        latest_output_p = pose.p;

        // cv::Mat cv_img_ar = fastba_player->image->image.clone();
        // draw_virtual_object();

        const double current_frame_time = time_now();
        const double dt = current_frame_time - last_frame_time;
        real_gui_fps = 1.0 / std::max(dt, 0.001);
        if (dt * target_fps <= 1.0) {
            double const t_wait = std::max(1.0 / target_fps - dt, 0.001);
            using namespace std::chrono;
            std::this_thread::sleep_for(
                milliseconds(static_cast<milliseconds::rep>(t_wait * 1000)));
        }
        last_frame_time = current_frame_time;
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
                context, "Overlays", nk_rect(w - 200, 0, 200, 60),
                NK_WINDOW_NO_SCROLLBAR | NK_WINDOW_NO_INPUT)) {
            char frameid_text[18];
            char keyframeids_text[40];
            snprintf(frameid_text, 18, "frame id: %d", frame_id);
            snprintf(keyframeids_text, 40, "keyframe id: 1,2,3,4,5");
            nk_command_buffer *canvas = nk_window_get_canvas(context);
            nk_draw_text(
                canvas, nk_rect(w - 190, 5, 190, 25), frameid_text, strlen(frameid_text),
                context->style.font, nk_rgba(255, 255, 255, 0), nk_rgba(255, 255, 255, 255));
            // nk_draw_text(
            //     canvas, nk_rect(w - 190, 35, 190, 25), keyframeids_text, strlen(keyframeids_text),
            //     context->style.font, nk_rgba(255, 255, 255, 0), nk_rgba(255, 255, 255, 255));
        }
        nk_end(context);
    }

    void draw(int w, int h) override {
        auto err = gl::glGetError();
        if (err != gl::GL_NONE) {
            // std::cout << err << std::endl;
            exit(0);
        }
        draw_camera(
            latest_output_p, latest_output_q, fastba_player->K, {1.0, 1.0, 0.0, 0.8},
            {0.0, 0.0, 0.0}, 0.001);
        for (auto pose : frame_poses) {
            draw_camera(
                pose.p, pose.q, fastba_player->K, {0.0, 1.0, 1.0, 0.8}, {0.0, 0.0, 0.0}, 0.0006);
        }
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

    bool show_help = false;

    std::string euroc_data_path;
    std::string config_file_path;

    auto cli =
        lyra::cli() | lyra::help(show_help).description("Run FastBA")
        | lyra::opt(euroc_data_path, "dataset path")["-d"]["--dataset-path"]("Euroc dataset path")
              .required()
        | lyra::opt(config_file_path, "config path")["-c"]["--config-path"]("Config file path");


    auto cli_result = cli.parse({argc, argv});
    if (!cli_result) {
        fmt::print(stderr, "{}\n\n{}\n", cli_result.message(), cli);
        return -1;
    }

    FastBAVisualizer visualizer(euroc_data_path, config_file_path);
    visualizer.show();
    return lightvis::main();
}