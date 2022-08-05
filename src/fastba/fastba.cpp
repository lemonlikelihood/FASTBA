#include "fastba.h"
#include "../../dataset/dataset.h"
#include "../map/frame.h"
#include "../map/sliding_window.h"
#include "../utils/debug.h"

bool FASTBA::track_monocular(Frame *frame) {
    if (!m_initialized) {
        tracker->track_monocular(frame);
    }
    return true;
}

std::unique_ptr<Frame>
FASTBA::create_frame(std::shared_ptr<ImageData> image, DatasetConfigurator *dataset_config) {
    auto frame = std::make_unique<Frame>();
    // frame->m_K = config->camera_intrinsic;
    log_debug("frame id: {}", frame->id());
    frame->m_image = image;
    frame->m_K = dataset_config->camera_intrinsic();
    frame->m_camera_extri.q_sensor2body = dataset_config->camera_to_body_rotation();
    frame->m_camera_extri.p_sensor2body = dataset_config->camera_to_body_translation();
    frame->m_imu_extri.q_sensor2body = dataset_config->camera_to_body_rotation();
    frame->m_imu_extri.p_sensor2body = dataset_config->camera_to_body_translation();
    frame->m_preintegration.cov_w = dataset_config->imu_gyro_white_noise();
    frame->m_preintegration.cov_a = dataset_config->imu_accel_white_noise();
    frame->m_preintegration.cov_bg = dataset_config->imu_gyro_random_walk();
    frame->m_preintegration.cov_ba = dataset_config->imu_accel_random_walk();
    return frame;
}

void FASTBA::feed_image(std::shared_ptr<ImageData> image, DatasetConfigurator *dataset_config) {
    auto frame = create_frame(image, dataset_config);
    track_monocular(frame.get());
    m_sliding_window->put_frame(std::move(frame));
}

FASTBA::FASTBA() {
    tracker = std::make_unique<KLTTracker>();
    m_sliding_window = std::make_unique<SlidingWindow>();
}

FASTBA::~FASTBA() {}
