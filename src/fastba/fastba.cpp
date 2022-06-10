#include "fastba.h"
#include "../map/frame.h"

bool FASTBA::track_monocular(Frame *frame) {}

void FASTBA::feed_image(double t, cv::Mat image) {
    auto frame = std::make_unique<Frame>();
    // frame->m_K = config->camera_intrinsic;
    frame->m_image = image;
    frame->m_timestamp = t;

    track_monocular(frame.get());
    // frame->m_camera.q_cs = config->camera_to_body_rotation.normalized();
    // frame->m_camera.p_cs = config->camera_to_body_translation;
}
