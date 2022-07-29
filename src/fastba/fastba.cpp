#include "fastba.h"
#include "../../dataset/dataset.h"
#include "../map/frame.h"
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
    return frame;
}

void FASTBA::feed_image(std::shared_ptr<ImageData> image, DatasetConfigurator *dataset_config) {
    auto frame = create_frame(image, dataset_config);
    track_monocular(frame.get());
}

FASTBA::FASTBA() {
    tracker = std::make_unique<KLTTracker>();
}

FASTBA::~FASTBA() {}
