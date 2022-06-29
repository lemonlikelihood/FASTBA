#include "forensics.h"
#include "common.h"
#include <fastba/version.h>

struct ForensicsSupport::VersionTag {
    int major;
    int minor;
    int patch;
};

ForensicsSupport::ForensicsSupport(const VersionTag &tag) : storage(FS_ITEM_COUNT) {
    storage[FS_RESERVED].first = tag;
}

ForensicsSupport::~ForensicsSupport() = default;

std::pair<std::any &, std::unique_lock<std::mutex>> ForensicsSupport::get(ForensicsItem item) {
    auto &si = support().storage[item];
    return {si.first, std::unique_lock(si.second)};
}

ForensicsSupport &ForensicsSupport::support() {
    static std::unique_ptr<ForensicsSupport> s_support = std::make_unique<ForensicsSupport>(
        VersionTag {PROJECTION_MAJOR_VERSION, PROJECTION_MINOR_VERSION, PROJECTION_PATCH_VERSION});
    return *s_support;
}