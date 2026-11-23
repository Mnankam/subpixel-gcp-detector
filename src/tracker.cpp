#include "gcp/tracker.hpp"

#include <algorithm>

namespace gcp {

Tracker::Tracker(TrackerParams params) : params_(params), detector_(params.detector) {}

void Tracker::reset() {
  tracks_.clear();
  frame_ = 0;
  next_id_ = 0;
  last_full_ = false;
}

const std::vector<Track>& Tracker::update(const cv::Mat& gray) {
  CV_Assert(gray.type() == CV_8UC1);

  // --- 1. cheap path: refine every track at its predicted position --------
  for (auto& t : tracks_) {
    ++t.age;
    const cv::Point2d predicted = t.detection.center + t.velocity;
    if (auto d = detector_.verify(gray, predicted, t.detection.scale)) {
      const cv::Point2d moved = d->center - t.detection.center;
      t.velocity = 0.5 * t.velocity + 0.5 * moved;  // light smoothing
      t.detection = *d;
      t.misses = 0;
    } else {
      ++t.misses;
      t.velocity *= 0.5;
    }
  }
  tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                               [&](const Track& t) { return t.misses >= params_.max_misses; }),
                tracks_.end());

  // --- 2. expensive path: full detection to (re)acquire targets -----------
  const bool any_lost =
      std::any_of(tracks_.begin(), tracks_.end(), [](const Track& t) { return t.misses > 0; });
  last_full_ = tracks_.empty() || any_lost ||
               (params_.redetect_interval > 0 &&
                frame_ % static_cast<std::uint64_t>(params_.redetect_interval) == 0);
  if (last_full_) {
    for (const auto& d : detector_.detect(gray)) {
      auto it = std::find_if(tracks_.begin(), tracks_.end(), [&](const Track& t) {
        return cv::norm(t.detection.center - d.center) < 0.5 * std::min(t.detection.scale, d.scale);
      });
      if (it != tracks_.end()) {
        if (it->misses > 0) {  // re-acquired
          it->detection = d;
          it->misses = 0;
        } else {
          it->detection.scale = d.scale;  // keep the refined centre, update scale
        }
      } else {
        tracks_.push_back(Track{next_id_++, d, {0.0, 0.0}, 0, 0});
      }
    }
  }

  ++frame_;
  return tracks_;
}

}  // namespace gcp
