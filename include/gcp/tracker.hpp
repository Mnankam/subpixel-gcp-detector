#pragma once

#include "gcp/target_detector.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <vector>

namespace gcp {

struct TrackerParams {
  DetectorParams detector{};
  /// Run the full detector every N frames to pick up new targets
  /// (in addition to whenever no target is tracked).
  int redetect_interval = 30;
  /// Drop a track after this many consecutive failed updates.
  int max_misses = 3;
};

struct Track {
  int id = 0;
  Detection detection;   ///< latest estimate
  cv::Point2d velocity;  ///< px / frame, constant-velocity prediction
  int age = 0;           ///< frames since creation
  int misses = 0;        ///< consecutive failed updates
};

/**
 * Real-time tracker for GCPs in an image stream.
 *
 * The full detector (CLAHE, thresholds, contours) costs tens of
 * milliseconds; refining a known target costs well under one. The tracker
 * therefore predicts each target with a constant-velocity model, runs only
 * the robust sub-pixel refinement + ring validation at the prediction, and
 * falls back to full detection periodically or when tracks are lost.
 */
class Tracker {
 public:
  explicit Tracker(TrackerParams params = {});

  /// Process one 8-bit grey frame and return the current tracks.
  const std::vector<Track>& update(const cv::Mat& gray);

  [[nodiscard]] const std::vector<Track>& tracks() const noexcept { return tracks_; }
  /// True if the last update() ran the full detector.
  [[nodiscard]] bool lastWasFullDetection() const noexcept { return last_full_; }
  void reset();

 private:
  TrackerParams params_;
  TargetDetector detector_;
  std::vector<Track> tracks_;
  std::uint64_t frame_ = 0;
  int next_id_ = 0;
  bool last_full_ = false;
};

}  // namespace gcp
