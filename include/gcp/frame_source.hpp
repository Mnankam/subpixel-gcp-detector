#pragma once

#include "gcp/synthetic.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace gcp {

/// One image delivered by a FrameSource.
struct Frame {
  cv::Mat image;                ///< CV_8UC1, owned by the frame
  std::uint64_t index = 0;      ///< running frame number
  std::uint64_t timestamp = 0;  ///< device ticks (camera) or ns (other sources)
  /// Exact target centres; only filled by synthetic sources.
  std::vector<cv::Point2d> ground_truth;
};

/**
 * Abstract image source: a real camera, a camera emulator, a list of files
 * or a synthetic generator. The processing pipeline only sees this
 * interface, so the same code runs in CI (synthetic) and on the shop floor
 * (camera).
 */
class FrameSource {
 public:
  virtual ~FrameSource() = default;
  FrameSource() = default;
  FrameSource(const FrameSource&) = delete;
  FrameSource& operator=(const FrameSource&) = delete;

  /// Blocks until the next frame is available. Returns false at end of stream.
  virtual bool grab(Frame& frame) = 0;
  /// Human readable description (camera model, serial, ...).
  [[nodiscard]] virtual std::string describe() const = 0;
};

struct SyntheticSourceParams {
  cv::Size image_size{1280, 720};
  int frames = 0;  ///< 0 = endless
  int targets = 2;
  double target_size_px = 130.0;
  double motion_amplitude_px = 60.0;  ///< targets move on Lissajous curves
  double noise_sigma = 2.0;
  double blur_sigma = 0.8;
  int supersampling = 32;  ///< lower than the benchmark default for speed
  std::uint64_t seed = 1;
};

/// Simulated camera: moving, rotating and slightly tilting targets with
/// exact ground truth in every frame.
class SyntheticSource final : public FrameSource {
 public:
  explicit SyntheticSource(SyntheticSourceParams params = {});
  bool grab(Frame& frame) override;
  [[nodiscard]] std::string describe() const override;

 private:
  SyntheticSourceParams params_;
  std::uint64_t index_ = 0;
};

}  // namespace gcp
