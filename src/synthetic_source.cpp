#include "gcp/frame_source.hpp"

#include <cmath>
#include <numbers>

namespace gcp {

SyntheticSource::SyntheticSource(SyntheticSourceParams params) : params_(params) {}

bool SyntheticSource::grab(Frame& frame) {
  if (params_.frames > 0 && index_ >= static_cast<std::uint64_t>(params_.frames)) return false;

  const double t = static_cast<double>(index_);
  const int n = std::max(1, params_.targets);
  const cv::Size sz = params_.image_size;

  std::vector<SyntheticParams> targets(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    auto& p = targets[static_cast<size_t>(i)];
    // Targets are spread horizontally and move on Lissajous curves with
    // incommensurate frequencies, so sub-pixel phases change every frame.
    const double base_x = sz.width * (i + 0.5) / n;
    const double base_y = 0.5 * sz.height;
    const double phase = 1.3 * i;
    p.image_size = sz;
    p.center = {base_x + params_.motion_amplitude_px * std::sin(0.031 * t + phase),
                base_y + params_.motion_amplitude_px * std::sin(0.047 * t + 2.0 * phase)};
    p.target_size_px = params_.target_size_px * (1.0 - 0.15 * i / n);
    p.rotation_deg = 20.0 * i + 0.4 * t;
    p.tilt_x_deg = 20.0 * std::sin(0.013 * t + phase);
    p.tilt_y_deg = 15.0 * std::cos(0.017 * t + phase);
    p.noise_sigma = params_.noise_sigma;
    p.blur_sigma = params_.blur_sigma;
    p.supersampling = params_.supersampling;
    p.seed = params_.seed * 1000003ULL + index_;
  }

  frame.image = renderScene(targets, &frame.ground_truth);
  frame.index = index_;
  frame.timestamp = index_ * 33'333'333ULL;  // 30 fps in ns
  ++index_;
  return true;
}

std::string SyntheticSource::describe() const {
  return "synthetic " + std::to_string(params_.image_size.width) + "x" +
         std::to_string(params_.image_size.height) + ", " + std::to_string(params_.targets) +
         " moving targets";
}

}  // namespace gcp
