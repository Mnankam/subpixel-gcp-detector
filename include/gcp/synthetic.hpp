#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <vector>

namespace gcp {

/// Parameters for rendering a synthetic 2x2 checkerboard GCP with exactly
/// known sub-pixel ground truth.
struct SyntheticParams {
  cv::Size image_size{640, 480};
  cv::Point2d center{320.25, 240.75};  ///< ground truth target centre [px]
  double target_size_px = 160.0;       ///< side length of the 2x2 pattern (fronto-parallel)
  double rotation_deg = 0.0;           ///< in-plane rotation
  double tilt_x_deg = 0.0;             ///< out-of-plane tilt -> perspective
  double tilt_y_deg = 0.0;
  double focal_px = 1000.0;  ///< smaller = stronger perspective

  double dark = 35.0;
  double light = 215.0;
  double panel_margin = 0.18;  ///< light border around the pattern (fraction of size)
  bool add_screws = true;      ///< dark screw heads in the panel corners
  double background_mean = 110.0;
  double background_texture = 25.0;  ///< amplitude of a smooth random background
  double material_texture = 4.0;     ///< fine texture on the panel

  double illumination_gradient = 0.0;  ///< e.g. 0.3 = +/-15 % across the image
  bool cast_shadow = false;            ///< half-plane shadow crossing the target
  double shadow_factor = 0.55;

  double blur_sigma = 0.7;   ///< optics / motion blur
  double noise_sigma = 2.0;  ///< sensor noise [grey levels]
  int supersampling = 128;   ///< samples per axis and pixel at edges
                             ///< (1/128 px ground-truth resolution)
  std::uint64_t seed = 42;
};

struct SyntheticImage {
  cv::Mat image;       ///< CV_8UC1
  cv::Point2d center;  ///< exact ground truth
  cv::Matx33d H;       ///< plane -> image homography (plane units = px at centre)
};

[[nodiscard]] SyntheticImage renderTarget(const SyntheticParams& params);

/// Render several targets into one image (each with its own geometry).
/// Image size, background and noise are taken from the first entry.
[[nodiscard]] cv::Mat renderScene(const std::vector<SyntheticParams>& targets,
                                  std::vector<cv::Point2d>* centers = nullptr);

}  // namespace gcp
