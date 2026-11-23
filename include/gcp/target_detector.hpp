#pragma once

#include "gcp/saddle_refiner.hpp"

#include <opencv2/core.hpp>

#include <optional>
#include <vector>

namespace gcp {

/// One detected 2x2 checkerboard target.
struct Detection {
  cv::Point2d center;         ///< Sub-pixel centre (OpenCV pixel convention).
  double scale = 0.0;         ///< Approx. quadrant side length [px].
  double contrast = 0.0;      ///< Light minus dark mean on the validation ring.
  double residual_rms = 0.0;  ///< From the saddle refiner [px].
  double isotropy = 0.0;      ///< From the saddle refiner.
};

struct DetectorParams {
  // --- Phase A: pre-processing ---------------------------------------------
  double clahe_clip_limit = 2.0;
  int clahe_tiles = 8;
  /// Adaptive threshold block size as a fraction of min(image width, height).
  double adaptive_block_ratio = 0.125;
  double adaptive_offset = 5.0;
  /// Pixels of erosion used to split dark quadrants touching at the centre.
  int split_erosion = 1;

  // --- Phase A: topological candidate extraction ----------------------------
  double min_quad_area = 36.0;          ///< [px^2]
  double max_quad_area_ratio = 0.25;    ///< relative to image area
  double polygon_epsilon_ratio = 0.05;  ///< approxPolyDP epsilon / perimeter
  double min_quad_side_ratio = 0.25;    ///< shortest / longest side
  /// |contour area / quad area - 1| must stay below this (rejects round blobs).
  double max_quad_fill_deviation = 0.15;

  // --- Phase A: ring validation (alternation test) --------------------------
  int ring_samples = 72;
  double min_contrast = 25.0;             ///< grey levels
  double min_antipodal_agreement = 0.85;  ///< fraction of samples
  double min_sector_fraction = 0.08;      ///< each of the 4 arcs >= this * 2pi

  // --- Phase B ---------------------------------------------------------------
  RefinerParams refiner{};
  int min_half_window = 4;
  int max_half_window = 25;
  /// Reject if residual RMS / half window exceeds this (not a clean saddle).
  double max_residual_ratio = 0.3;
};

/**
 * Detector for 2x2 checkerboard ground control points.
 *
 * Phase A  CLAHE -> adaptive threshold -> closed contours -> convex quads.
 *          Quad vertices are centre candidates; a candidate survives only if
 *          a ring around it shows exactly four alternating intensity sectors
 *          with centrally symmetric layout (dark-light-dark-light).
 * Phase B  Robust Förstner saddle point refinement (see SaddleRefiner).
 */
class TargetDetector {
 public:
  explicit TargetDetector(DetectorParams params = {});

  /// Detect all targets in `image` (8-bit grey or BGR).
  [[nodiscard]] std::vector<Detection> detect(const cv::Mat& image) const;

  /// Phase B + strict validation for a single known approximate position,
  /// e.g. the predicted position of a tracked target. `gray` must be CV_8UC1,
  /// `scale` is the approximate quadrant side length in pixels.
  [[nodiscard]] std::optional<Detection> verify(const cv::Mat& gray, cv::Point2d guess,
                                                double scale) const;

  [[nodiscard]] const DetectorParams& params() const noexcept { return params_; }

 private:
  DetectorParams params_;
};

/// Ring validation result, exposed for testing.
struct RingCheck {
  bool valid = false;
  double contrast = 0.0;
  int transitions = 0;
};

/// Sample a circle of radius `radius` around `c` and test for the
/// dark-light-dark-light saddle pattern.
[[nodiscard]] RingCheck checkSaddleRing(const cv::Mat& gray, cv::Point2d c, double radius,
                                        const DetectorParams& params);

}  // namespace gcp
