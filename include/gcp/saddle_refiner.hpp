#pragma once

#include <opencv2/core.hpp>

#include <limits>

namespace gcp {

/// Parameters of the sub-pixel saddle point estimator.
struct RefinerParams {
  /// Half size of the square estimation window (window = 2*half_window+1).
  int half_window = 7;
  /// Std. deviation of the Gaussian spatial weight; <= 0 means half_window / 2.
  double spatial_sigma = 0.0;
  /// Pixels closer than this to the current estimate are ignored. Right at
  /// the saddle the blurred gradient field is not orthogonal to (p - c) and
  /// changes direction within a pixel, which only adds sampling bias.
  double zero_zone_radius = 0.0;
  /// Huber tuning constant in units of the robust residual scale
  /// (1.4826 * weighted median of |n^T (p - c)|, re-estimated every
  /// iteration). 1.345 gives 95 % efficiency under Gaussian residuals.
  /// Use std::numeric_limits<double>::infinity() for plain least squares.
  double huber_k = 1.345;
  /// Pixels whose gradient magnitude is below this fraction of the window
  /// maximum are ignored (they carry only noise direction information).
  double min_gradient_ratio = 0.0;
  /// Exponent gamma of the gradient magnitude in the weight
  /// (2 = classic Förstner / structure tensor).
  double gradient_weight_power = 2.0;
  /// Maximum number of IRLS iterations.
  int max_iterations = 30;
  /// Convergence threshold on the update step in pixels.
  double epsilon = 1e-4;
  /// Reject the solution if lambda_min / lambda_max of the normal matrix is
  /// below this value (an edge or a flat region instead of a saddle point).
  double min_isotropy = 0.05;
};

/// Result of a single refinement.
struct RefineResult {
  cv::Point2d center;         ///< Estimated saddle point (OpenCV pixel convention).
  double residual_rms = 0.0;  ///< Weighted RMS point-to-line distance [px].
  double isotropy = 0.0;      ///< lambda_min / lambda_max of the normal matrix.
  int iterations = 0;         ///< Number of IRLS iterations performed.
  bool converged = false;     ///< True if the estimate is valid.
};

/**
 * Förstner-type sub-pixel saddle point estimator with robust (Huber) IRLS.
 *
 * Every pixel p with image gradient g defines a line through p with normal
 * n = g / |g|. At a checkerboard saddle point all edges pass through the
 * centre c, so ideally n^T (p - c) = 0 for every edge pixel. The estimator
 * minimises
 *
 *     E(c) = sum_p  w_s(p) |g(p)|^gamma rho( n(p)^T (p - c) )
 *
 * with a Gaussian spatial weight w_s, gamma = gradient_weight_power
 * (classic Förstner: gamma = 2) and Huber's rho, by iteratively
 * reweighted least squares. Each iteration solves a 2x2 linear system via SVD
 * (Eigen). Because straight lines stay straight under a homography, the
 * model is exact for perspectively distorted targets.
 */
class SaddleRefiner {
 public:
  explicit SaddleRefiner(RefinerParams params = {});

  /// Refine a single initial guess. `image` must be single-channel
  /// (CV_8U or CV_32F). Only a small neighbourhood is processed.
  [[nodiscard]] RefineResult refine(const cv::Mat& image, cv::Point2d initial) const;

  [[nodiscard]] const RefinerParams& params() const noexcept { return params_; }

 private:
  RefinerParams params_;
};

}  // namespace gcp
