#include "gcp/saddle_refiner.hpp"

#include <Eigen/Dense>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <vector>

namespace gcp {

namespace {

struct Sample {
  double dx, dy;  // offset from the current estimate (integer on the c-centred grid)
  double nx, ny;  // unit gradient direction
  double w;       // spatial weight * gradient weight
};

/// Normalised Lanczos-3 weights for sampling at fractional offset t in [0, 1)
/// from integer taps -2..3.
std::array<double, 6> lanczos3Weights(double t) {
  std::array<double, 6> w{};
  double sum = 0.0;
  for (int k = -2; k <= 3; ++k) {
    const double x = t - k;
    double v = 1.0;
    if (std::abs(x) > 1e-12) {
      const double px = std::numbers::pi * x;
      v = 3.0 * std::sin(px) * std::sin(px / 3.0) / (px * px);
    }
    w[static_cast<size_t>(k + 2)] = v;
    sum += v;
  }
  for (auto& v : w) v /= sum;
  return w;
}

/// Exactly shift-resample an n x n patch whose (0,0) element is located at
/// image position `origin` (non-integer). Separable Lanczos-3, double
/// precision. Unlike cv::remap / cv::getRectSubPix this does not quantise the
/// sub-pixel phase to 1/32 px, which would otherwise cap the accuracy.
void resamplePatch(const cv::Mat& src, cv::Point2d origin, int n, cv::Mat& out) {
  const int bx = static_cast<int>(std::floor(origin.x));
  const int by = static_cast<int>(std::floor(origin.y));
  const auto wx = lanczos3Weights(origin.x - bx);
  const auto wy = lanczos3Weights(origin.y - by);

  // Horizontal pass over the rows needed by the vertical pass.
  cv::Mat tmp(n + 5, n, CV_64F);
  for (int r = 0; r < n + 5; ++r) {
    const int y = std::clamp(by - 2 + r, 0, src.rows - 1);
    const auto* row = src.ptr<float>(y);
    auto* t = tmp.ptr<double>(r);
    for (int i = 0; i < n; ++i) {
      double acc = 0.0;
      for (int k = 0; k < 6; ++k) {
        const int x = std::clamp(bx + i - 2 + k, 0, src.cols - 1);
        acc += wx[static_cast<size_t>(k)] * row[x];
      }
      t[i] = acc;
    }
  }
  out.create(n, n, CV_64F);
  for (int j = 0; j < n; ++j) {
    auto* o = out.ptr<double>(j);
    for (int i = 0; i < n; ++i) {
      double acc = 0.0;
      for (int k = 0; k < 6; ++k) acc += wy[static_cast<size_t>(k)] * tmp.at<double>(j + k, i);
      o[i] = acc;
    }
  }
}

inline double huberWeight(double r, double k) noexcept {
  const double a = std::abs(r);
  return a <= k ? 1.0 : k / a;
}

}  // namespace

SaddleRefiner::SaddleRefiner(RefinerParams params) : params_(params) {}

RefineResult SaddleRefiner::refine(const cv::Mat& image, cv::Point2d initial) const {
  CV_Assert(!image.empty() && image.channels() == 1);

  RefineResult result;
  result.center = initial;

  const int hw = std::max(2, params_.half_window);
  const int side = 2 * hw + 1;
  const double sigma = params_.spatial_sigma > 0.0 ? params_.spatial_sigma : 0.5 * hw;

  // Bounds check: the window may drift at most `hw` pixels from the guess.
  const double margin = 2.0 * hw + 3.0;
  if (initial.x < margin || initial.y < margin || initial.x > image.cols - 1 - margin ||
      initial.y > image.rows - 1 - margin) {
    return result;
  }

  // Only the region the iterations can touch is converted to float.
  const int r_roi = 2 * hw + 6;
  const cv::Rect roi = cv::Rect(cvRound(initial.x) - r_roi, cvRound(initial.y) - r_roi,
                                2 * r_roi + 1, 2 * r_roi + 1) &
                       cv::Rect(0, 0, image.cols, image.rows);
  cv::Mat src;
  image(roi).convertTo(src, CV_32F);
  const cv::Point2d roi_origin(roi.x, roi.y);

  // The grid is centred on c, so sample offsets are the same integers in
  // every iteration and the Gaussian spatial weights can be tabulated once.
  std::vector<double> spatial(static_cast<size_t>(side * side));
  const double zz2 = params_.zero_zone_radius * params_.zero_zone_radius;
  for (int j = -hw; j <= hw; ++j) {
    for (int i = -hw; i <= hw; ++i) {
      const double d2 = i * i + j * j;
      spatial[static_cast<size_t>((j + hw) * side + (i + hw))] =
          d2 < zz2 ? 0.0 : std::exp(-d2 / (2.0 * sigma * sigma));
    }
  }

  const double half_power = 0.5 * params_.gradient_weight_power;
  auto gradientWeight = [half_power](double g2) {
    return half_power == 1.0 ? g2 : (half_power == 0.5 ? std::sqrt(g2) : std::pow(g2, half_power));
  };

  // Resampling grid: (2hw+3)^2 points; the extra ring feeds the 3x3 Scharr kernel.
  const int n = side + 2;
  cv::Mat patch, gx, gy;
  std::vector<Sample> samples;
  samples.reserve(static_cast<size_t>(side * side));
  std::vector<std::pair<double, double>> scale_buf;  // (|residual|, weight)
  scale_buf.reserve(static_cast<size_t>(side * side));
  double k_abs = std::numeric_limits<double>::infinity();

  cv::Point2d c = initial;
  double step = 0.0;
  for (int it = 0; it < params_.max_iterations; ++it) {
    result.iterations = it + 1;

    // --- resample a patch centred exactly on c ----------------------------
    // Sampling the image at c + (i, j) keeps the estimation window
    // point-symmetric about the current estimate. A (blurred) checkerboard
    // saddle is locally point-symmetric, so all asymmetric contributions
    // cancel at the fixed point and the "pixel locking" bias of integer-grid
    // windows (cv::cornerSubPix) vanishes. Lanczos-3 keeps the interpolation
    // error far below the target accuracy.
    resamplePatch(src, c - roi_origin - cv::Point2d(hw + 1, hw + 1), n, patch);
    // Scharr kernels are close to rotation invariant, which matters for
    // gradient *directions*. Scale 1/32 yields grey levels per pixel.
    cv::Scharr(patch, gx, CV_64F, 1, 0, 1.0 / 32.0);
    cv::Scharr(patch, gy, CV_64F, 0, 1, 1.0 / 32.0);

    double g2_min = 1e-12;
    if (params_.min_gradient_ratio > 0.0) {
      double g2_max = 0.0;
      for (int j = 1; j <= side; ++j) {
        for (int i = 1; i <= side; ++i) {
          const double ggx = gx.at<double>(j, i);
          const double ggy = gy.at<double>(j, i);
          g2_max = std::max(g2_max, ggx * ggx + ggy * ggy);
        }
      }
      g2_min = std::max(g2_min, params_.min_gradient_ratio * params_.min_gradient_ratio * g2_max);
    }

    samples.clear();
    for (int j = -hw; j <= hw; ++j) {
      const auto* gxr = gx.ptr<double>(j + hw + 1) + hw + 1;
      const auto* gyr = gy.ptr<double>(j + hw + 1) + hw + 1;
      const double* wsr = &spatial[static_cast<size_t>((j + hw) * side + hw)];
      for (int i = -hw; i <= hw; ++i) {
        const double ws = wsr[i];
        if (ws <= 0.0) continue;
        const double ggx = gxr[i];
        const double ggy = gyr[i];
        const double g2 = ggx * ggx + ggy * ggy;
        if (g2 < g2_min) continue;
        const double inv = 1.0 / std::sqrt(g2);
        samples.push_back({static_cast<double>(i), static_cast<double>(j), ggx * inv, ggy * inv,
                           ws * gradientWeight(g2)});
      }
    }
    if (samples.size() < 8) return result;

    // --- robust scale: weighted median of |residual| -----------------------
    // The residual scale grows with blur (flank pixels of a wide edge sit a
    // few pixels off the edge line), so the Huber threshold must adapt.
    // Re-estimated in the first iterations only, where the estimate moves.
    if (std::isfinite(params_.huber_k) && it < 3) {
      scale_buf.clear();
      double wtot = 0.0;
      for (const auto& s : samples) {
        scale_buf.emplace_back(std::abs(s.nx * s.dx + s.ny * s.dy), s.w);
        wtot += s.w;
      }
      std::sort(scale_buf.begin(), scale_buf.end());
      double acc = 0.0;
      double med = 0.0;
      for (const auto& [r, w] : scale_buf) {
        acc += w;
        if (acc >= 0.5 * wtot) {
          med = r;
          break;
        }
      }
      k_abs = params_.huber_k * std::max(1.4826 * med, 0.25);
    }

    // --- one IRLS step: weighted normal equations for the update delta -----
    //   sum w n n^T delta = sum w n n^T (p - c)
    Eigen::Matrix2d A = Eigen::Matrix2d::Zero();
    Eigen::Vector2d b = Eigen::Vector2d::Zero();
    double wsum = 0.0;
    double wres2 = 0.0;
    for (const auto& s : samples) {
      const double r = s.nx * s.dx + s.ny * s.dy;  // signed point-to-line distance
      const double w = s.w * huberWeight(r, k_abs);
      A(0, 0) += w * s.nx * s.nx;
      A(0, 1) += w * s.nx * s.ny;
      A(1, 1) += w * s.ny * s.ny;
      b(0) += w * s.nx * r;
      b(1) += w * s.ny * r;
      wsum += w;
      wres2 += w * r * r;
    }
    A(1, 0) = A(0, 1);
    if (wsum <= 0.0) return result;

    const Eigen::JacobiSVD<Eigen::Matrix2d> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const auto& sv = svd.singularValues();
    result.isotropy = sv(0) > 0.0 ? sv(1) / sv(0) : 0.0;
    result.residual_rms = std::sqrt(wres2 / wsum);
    if (result.isotropy < params_.min_isotropy) return result;  // edge, not a saddle

    const Eigen::Vector2d delta = svd.solve(b);
    step = delta.norm();
    c += cv::Point2d(delta(0), delta(1));
    result.center = c;

    if (cv::norm(c - initial) > hw) return result;  // drifted away: reject
    if (step < params_.epsilon) {
      result.converged = true;
      return result;
    }
  }
  // Iteration budget exhausted: accept only if the estimate has settled.
  result.converged = step < 1e-2;
  return result;
}

}  // namespace gcp
