#include "gcp/saddle_refiner.hpp"
#include "gcp/synthetic.hpp"

#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>

#include <limits>

using gcp::RefinerParams;
using gcp::SaddleRefiner;
using gcp::SyntheticParams;

namespace {

double refineError(const SyntheticParams& p, const RefinerParams& rp, cv::Point2d offset) {
  const auto s = gcp::renderTarget(p);
  const auto r = SaddleRefiner(rp).refine(s.image, s.center + offset);
  EXPECT_TRUE(r.converged);
  return cv::norm(r.center - s.center);
}

}  // namespace

// Ground truth on a dense sub-pixel grid, fronto-parallel, mild noise.
class SubpixelGrid : public ::testing::TestWithParam<std::pair<double, double>> {};

TEST_P(SubpixelGrid, FrontoParallelAccuracy) {
  SyntheticParams p;
  p.center = {320.0 + GetParam().first, 240.0 + GetParam().second};
  p.noise_sigma = 1.0;
  // Axis-aligned edges with little blur are the worst case (aliasing).
  EXPECT_LT(refineError(p, {}, {1.3, -1.1}), 0.04);
}

INSTANTIATE_TEST_SUITE_P(Offsets, SubpixelGrid,
                         ::testing::Values(std::pair{0.0, 0.0}, std::pair{0.25, 0.0},
                                           std::pair{0.5, 0.5}, std::pair{0.13, 0.77},
                                           std::pair{0.91, 0.42}, std::pair{0.66, 0.05}));

TEST(SaddleRefiner, PerspectiveAndRotation) {
  SyntheticParams p;
  p.center = {311.42, 247.18};
  p.rotation_deg = 27.0;
  p.tilt_x_deg = 45.0;
  p.tilt_y_deg = 25.0;
  p.focal_px = 600.0;
  p.noise_sigma = 2.0;
  EXPECT_LT(refineError(p, {}, {-2.0, 1.5}), 0.06);
}

TEST(SaddleRefiner, BlurDoesNotBiasCenter) {
  SyntheticParams p;
  p.center = {300.6, 200.3};
  p.blur_sigma = 2.0;
  p.rotation_deg = 10.0;
  p.noise_sigma = 0.0;  // isolate bias from noise
  p.material_texture = 0.0;
  RefinerParams rp;
  rp.half_window = 10;
  EXPECT_LT(refineError(p, rp, {1.0, 1.0}), 0.02);
}

TEST(SaddleRefiner, HuberSuppressesShadowEdge) {
  // A cast shadow edge crosses the estimation window but not the saddle.
  double err_ls = 0.0;
  double err_huber = 0.0;
  for (std::uint64_t seed = 1; seed <= 10; ++seed) {
    SyntheticParams p;
    p.center = {320.37, 240.61};
    p.target_size_px = 60.0;
    p.cast_shadow = true;
    p.shadow_factor = 0.4;
    p.seed = seed;
    RefinerParams huber;
    huber.half_window = 12;
    RefinerParams ls = huber;
    ls.huber_k = std::numeric_limits<double>::infinity();
    err_ls += refineError(p, ls, {});
    err_huber += refineError(p, huber, {});
  }
  EXPECT_LT(err_huber, err_ls);
  EXPECT_LT(err_huber / 10.0, 0.1);
}

TEST(SaddleRefiner, LowerSystematicErrorThanCornerSubPix) {
  // Noise-free: measures the estimator bias ("pixel locking"), not noise.
  cv::RNG rng(123);
  double ours = 0.0;
  double opencv = 0.0;
  constexpr int kTrials = 25;
  for (int t = 0; t < kTrials; ++t) {
    SyntheticParams p;
    p.center = {rng.uniform(250.0, 390.0), rng.uniform(180.0, 300.0)};
    p.rotation_deg = rng.uniform(0.0, 90.0);
    p.tilt_x_deg = rng.uniform(-30.0, 30.0);
    p.blur_sigma = rng.uniform(0.5, 1.5);
    p.noise_sigma = 0.0;
    p.material_texture = 0.0;
    p.seed = static_cast<std::uint64_t>(t + 1);
    const auto s = gcp::renderTarget(p);
    const cv::Point2d init = s.center + cv::Point2d(rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0));

    RefinerParams rp;
    rp.half_window = 9;
    ours += cv::norm(SaddleRefiner(rp).refine(s.image, init).center - s.center);

    std::vector<cv::Point2f> pts{cv::Point2f(init)};
    cv::cornerSubPix(s.image, pts, cv::Size(9, 9), cv::Size(-1, -1),
                     cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 1e-4));
    opencv += cv::norm(cv::Point2d(pts[0]) - s.center);
  }
  EXPECT_LT(ours, 0.5 * opencv) << "mean ours " << ours / kTrials << " px, cornerSubPix "
                                << opencv / kTrials << " px";
}

TEST(SaddleRefiner, RejectsStraightEdge) {
  cv::Mat img(100, 100, CV_8U, cv::Scalar(40));
  img(cv::Rect(50, 0, 50, 100)).setTo(200);
  const auto r = SaddleRefiner().refine(img, {50.0, 50.0});
  EXPECT_FALSE(r.converged);
  EXPECT_LT(r.isotropy, 0.05);
}

TEST(SaddleRefiner, RejectsFlatRegion) {
  const cv::Mat img(100, 100, CV_8U, cv::Scalar(128));
  EXPECT_FALSE(SaddleRefiner().refine(img, {50.0, 50.0}).converged);
}

TEST(SaddleRefiner, HandlesImageBorder) {
  SyntheticParams p;
  p.center = {6.4, 7.2};
  const auto s = gcp::renderTarget(p);
  // Must not crash; result validity is not required this close to the border.
  (void)SaddleRefiner().refine(s.image, {6.0, 7.0});
  SUCCEED();
}
