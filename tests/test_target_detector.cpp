#include "gcp/synthetic.hpp"
#include "gcp/target_detector.hpp"

#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>

using gcp::DetectorParams;
using gcp::SyntheticParams;
using gcp::TargetDetector;

TEST(RingCheck, AcceptsSaddleRejectsCorner) {
  SyntheticParams p;
  p.noise_sigma = 0.5;
  const auto s = gcp::renderTarget(p);
  const DetectorParams dp;
  EXPECT_TRUE(gcp::checkSaddleRing(s.image, s.center, 20.0, dp).valid);
  // Outer corner of the 2x2 pattern: only two sectors.
  const cv::Point2d corner = s.center + cv::Point2d(80.0, 80.0);
  const auto rc = gcp::checkSaddleRing(s.image, corner, 10.0, dp);
  EXPECT_FALSE(rc.valid);
}

TEST(TargetDetector, FindsSingleTargetWithArtefacts) {
  SyntheticParams p;
  p.center = {333.27, 228.64};
  p.rotation_deg = 12.0;
  p.tilt_x_deg = 30.0;
  p.illumination_gradient = 0.4;
  p.add_screws = true;
  p.noise_sigma = 3.0;
  const auto s = gcp::renderTarget(p);

  const auto dets = TargetDetector().detect(s.image);
  ASSERT_EQ(dets.size(), 1u);
  EXPECT_LT(cv::norm(dets[0].center - s.center), 0.08);
}

TEST(TargetDetector, FindsMultipleTargets) {
  std::vector<SyntheticParams> ts(3);
  ts[0].image_size = {960, 540};
  ts[0].center = {180.3, 160.7};
  ts[0].target_size_px = 110;
  ts[1].center = {500.8, 330.1};
  ts[1].target_size_px = 90;
  ts[1].rotation_deg = 40;
  ts[1].tilt_y_deg = 35;
  ts[2].center = {790.45, 170.2};
  ts[2].target_size_px = 70;
  ts[2].tilt_x_deg = -40;
  std::vector<cv::Point2d> gt;
  const cv::Mat img = gcp::renderScene(ts, &gt);

  const auto dets = TargetDetector().detect(img);
  ASSERT_EQ(dets.size(), gt.size());
  for (const auto& g : gt) {
    double best = 1e9;
    for (const auto& d : dets) best = std::min(best, cv::norm(d.center - g));
    EXPECT_LT(best, 0.1);
  }
}

TEST(TargetDetector, NoFalsePositivesOnClutter) {
  // Background texture, a single dark square, a shadow edge — no saddle.
  SyntheticParams p;
  p.background_texture = 40.0;
  p.noise_sigma = 3.0;
  p.target_size_px = 0.0;  // no target
  p.panel_margin = 0.0;
  p.add_screws = false;
  cv::Mat img = gcp::renderTarget(p).image;
  cv::rectangle(img, cv::Rect(100, 100, 80, 80), cv::Scalar(30), cv::FILLED);
  cv::rectangle(img, cv::Rect(400, 250, 120, 60), cv::Scalar(220), cv::FILLED);
  img(cv::Rect(0, 350, 640, 130)) *= 0.5;

  EXPECT_TRUE(TargetDetector().detect(img).empty());
}

TEST(TargetDetector, AcceptsColorInput) {
  SyntheticParams p;
  const auto s = gcp::renderTarget(p);
  cv::Mat bgr;
  cv::cvtColor(s.image, bgr, cv::COLOR_GRAY2BGR);
  const auto dets = TargetDetector().detect(bgr);
  ASSERT_EQ(dets.size(), 1u);
  EXPECT_LT(cv::norm(dets[0].center - s.center), 0.08);
}

TEST(TargetDetector, SurvivesCastShadow) {
  int found = 0;
  for (std::uint64_t seed = 1; seed <= 8; ++seed) {
    SyntheticParams p;
    p.center = {320.4, 240.7};
    p.rotation_deg = 15.0 + 7.0 * static_cast<double>(seed);
    p.cast_shadow = true;
    p.shadow_factor = 0.45;
    p.seed = seed;
    const auto s = gcp::renderTarget(p);
    const auto dets = TargetDetector().detect(s.image);
    ASSERT_LE(dets.size(), 1u) << "false positive, seed " << seed;
    if (!dets.empty() && cv::norm(dets[0].center - s.center) < 0.1) ++found;
  }
  EXPECT_GE(found, 7);
}
