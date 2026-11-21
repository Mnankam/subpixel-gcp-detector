#include "gcp/synthetic.hpp"

#include <gtest/gtest.h>

using gcp::SyntheticParams;

TEST(Synthetic, CenterProjectsToGroundTruth) {
  SyntheticParams p;
  p.center = {301.37, 222.81};
  p.tilt_x_deg = 35.0;
  p.tilt_y_deg = -20.0;
  p.rotation_deg = 17.0;
  const auto s = gcp::renderTarget(p);
  const cv::Vec3d h = s.H * cv::Vec3d(0, 0, 1);
  EXPECT_NEAR(h[0] / h[2], p.center.x, 1e-9);
  EXPECT_NEAR(h[1] / h[2], p.center.y, 1e-9);
}

TEST(Synthetic, IsDeterministicForSeed) {
  SyntheticParams p;
  p.seed = 7;
  const auto a = gcp::renderTarget(p);
  const auto b = gcp::renderTarget(p);
  EXPECT_EQ(cv::norm(a.image, b.image, cv::NORM_INF), 0.0);
}

TEST(Synthetic, QuadrantsHaveExpectedPolarity) {
  SyntheticParams p;
  p.noise_sigma = 0.0;
  p.illumination_gradient = 0.0;
  const auto s = gcp::renderTarget(p);
  const double q = 0.25 * p.target_size_px;
  auto at = [&](double dx, double dy) {
    return s.image.at<uchar>(cvRound(p.center.y + dy), cvRound(p.center.x + dx));
  };
  EXPECT_LT(at(q, q), 80);    // dark
  EXPECT_LT(at(-q, -q), 80);  // dark
  EXPECT_GT(at(q, -q), 170);  // light
  EXPECT_GT(at(-q, q), 170);  // light
}
