#include "gcp/frame_source.hpp"
#include "gcp/io_sources.hpp"
#include "gcp/tracker.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <map>

TEST(SyntheticSource, DeliversGroundTruthAndStops) {
  gcp::SyntheticSourceParams p;
  p.image_size = {640, 360};
  p.frames = 3;
  p.targets = 2;
  gcp::SyntheticSource src(p);
  gcp::Frame f;
  int n = 0;
  while (src.grab(f)) {
    EXPECT_EQ(f.image.type(), CV_8UC1);
    EXPECT_EQ(f.ground_truth.size(), 2u);
    EXPECT_EQ(f.index, static_cast<std::uint64_t>(n));
    ++n;
  }
  EXPECT_EQ(n, 3);
}

TEST(Tracker, FollowsMovingTargetsWithStableIds) {
  gcp::SyntheticSourceParams sp;
  sp.image_size = {800, 450};
  sp.frames = 60;
  sp.targets = 2;
  sp.target_size_px = 110;
  gcp::SyntheticSource src(sp);

  gcp::TrackerParams tp;
  tp.redetect_interval = 30;
  gcp::Tracker tracker(tp);

  gcp::Frame f;
  int full = 0;
  double sq = 0.0;
  int n = 0;
  std::map<int, int> id_of_gt;  // ground-truth index -> track id
  while (src.grab(f)) {
    const auto& tracks = tracker.update(f.image);
    full += tracker.lastWasFullDetection();
    ASSERT_EQ(tracks.size(), 2u) << "frame " << f.index;
    for (size_t g = 0; g < f.ground_truth.size(); ++g) {
      const gcp::Track* best = nullptr;
      double dist = 1e9;
      for (const auto& t : tracks) {
        const double d = cv::norm(t.detection.center - f.ground_truth[g]);
        if (d < dist) {
          dist = d;
          best = &t;
        }
      }
      ASSERT_NE(best, nullptr);
      EXPECT_LT(dist, 0.15) << "frame " << f.index;
      sq += dist * dist;
      ++n;
      const auto [it, inserted] = id_of_gt.emplace(static_cast<int>(g), best->id);
      if (!inserted) {
        EXPECT_EQ(it->second, best->id) << "identity switch at frame " << f.index;
      }
    }
  }
  EXPECT_LT(std::sqrt(sq / n), 0.06);
  // Frame 0 and frame 30: the cheap tracking path handles everything else.
  EXPECT_LE(full, 3);
}

TEST(Tracker, ReacquiresAfterTargetDisappears) {
  gcp::SyntheticSourceParams sp;
  sp.image_size = {640, 360};
  sp.frames = 5;
  sp.targets = 1;
  gcp::SyntheticSource src(sp);
  gcp::Tracker tracker;
  gcp::Frame f;
  ASSERT_TRUE(src.grab(f));
  EXPECT_EQ(tracker.update(f.image).size(), 1u);

  const cv::Mat empty(f.image.size(), CV_8UC1, cv::Scalar(110));
  for (int i = 0; i < 3; ++i) (void)tracker.update(empty);
  EXPECT_TRUE(tracker.tracks().empty());

  ASSERT_TRUE(src.grab(f));
  EXPECT_EQ(tracker.update(f.image).size(), 1u);
  EXPECT_TRUE(tracker.lastWasFullDetection());
}

TEST(ImageFileSource, SkipsUnreadableAndEnds) {
  EXPECT_THROW((void)gcp::makeImageFileSource({}), std::runtime_error);
  gcp::Frame f;
  auto once = gcp::makeImageFileSource({"/nonexistent/a.png"});
  EXPECT_FALSE(once->grab(f));  // unreadable file skipped, then end of stream
  auto looping = gcp::makeImageFileSource({"/nonexistent/a.png"}, true);
  EXPECT_THROW((void)looping->grab(f), std::runtime_error);  // no endless loop
}

TEST(PylonSource, ReportsAvailability) {
  if (gcp::pylonAvailable()) GTEST_SKIP() << "pylon build: needs a camera or the emulator";
  EXPECT_THROW((void)gcp::makePylonSource({}), std::runtime_error);
}
