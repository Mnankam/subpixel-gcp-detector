// gcp_eval — Monte-Carlo accuracy benchmark against synthetic ground truth.
//
// Compares, per scenario:
//   * full pipeline (TargetDetector)          -> detection rate + error
//   * SaddleRefiner, robust (Huber IRLS)      -> error from a perturbed init
//   * SaddleRefiner, plain least squares      -> ablation of the Huber term
//   * cv::cornerSubPix (same window, same init)
//
//   gcp_eval [--trials N] [--seed S] [--samples DIR]

#include "gcp/saddle_refiner.hpp"
#include "gcp/synthetic.hpp"
#include "gcp/target_detector.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Stats {
  std::vector<double> err;
  double time_us = 0.0;
  int failures = 0;

  void add(double e) { err.push_back(e); }
  [[nodiscard]] double rmse() const {
    double s = 0.0;
    for (double e : err) s += e * e;
    return err.empty() ? NAN : std::sqrt(s / err.size());
  }
  [[nodiscard]] double pct(double q) const {
    if (err.empty()) return NAN;
    std::vector<double> v = err;
    const auto k = static_cast<size_t>(q * (v.size() - 1));
    std::nth_element(v.begin(), v.begin() + static_cast<long>(k), v.end());
    return v[k];
  }
};

struct Scenario {
  std::string name;
  std::function<void(gcp::SyntheticParams&, cv::RNG&)> configure;
};

}  // namespace

int main(int argc, char** argv) {
  int trials = 200;
  std::uint64_t seed = 2026;
  std::string samples_dir;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--trials" && i + 1 < argc)
      trials = std::stoi(argv[++i]);
    else if (a == "--seed" && i + 1 < argc)
      seed = std::stoull(argv[++i]);
    else if (a == "--samples" && i + 1 < argc)
      samples_dir = argv[++i];
  }

  const std::vector<Scenario> scenarios = {
      {"clean", [](auto&, auto&) {}},
      {"perspective",
       [](gcp::SyntheticParams& p, cv::RNG& r) {
         p.tilt_x_deg = r.uniform(-50.0, 50.0);
         p.tilt_y_deg = r.uniform(-50.0, 50.0);
         p.focal_px = 600.0;
       }},
      {"blur+noise+light",
       [](gcp::SyntheticParams& p, cv::RNG& r) {
         p.blur_sigma = r.uniform(1.0, 2.5);
         p.noise_sigma = r.uniform(4.0, 8.0);
         p.illumination_gradient = 0.5;
       }},
      {"cast shadow",
       [](gcp::SyntheticParams& p, cv::RNG&) {
         p.cast_shadow = true;
         p.shadow_factor = 0.45;
       }},
  };

  std::printf("# Sub-pixel GCP benchmark (%d trials per scenario, seed %llu)\n\n", trials,
              static_cast<unsigned long long>(seed));
  std::printf("Errors are Euclidean distances to the analytic ground truth in pixels.\n\n");
  std::printf("| Scenario | Method | RMSE | Median | P95 | Fail | Time/pt |\n");
  std::printf("|---|---|---:|---:|---:|---:|---:|\n");

  cv::RNG rng(seed);
  const gcp::TargetDetector detector;

  for (const auto& sc : scenarios) {
    Stats det, huber, ls, opencv;
    int false_pos = 0;
    for (int t = 0; t < trials; ++t) {
      gcp::SyntheticParams p;
      p.seed = rng.next();
      p.center = {rng.uniform(220.0, 420.0), rng.uniform(170.0, 310.0)};
      p.target_size_px = rng.uniform(60.0, 160.0);
      p.rotation_deg = rng.uniform(0.0, 90.0);
      p.noise_sigma = 2.0;
      sc.configure(p, rng);
      const auto s = gcp::renderTarget(p);

      if (!samples_dir.empty() && t == 0) {
        std::string fname = sc.name;
        std::replace_if(fname.begin(), fname.end(), [](char c) { return !std::isalnum(c); }, '_');
        cv::imwrite(samples_dir + "/" + fname + ".png", s.image);
      }

      // --- full pipeline ---
      auto t0 = Clock::now();
      const auto dets = detector.detect(s.image);
      det.time_us += std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
      double best = std::numeric_limits<double>::infinity();
      for (const auto& d : dets) best = std::min(best, cv::norm(d.center - s.center));
      if (best < 1.0) {
        det.add(best);
        false_pos += static_cast<int>(dets.size()) - 1;
      } else {
        ++det.failures;
        false_pos += static_cast<int>(dets.size());
      }

      // --- refiners from a perturbed initial guess (+-2 px) ---
      const cv::Point2d init =
          s.center + cv::Point2d(rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0));
      const int hw =
          std::clamp(static_cast<int>(std::lround(0.35 * 0.5 * p.target_size_px)), 4, 25);

      gcp::RefinerParams rp;
      rp.half_window = hw;
      auto runRefiner = [&](const gcp::RefinerParams& params, Stats& st) {
        const auto t1 = Clock::now();
        const auto r = gcp::SaddleRefiner(params).refine(s.image, init);
        st.time_us += std::chrono::duration<double, std::micro>(Clock::now() - t1).count();
        if (r.converged)
          st.add(cv::norm(r.center - s.center));
        else
          ++st.failures;
      };
      runRefiner(rp, huber);
      gcp::RefinerParams rp_ls = rp;
      rp_ls.huber_k = std::numeric_limits<double>::infinity();
      runRefiner(rp_ls, ls);

      std::vector<cv::Point2f> pts{
          cv::Point2f(static_cast<float>(init.x), static_cast<float>(init.y))};
      const auto t2 = Clock::now();
      cv::cornerSubPix(s.image, pts, cv::Size(hw, hw), cv::Size(-1, -1),
                       cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 1e-4));
      opencv.time_us += std::chrono::duration<double, std::micro>(Clock::now() - t2).count();
      opencv.add(cv::norm(cv::Point2d(pts[0]) - s.center));
    }

    auto row = [&](const char* method, const Stats& st, bool per_image) {
      const int n = static_cast<int>(st.err.size()) + st.failures;
      std::printf("| %s | %s | %.4f | %.4f | %.4f | %d/%d | %.0f %s |\n", sc.name.c_str(), method,
                  st.rmse(), st.pct(0.5), st.pct(0.95), st.failures, n, st.time_us / n,
                  per_image ? "µs/img" : "µs");
    };
    row("**Detector (full pipeline)**", det, true);
    row("Refiner Förstner + Huber", huber, false);
    row("Refiner Förstner, plain LS", ls, false);
    row("cv::cornerSubPix", opencv, false);
    if (false_pos > 0)
      std::printf("| %s | false positives | %d | | | | |\n", sc.name.c_str(), false_pos);
  }
  return 0;
}
