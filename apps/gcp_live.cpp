// gcp_live — real-time GCP tracking on an image stream.
//
//   gcp_live --source synthetic --frames 300
//   gcp_live --source pylon --emulator --emu-images ./docs   (PYLON_CAMEMU=1)
//   gcp_live --source pylon --exposure-us 2000 --gain-db 0 --show
//   gcp_live --source files img_*.png
//
// Prints one CSV line per tracked target and frame, and a latency / accuracy
// summary at the end.

#include "gcp/frame_source.hpp"
#include "gcp/io_sources.hpp"
#include "gcp/tracker.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#ifdef GCP_HAVE_HIGHGUI
#include <opencv2/highgui.hpp>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
  std::string source = "synthetic";
  int frames = 300;
  int redetect = 30;
  bool show = false;
  bool quiet = false;
  std::string save_dir;
  int save_every = 0;
  gcp::PylonSourceParams pylon;
  std::vector<std::string> files;
};

void usage() {
  std::cerr <<
      R"(usage: gcp_live [options] [image files...]
  --source NAME       synthetic (default) | pylon | files
  --frames N          stop after N frames (0 = endless, default 300)
  --redetect N        full detection every N frames (default 30)
  --quiet             no per-frame CSV, summary only
  --save-dir DIR      write annotated frames to DIR
  --save-every N      ... every N-th frame (default 30 when --save-dir is set)
  --show              live preview window (needs build with -DGCP_WITH_GUI=ON)
 pylon options:
  --emulator          use the pylon camera emulator (set PYLON_CAMEMU=1)
  --serial SN         open the camera with this serial number
  --exposure-us X     exposure time in microseconds
  --gain-db G         gain in dB
  --emu-images PATH   emulator replays these images (file or folder)
)";
}

bool parse(int argc, char** argv, Options& o) {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
      return argv[++i];
    };
    if (a == "--source")
      o.source = next();
    else if (a == "--frames")
      o.frames = std::stoi(next());
    else if (a == "--redetect")
      o.redetect = std::stoi(next());
    else if (a == "--quiet")
      o.quiet = true;
    else if (a == "--save-dir")
      o.save_dir = next();
    else if (a == "--save-every")
      o.save_every = std::stoi(next());
    else if (a == "--show")
      o.show = true;
    else if (a == "--emulator")
      o.pylon.emulator = true;
    else if (a == "--serial")
      o.pylon.serial = next();
    else if (a == "--exposure-us")
      o.pylon.exposure_us = std::stod(next());
    else if (a == "--gain-db")
      o.pylon.gain_db = std::stod(next());
    else if (a == "--emu-images")
      o.pylon.emulator_images = next();
    else if (a == "-h" || a == "--help")
      return false;
    else if (!a.empty() && a[0] == '-')
      throw std::runtime_error("unknown option " + a);
    else
      o.files.push_back(a);
  }
  if (!o.save_dir.empty() && o.save_every <= 0) o.save_every = 30;
  return true;
}

std::unique_ptr<gcp::FrameSource> openSource(const Options& o) {
  if (o.source == "synthetic") {
    gcp::SyntheticSourceParams p;
    p.frames = o.frames;
    return std::make_unique<gcp::SyntheticSource>(p);
  }
  if (o.source == "pylon") {
    gcp::PylonSourceParams p = o.pylon;
    p.max_frames = o.frames;
    return gcp::makePylonSource(p);
  }
  if (o.source == "files") return gcp::makeImageFileSource(o.files, false);
  throw std::runtime_error("unknown source '" + o.source + "'");
}

cv::Mat annotate(const gcp::Frame& f, const std::vector<gcp::Track>& tracks, bool full) {
  cv::Mat vis;
  cv::cvtColor(f.image, vis, cv::COLOR_GRAY2BGR);
  constexpr int kShift = 4;
  constexpr double kScale = 1 << kShift;
  for (const auto& t : tracks) {
    const auto& c = t.detection.center;
    const cv::Scalar col = t.misses ? cv::Scalar(0, 140, 255) : cv::Scalar(0, 220, 0);
    cv::circle(vis, cv::Point(cvRound(c.x * kScale), cvRound(c.y * kScale)),
               cvRound(0.5 * t.detection.scale * kScale), col, 2, cv::LINE_AA, kShift);
    cv::drawMarker(vis, cv::Point(cvRound(c.x), cvRound(c.y)), {0, 0, 255}, cv::MARKER_CROSS, 14, 1,
                   cv::LINE_AA);
    char buf[64];
    std::snprintf(buf, sizeof buf, "#%d (%.2f, %.2f)", t.id, c.x, c.y);
    const cv::Point org(cvRound(c.x) + 10, cvRound(c.y) - 10);
    cv::putText(vis, buf, org, cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 0, 0}, 3, cv::LINE_AA);
    cv::putText(vis, buf, org, cv::FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv::LINE_AA);
  }
  const std::string mode = full ? "DETECT" : "TRACK";
  cv::putText(vis, "frame " + std::to_string(f.index) + "  " + mode, {10, 24},
              cv::FONT_HERSHEY_SIMPLEX, 0.7, {255, 255, 0}, 2, cv::LINE_AA);
  return vis;
}

double percentile(std::vector<double> v, double q) {
  if (v.empty()) return NAN;
  const auto k = static_cast<size_t>(q * static_cast<double>(v.size() - 1));
  std::nth_element(v.begin(), v.begin() + static_cast<long>(k), v.end());
  return v[k];
}

}  // namespace

int main(int argc, char** argv) {
  Options opt;
  try {
    if (!parse(argc, argv, opt)) {
      usage();
      return 0;
    }
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    usage();
    return 2;
  }
#ifndef GCP_HAVE_HIGHGUI
  if (opt.show) {
    std::cerr << "warning: --show ignored (built without GCP_WITH_GUI)\n";
    opt.show = false;
  }
#endif

  std::unique_ptr<gcp::FrameSource> source;
  try {
    source = openSource(opt);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  std::cerr << "source: " << source->describe() << "\n";

  gcp::TrackerParams tp;
  tp.redetect_interval = opt.redetect;
  gcp::Tracker tracker(tp);

  std::vector<double> t_track, t_detect, t_grab, errors;
  int frames = 0;
  int gt_missed = 0;
  int full_runs = 0;
  const auto t_start = Clock::now();

  if (!opt.quiet) std::cout << "frame,track_id,x,y,scale,mode,proc_ms\n";

  gcp::Frame frame;
  while (true) {
    const auto t0 = Clock::now();
    bool ok = false;
    try {
      ok = source->grab(frame);
    } catch (const std::exception& e) {
      std::cerr << "error: " << e.what() << "\n";
      return 1;
    }
    if (!ok) break;
    const auto t1 = Clock::now();
    const auto& tracks = tracker.update(frame.image);
    const auto t2 = Clock::now();

    const double grab_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double proc_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    const bool full = tracker.lastWasFullDetection();
    t_grab.push_back(grab_ms);
    (full ? t_detect : t_track).push_back(proc_ms);
    full_runs += full;
    ++frames;

    // Accuracy against ground truth (synthetic source only).
    for (const auto& gt : frame.ground_truth) {
      double best = 1e9;
      for (const auto& t : tracks) {
        if (t.misses == 0) best = std::min(best, cv::norm(t.detection.center - gt));
      }
      if (best < 2.0)
        errors.push_back(best);
      else
        ++gt_missed;
    }

    if (!opt.quiet) {
      for (const auto& t : tracks) {
        if (t.misses) continue;
        std::printf("%llu,%d,%.4f,%.4f,%.1f,%s,%.3f\n",
                    static_cast<unsigned long long>(frame.index), t.id, t.detection.center.x,
                    t.detection.center.y, t.detection.scale, full ? "detect" : "track", proc_ms);
      }
    }

    const bool save =
        opt.save_every > 0 && frame.index % static_cast<unsigned>(opt.save_every) == 0;
    if (save || opt.show) {
      const cv::Mat vis = annotate(frame, tracks, full);
      if (save) {
        cv::imwrite(opt.save_dir + "/frame_" + std::to_string(frame.index) + ".png", vis);
      }
#ifdef GCP_HAVE_HIGHGUI
      if (opt.show) {
        cv::imshow("gcp_live", vis);
        const int key = cv::waitKey(1);
        if (key == 27 || key == 'q') break;
      }
#endif
    }
  }

  const double wall_s = std::chrono::duration<double>(Clock::now() - t_start).count();
  double sq = 0.0;
  for (double e : errors) sq += e * e;

  std::fprintf(stderr, "\n## gcp_live summary\n\n");
  std::fprintf(stderr, "| Metric | Value |\n|---|---:|\n");
  std::fprintf(stderr, "| Source | %s |\n", source->describe().c_str());
  std::fprintf(stderr, "| Frames | %d (%d full detections) |\n", frames, full_runs);
  std::fprintf(stderr, "| Throughput | %.1f fps |\n", wall_s > 0 ? frames / wall_s : 0.0);
  std::fprintf(stderr, "| Grab time, median | %.2f ms |\n", percentile(t_grab, 0.5));
  std::fprintf(stderr, "| Tracking update, median / P95 | %.2f / %.2f ms |\n",
               percentile(t_track, 0.5), percentile(t_track, 0.95));
  std::fprintf(stderr, "| Full detection, median / P95 | %.2f / %.2f ms |\n",
               percentile(t_detect, 0.5), percentile(t_detect, 0.95));
  if (!errors.empty()) {
    std::fprintf(stderr, "| Ground-truth RMSE | %.4f px (%zu measurements) |\n",
                 std::sqrt(sq / static_cast<double>(errors.size())), errors.size());
    std::fprintf(stderr, "| Targets missed | %d |\n", gt_missed);
  }
  return 0;
}
