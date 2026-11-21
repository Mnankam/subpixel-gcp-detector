// gcp_detect — detect 2x2 checkerboard GCPs in images and print sub-pixel centres.
//
//   gcp_detect [--json] [--draw out.png] image1.jpg [image2.jpg ...]

#include "gcp/target_detector.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

namespace {

void usage() {
  std::cerr << "usage: gcp_detect [--json] [--draw out.png] <image> [<image> ...]\n"
               "  --json        print JSON instead of CSV\n"
               "  --draw FILE   write an annotated copy of the (first) image\n";
}

void draw(const cv::Mat& gray, const std::vector<gcp::Detection>& dets, const std::string& path) {
  cv::Mat vis;
  cv::cvtColor(gray, vis, cv::COLOR_GRAY2BGR);
  constexpr int kShift = 4;  // sub-pixel drawing
  constexpr double kScale = 1 << kShift;
  for (size_t i = 0; i < dets.size(); ++i) {
    const auto& d = dets[i];
    const cv::Point c(cvRound(d.center.x * kScale), cvRound(d.center.y * kScale));
    const int r = cvRound(0.5 * d.scale * kScale);
    cv::circle(vis, c, r, {0, 200, 0}, 1, cv::LINE_AA, kShift);
    cv::drawMarker(vis, cv::Point(cvRound(d.center.x), cvRound(d.center.y)), {0, 0, 255},
                   cv::MARKER_CROSS, 12, 1, cv::LINE_AA);
    cv::putText(vis, "#" + std::to_string(i),
                cv::Point(cvRound(d.center.x) + 8, cvRound(d.center.y) - 8),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 200, 255}, 1, cv::LINE_AA);
  }
  cv::imwrite(path, vis);
}

}  // namespace

int main(int argc, char** argv) {
  bool json = false;
  std::string draw_path;
  std::vector<std::string> files;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--json") {
      json = true;
    } else if (a == "--draw" && i + 1 < argc) {
      draw_path = argv[++i];
    } else if (a == "-h" || a == "--help") {
      usage();
      return 0;
    } else {
      files.push_back(a);
    }
  }
  if (files.empty()) {
    usage();
    return 2;
  }

  const gcp::TargetDetector detector;
  if (json)
    std::cout << "[\n";
  else
    std::cout << "file,id,x,y,scale,contrast,residual_rms,isotropy,time_ms\n";

  int exit_code = 0;
  for (size_t f = 0; f < files.size(); ++f) {
    const cv::Mat gray = cv::imread(files[f], cv::IMREAD_GRAYSCALE);
    if (gray.empty()) {
      std::cerr << "error: cannot read " << files[f] << "\n";
      exit_code = 1;
      continue;
    }
    const auto t0 = std::chrono::steady_clock::now();
    const auto dets = detector.detect(gray);
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    if (json) {
      std::printf("  {\"file\": \"%s\", \"time_ms\": %.2f, \"targets\": [", files[f].c_str(), ms);
      for (size_t i = 0; i < dets.size(); ++i) {
        const auto& d = dets[i];
        std::printf(
            "%s\n    {\"x\": %.4f, \"y\": %.4f, \"scale\": %.1f, \"contrast\": %.1f, "
            "\"residual_rms\": %.4f, \"isotropy\": %.3f}",
            i ? "," : "", d.center.x, d.center.y, d.scale, d.contrast, d.residual_rms, d.isotropy);
      }
      std::printf("%s]}%s\n", dets.empty() ? "" : "\n  ", f + 1 < files.size() ? "," : "");
    } else {
      for (size_t i = 0; i < dets.size(); ++i) {
        const auto& d = dets[i];
        std::printf("%s,%zu,%.4f,%.4f,%.1f,%.1f,%.4f,%.3f,%.2f\n", files[f].c_str(), i, d.center.x,
                    d.center.y, d.scale, d.contrast, d.residual_rms, d.isotropy, ms);
      }
    }
    if (f == 0 && !draw_path.empty()) draw(gray, dets, draw_path);
  }
  if (json) std::cout << "]\n";
  return exit_code;
}
