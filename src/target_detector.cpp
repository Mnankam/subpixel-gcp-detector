#include "gcp/target_detector.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

namespace gcp {

namespace {

struct Candidate {
  cv::Point2d p;
  double scale;
};

/// Bilinear sample; returns false if outside the image.
inline bool sampleBilinear(const cv::Mat& img, double x, double y, double& out) {
  const int x0 = static_cast<int>(std::floor(x));
  const int y0 = static_cast<int>(std::floor(y));
  if (x0 < 0 || y0 < 0 || x0 + 1 >= img.cols || y0 + 1 >= img.rows) return false;
  const double ax = x - x0;
  const double ay = y - y0;
  const auto* r0 = img.ptr<uchar>(y0);
  const auto* r1 = img.ptr<uchar>(y0 + 1);
  out =
      (1 - ay) * ((1 - ax) * r0[x0] + ax * r0[x0 + 1]) + ay * ((1 - ax) * r1[x0] + ax * r1[x0 + 1]);
  return true;
}

cv::Mat toGray8(const cv::Mat& image) {
  cv::Mat gray;
  if (image.channels() == 3) {
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  } else if (image.channels() == 4) {
    cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
  } else {
    gray = image;
  }
  if (gray.depth() != CV_8U) {
    cv::Mat tmp;
    cv::normalize(gray, tmp, 0, 255, cv::NORM_MINMAX, CV_8U);
    gray = tmp;
  }
  return gray;
}

/// Extract convex quadrilaterals from a binary image and push their vertices.
void collectQuadVertices(const cv::Mat& binary, const DetectorParams& p, double max_area,
                         std::vector<Candidate>& out) {
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
  std::vector<cv::Point> poly;
  for (const auto& contour : contours) {
    const double area = std::abs(cv::contourArea(contour));
    if (area < p.min_quad_area || area > max_area) continue;
    cv::approxPolyDP(contour, poly, p.polygon_epsilon_ratio * cv::arcLength(contour, true), true);
    if (poly.size() != 4 || !cv::isContourConvex(poly)) continue;
    // A quad that only approximates a round blob (screw heads, bolts) covers
    // far less area than the blob itself (inscribed square: 2/pi).
    const double poly_area = std::abs(cv::contourArea(poly));
    if (poly_area <= 0.0 || std::abs(area / poly_area - 1.0) > p.max_quad_fill_deviation) continue;

    double smin = 1e300;
    double smax = 0.0;
    for (size_t i = 0; i < 4; ++i) {
      const double s = cv::norm(poly[i] - poly[(i + 1) % 4]);
      smin = std::min(smin, s);
      smax = std::max(smax, s);
    }
    if (smin < p.min_quad_side_ratio * smax) continue;

    const double scale = std::sqrt(area);
    for (const auto& v : poly) out.push_back({cv::Point2d(v), scale});
  }
}

/// Greedy merge of candidates closer than a fraction of their scale.
std::vector<Candidate> mergeCandidates(std::vector<Candidate> in) {
  std::sort(in.begin(), in.end(),
            [](const Candidate& a, const Candidate& b) { return a.scale > b.scale; });
  std::vector<Candidate> out;
  std::vector<int> count;
  for (const auto& c : in) {
    bool merged = false;
    for (size_t i = 0; i < out.size(); ++i) {
      const double r = std::max(3.0, 0.15 * std::min(out[i].scale, c.scale));
      if (cv::norm(out[i].p - c.p) < r) {
        const int n = count[i]++;
        out[i].p = (out[i].p * n + c.p) / (n + 1);
        merged = true;
        break;
      }
    }
    if (!merged) {
      out.push_back(c);
      count.push_back(1);
    }
  }
  return out;
}

}  // namespace

RingCheck checkSaddleRing(const cv::Mat& gray, cv::Point2d c, double radius,
                          const DetectorParams& params) {
  RingCheck rc;
  const int n = std::max(16, params.ring_samples / 2 * 2);  // even
  std::vector<double> vals(static_cast<size_t>(n));
  double vmin = 1e300;
  double vmax = -1e300;
  for (int i = 0; i < n; ++i) {
    const double a = 2.0 * std::numbers::pi * i / n;
    double v = 0.0;
    if (!sampleBilinear(gray, c.x + radius * std::cos(a), c.y + radius * std::sin(a), v)) {
      return rc;
    }
    vals[static_cast<size_t>(i)] = v;
    vmin = std::min(vmin, v);
    vmax = std::max(vmax, v);
  }
  if (vmax - vmin < params.min_contrast) return rc;

  // Local threshold: mean over the half circle centred on each sample. For
  // an ideal 4-sector pattern every half circle holds equal parts dark and
  // light, so this equals the global midpoint - but it adapts to cast
  // shadows and illumination gradients that scale one side of the ring.
  const int half = n / 2;
  std::vector<double> prefix(static_cast<size_t>(3 * n + 1), 0.0);  // cyclic prefix sums
  for (int i = 0; i < 3 * n; ++i) {
    prefix[static_cast<size_t>(i + 1)] =
        prefix[static_cast<size_t>(i)] + vals[static_cast<size_t>(i % n)];
  }
  std::vector<char> lab(static_cast<size_t>(n));
  double contrast_sum = 0.0;
  int n_hi = 0;
  for (int i = 0; i < n; ++i) {
    const int lo = i - half / 2 + n;  // window [i - n/4, i + n/4)
    const double thr =
        (prefix[static_cast<size_t>(lo + half)] - prefix[static_cast<size_t>(lo)]) / half;
    const double d = vals[static_cast<size_t>(i)] - thr;
    lab[static_cast<size_t>(i)] = d > 0.0;
    n_hi += d > 0.0;
    contrast_sum += std::abs(d);
  }
  if (n_hi == 0 || n_hi == n) return rc;
  rc.contrast = 2.0 * contrast_sum / n;  // = light - dark for a clean pattern

  // Count transitions and measure the shortest run (sector length).
  int start = 0;
  while (start < n &&
         lab[static_cast<size_t>(start)] == lab[static_cast<size_t>((start + n - 1) % n)]) {
    ++start;
  }
  int transitions = 0;
  int shortest = n;
  int run = 0;
  for (int k = 0; k < n; ++k) {
    const int i = (start + k) % n;
    const int prev = (i + n - 1) % n;
    if (k > 0 && lab[static_cast<size_t>(i)] != lab[static_cast<size_t>(prev)]) {
      ++transitions;
      shortest = std::min(shortest, run);
      run = 0;
    }
    ++run;
  }
  if (start < n) {  // close the cyclic run
    ++transitions;
    shortest = std::min(shortest, run);
  }
  rc.transitions = transitions;

  int agree = 0;
  for (int i = 0; i < n / 2; ++i) {
    agree += lab[static_cast<size_t>(i)] == lab[static_cast<size_t>(i + n / 2)];
  }
  const double agreement = static_cast<double>(agree) / (n / 2);

  rc.valid = transitions == 4 && rc.contrast >= params.min_contrast &&
             agreement >= params.min_antipodal_agreement &&
             shortest >= params.min_sector_fraction * n;
  return rc;
}

TargetDetector::TargetDetector(DetectorParams params) : params_(std::move(params)) {}

std::vector<Detection> TargetDetector::detect(const cv::Mat& image) const {
  CV_Assert(!image.empty());
  const DetectorParams& p = params_;
  const cv::Mat gray = toGray8(image);

  // ---------------- Phase A: pre-processing ---------------------------------
  cv::Mat enhanced;
  auto clahe = cv::createCLAHE(p.clahe_clip_limit, cv::Size(p.clahe_tiles, p.clahe_tiles));
  clahe->apply(gray, enhanced);

  const int min_dim = std::min(gray.cols, gray.rows);
  const double max_area = p.max_quad_area_ratio * gray.cols * gray.rows;
  const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));

  std::vector<Candidate> raw;
  cv::Mat binary;
  auto harvest = [&](cv::Mat& bin) {
    if (p.split_erosion > 0) {
      cv::erode(bin, bin, kernel, cv::Point(-1, -1), p.split_erosion);
    }
    collectQuadVertices(bin, p, max_area, raw);
  };

  // Two adaptive block sizes (scale robustness) + global Otsu, both polarities.
  for (const double ratio : {p.adaptive_block_ratio, 2.0 * p.adaptive_block_ratio}) {
    const int block = std::max(3, static_cast<int>(min_dim * ratio)) | 1;
    for (const int type : {cv::THRESH_BINARY_INV, cv::THRESH_BINARY}) {
      cv::adaptiveThreshold(enhanced, binary, 255, cv::ADAPTIVE_THRESH_MEAN_C, type, block,
                            p.adaptive_offset);
      harvest(binary);
    }
  }
  for (const int type : {cv::THRESH_BINARY_INV, cv::THRESH_BINARY}) {
    cv::threshold(enhanced, binary, 0, 255, type | cv::THRESH_OTSU);
    harvest(binary);
  }

  const std::vector<Candidate> candidates = mergeCandidates(std::move(raw));

  // ---------------- Phase A': ring validation + Phase B: refinement ---------
  std::vector<Detection> detections;
  // A vertex of an eroded quad is only ~1-3 px from the true centre, which
  // already breaks the central symmetry on small or foreshortened targets.
  // Hence: loose ring test -> refinement -> strict ring test at the
  // refined centre.
  DetectorParams loose = p;
  loose.min_antipodal_agreement = 0.5;
  loose.min_sector_fraction = 0.5 * p.min_sector_fraction;

  for (const auto& cand : candidates) {
    // Several quad vertices map to the same target: skip the expensive
    // refinement once a target has been confirmed nearby.
    const bool known = std::any_of(detections.begin(), detections.end(), [&](const Detection& d) {
      return cv::norm(d.center - cand.p) < 0.25 * std::min(d.scale, cand.scale);
    });
    if (known) continue;
    const double r_inner = std::max(3.0, 0.3 * cand.scale);
    const double r_outer = std::max(5.0, 0.6 * cand.scale);
    if (!checkSaddleRing(gray, cand.p, r_outer, loose).valid) continue;

    RefinerParams rp = p.refiner;
    rp.half_window = std::clamp(static_cast<int>(std::lround(0.35 * cand.scale)), p.min_half_window,
                                p.max_half_window);
    const RefineResult res = SaddleRefiner(rp).refine(gray, cand.p);
    if (!res.converged || res.residual_rms > p.max_residual_ratio * rp.half_window) continue;
    if (cv::norm(res.center - cand.p) > 0.3 * cand.scale) continue;

    const RingCheck inner = checkSaddleRing(gray, res.center, r_inner, p);
    if (!inner.valid) continue;
    const RingCheck outer = checkSaddleRing(gray, res.center, r_outer, p);
    if (!outer.valid) continue;

    detections.push_back({res.center, cand.scale, std::min(inner.contrast, outer.contrast),
                          res.residual_rms, res.isotropy});
  }

  // Non-maximum suppression (several quads share one centre).
  std::sort(detections.begin(), detections.end(),
            [](const Detection& a, const Detection& b) { return a.contrast > b.contrast; });
  std::vector<Detection> kept;
  for (const auto& d : detections) {
    const bool dup = std::any_of(kept.begin(), kept.end(), [&](const Detection& k) {
      return cv::norm(k.center - d.center) < 0.5 * std::min(k.scale, d.scale);
    });
    if (!dup) kept.push_back(d);
  }
  return kept;
}

}  // namespace gcp
