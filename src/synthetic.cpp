#include "gcp/synthetic.hpp"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <numbers>

namespace gcp {

namespace {

constexpr double kDeg = std::numbers::pi / 180.0;

cv::Matx33d buildHomography(const SyntheticParams& p) {
  const double ax = p.tilt_x_deg * kDeg;
  const double ay = p.tilt_y_deg * kDeg;
  const double az = p.rotation_deg * kDeg;
  const cv::Matx33d Rx(1, 0, 0, 0, std::cos(ax), -std::sin(ax), 0, std::sin(ax), std::cos(ax));
  const cv::Matx33d Ry(std::cos(ay), 0, std::sin(ay), 0, 1, 0, -std::sin(ay), 0, std::cos(ay));
  const cv::Matx33d Rz(std::cos(az), -std::sin(az), 0, std::sin(az), std::cos(az), 0, 0, 0, 1);
  const cv::Matx33d R = Rx * Ry * Rz;
  const double f = p.focal_px;
  // Plane at depth f through the optical axis -> plane units == pixels at the
  // centre; the plane origin projects exactly onto the principal point.
  const cv::Matx33d K(f, 0, p.center.x, 0, f, p.center.y, 0, 0, 1);
  const cv::Matx33d Rt(R(0, 0), R(0, 1), 0.0, R(1, 0), R(1, 1), 0.0, R(2, 0), R(2, 1), f);
  return K * Rt;
}

enum Region : int { kBackground = 0, kDark, kLight, kPanel, kScrew };

struct PlaneModel {
  double half;        // half size of the 2x2 pattern
  double panel_half;  // half size of the panel
  double screw_off;   // screw centre offset from origin (both axes)
  double screw_r;
  bool screws;

  [[nodiscard]] Region classify(double u, double v) const {
    const double au = std::abs(u);
    const double av = std::abs(v);
    if (au > panel_half || av > panel_half) return kBackground;
    if (au <= half && av <= half) return (u * v > 0.0) ? kDark : kLight;
    if (screws) {
      const double du = au - screw_off;
      const double dv = av - screw_off;
      if (du * du + dv * dv <= screw_r * screw_r) return kScrew;
    }
    return kPanel;
  }
};

/// Render one target layer as premultiplied value + coverage at image
/// resolution. The optical blur (PSF) is applied on a 4x oversampled grid
/// *before* pixel integration, as in a real camera. Blurring an already
/// pixel-integrated image instead would bake aliasing into the "ground truth"
/// and bias every estimator towards pixel centres.
void renderLayer(const SyntheticParams& p, cv::Mat& premult, cv::Mat& coverage) {
  const cv::Matx33d H = buildHomography(p);
  const cv::Matx33d Hi = H.inv();

  const double S = p.target_size_px;
  const PlaneModel model{0.5 * S, 0.5 * S + p.panel_margin * S, 0.5 * S + 0.5 * p.panel_margin * S,
                         0.25 * p.panel_margin * S, p.add_screws};
  const double screw_value = 0.45 * 0.5 * (p.dark + p.light);
  const double region_value[] = {0.0, p.dark, p.light, p.light, screw_value};

  premult = cv::Mat::zeros(p.image_size, CV_64F);
  coverage = cv::Mat::zeros(p.image_size, CV_64F);
  if (S <= 0.0) return;

  // Bounding box of the panel in the image, padded for the blur kernel.
  double x0 = 1e300, y0 = 1e300, x1 = -1e300, y1 = -1e300;
  for (const double su : {-1.0, 1.0}) {
    for (const double sv : {-1.0, 1.0}) {
      const cv::Vec3d q = H * cv::Vec3d(su * model.panel_half, sv * model.panel_half, 1.0);
      if (q[2] <= 0.0) return;  // panel crosses the camera plane
      x0 = std::min(x0, q[0] / q[2]);
      x1 = std::max(x1, q[0] / q[2]);
      y0 = std::min(y0, q[1] / q[2]);
      y1 = std::max(y1, q[1] / q[2]);
    }
  }
  const int pad = static_cast<int>(std::ceil(4.0 * p.blur_sigma)) + 3;
  const cv::Rect bbox = cv::Rect(cv::Point(static_cast<int>(std::floor(x0)) - pad,
                                           static_cast<int>(std::floor(y0)) - pad),
                                 cv::Point(static_cast<int>(std::ceil(x1)) + pad + 1,
                                           static_cast<int>(std::ceil(y1)) + pad + 1)) &
                        cv::Rect(0, 0, p.image_size.width, p.image_size.height);
  if (bbox.empty()) return;

  auto classifyImage = [&](double x, double y) {
    const double w = Hi(2, 0) * x + Hi(2, 1) * y + Hi(2, 2);
    if (w <= 0.0) return kBackground;
    const double u = (Hi(0, 0) * x + Hi(0, 1) * y + Hi(0, 2)) / w;
    const double v = (Hi(1, 0) * x + Hi(1, 1) * y + Hi(1, 2)) / w;
    return model.classify(u, v);
  };

  constexpr int F = 4;  // oversampling factor of the PSF grid
  const int sub = std::max(1, p.supersampling / F);
  const double step = 1.0 / F;
  cv::Mat hv(bbox.height * F, bbox.width * F, CV_64F);
  cv::Mat hc(hv.size(), CV_64F);
  for (int Y = 0; Y < hv.rows; ++Y) {
    auto* vr = hv.ptr<double>(Y);
    auto* cr = hc.ptr<double>(Y);
    // Low-res pixel x covers [x-0.5, x+0.5] (OpenCV convention).
    const double yc = bbox.y - 0.5 + (Y + 0.5) * step;
    for (int X = 0; X < hv.cols; ++X) {
      const double xc = bbox.x - 0.5 + (X + 0.5) * step;
      const double h = 0.5 * step;
      const Region c0 = classifyImage(xc, yc);
      if (c0 == classifyImage(xc - h, yc - h) && c0 == classifyImage(xc + h, yc - h) &&
          c0 == classifyImage(xc - h, yc + h) && c0 == classifyImage(xc + h, yc + h)) {
        vr[X] = region_value[c0];
        cr[X] = c0 == kBackground ? 0.0 : 1.0;
        continue;
      }
      double sum = 0.0;
      int hits = 0;
      for (int j = 0; j < sub; ++j) {
        for (int i = 0; i < sub; ++i) {
          const Region r =
              classifyImage(xc - h + (i + 0.5) * step / sub, yc - h + (j + 0.5) * step / sub);
          if (r != kBackground) {
            sum += region_value[r];
            ++hits;
          }
        }
      }
      vr[X] = sum / (sub * sub);  // premultiplied by coverage
      cr[X] = static_cast<double>(hits) / (sub * sub);
    }
  }
  if (p.blur_sigma > 0.0) {
    cv::GaussianBlur(hv, hv, cv::Size(), p.blur_sigma * F, 0.0, cv::BORDER_REPLICATE);
    cv::GaussianBlur(hc, hc, cv::Size(), p.blur_sigma * F, 0.0, cv::BORDER_REPLICATE);
  }
  cv::Mat lo;
  cv::resize(hv, lo, bbox.size(), 0, 0, cv::INTER_AREA);  // exact F x F box average
  lo.copyTo(premult(bbox));
  cv::resize(hc, lo, bbox.size(), 0, 0, cv::INTER_AREA);
  lo.copyTo(coverage(bbox));
}

cv::Mat smoothNoise(cv::Size size, double cell, double amplitude, cv::RNG& rng) {
  const cv::Size small(std::max(2, static_cast<int>(size.width / cell)),
                       std::max(2, static_cast<int>(size.height / cell)));
  cv::Mat n(small, CV_64F);
  rng.fill(n, cv::RNG::NORMAL, 0.0, amplitude);
  cv::Mat out;
  cv::resize(n, out, size, 0, 0, cv::INTER_CUBIC);
  return out;
}

}  // namespace

cv::Mat renderScene(const std::vector<SyntheticParams>& targets,
                    std::vector<cv::Point2d>* centers) {
  CV_Assert(!targets.empty());
  const SyntheticParams& g = targets.front();
  cv::RNG rng(g.seed);

  cv::Mat img = smoothNoise(g.image_size, 24.0, g.background_texture, rng) + g.background_mean;
  cv::Mat panel_mask = cv::Mat::zeros(g.image_size, CV_64F);

  if (centers) centers->clear();
  for (auto t : targets) {
    t.image_size = g.image_size;
    cv::Mat premult, cov;
    renderLayer(t, premult, cov);
    img = img.mul(1.0 - cov) + premult;
    panel_mask = cv::max(panel_mask, cov);
    if (centers) centers->push_back(t.center);
  }

  // Fine material texture on the panel (paint, fabric, wood grain...).
  if (g.material_texture > 0.0) {
    cv::Mat tex(g.image_size, CV_64F);
    rng.fill(tex, cv::RNG::NORMAL, 0.0, g.material_texture);
    cv::GaussianBlur(tex, tex, cv::Size(), 0.8);
    img += tex.mul(panel_mask);
  }

  // Illumination: linear gradient in a random direction + optional cast shadow.
  const int W = g.image_size.width;
  const int Hh = g.image_size.height;
  cv::Mat shade(g.image_size, CV_64F, cv::Scalar(1.0));
  if (g.illumination_gradient != 0.0) {
    const double phi = rng.uniform(0.0, 2.0 * std::numbers::pi);
    const double norm = std::max(W, Hh);
    for (int y = 0; y < Hh; ++y) {
      auto* s = shade.ptr<double>(y);
      for (int x = 0; x < W; ++x) {
        s[x] *= 1.0 + g.illumination_gradient *
                          ((x - 0.5 * W) * std::cos(phi) + (y - 0.5 * Hh) * std::sin(phi)) / norm;
      }
    }
  }
  if (g.cast_shadow) {
    // Shadow edge passes at a distance of 0.15..0.35 target sizes from the
    // centre, i.e. through the estimation window but not through the saddle.
    const double phi = rng.uniform(0.0, 2.0 * std::numbers::pi);
    const double d = rng.uniform(0.15, 0.35) * g.target_size_px;
    const cv::Point2d n(std::cos(phi), std::sin(phi));
    cv::Mat sm(g.image_size, CV_64F);
    for (int y = 0; y < Hh; ++y) {
      auto* s = sm.ptr<double>(y);
      for (int x = 0; x < W; ++x) {
        const double dist = (x - g.center.x) * n.x + (y - g.center.y) * n.y - d;
        s[x] = dist > 0.0 ? g.shadow_factor : 1.0;
      }
    }
    cv::GaussianBlur(sm, sm, cv::Size(), 1.5);  // penumbra
    shade = shade.mul(sm);
  }
  img = img.mul(shade);

  if (g.noise_sigma > 0.0) {
    cv::Mat noise(g.image_size, CV_64F);
    rng.fill(noise, cv::RNG::NORMAL, 0.0, g.noise_sigma);
    img += noise;
  }

  cv::Mat out;
  img.convertTo(out, CV_8U);  // saturating
  return out;
}

SyntheticImage renderTarget(const SyntheticParams& params) {
  SyntheticImage s;
  s.image = renderScene({params});
  s.H = buildHomography(params);
  s.center = params.center;
  return s;
}

}  // namespace gcp
