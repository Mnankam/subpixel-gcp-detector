// Basler pylon camera source (GenICam). Compiled with real camera support only
// when CMake finds the pylon SDK and GCP_WITH_PYLON=ON; otherwise a stub that
// reports the missing support is built, so applications need no #ifdefs.

#include "gcp/io_sources.hpp"

#include <stdexcept>

#ifdef GCP_HAVE_PYLON

#include <pylon/PylonIncludes.h>

#include <iostream>

namespace gcp {

namespace {

/// Sets a float feature if it exists and is writable; tries several names
/// because older cameras / the emulator use SFNC 1.x names (e.g.
/// ExposureTimeAbs) while current cameras use SFNC 2.x (ExposureTime).
bool trySetFloat(GenApi::INodeMap& nodemap, std::initializer_list<const char*> names,
                 double value) {
  for (const char* name : names) {
    if (Pylon::CFloatParameter(nodemap, name).TrySetValue(value)) return true;
  }
  return false;
}

class PylonSource final : public FrameSource {
 public:
  explicit PylonSource(const PylonSourceParams& params) : params_(params) {
    try {
      Pylon::CDeviceInfo filter;
      if (params_.emulator) filter.SetDeviceClass("BaslerCamEmu");
      if (!params_.serial.empty()) filter.SetSerialNumber(params_.serial.c_str());

      camera_.Attach(Pylon::CTlFactory::GetInstance().CreateFirstDevice(filter));
      camera_.Open();
      configure();

      // Mono8 is what the detector needs; anything else (Bayer, Mono12, ...)
      // is converted by the pylon image format converter.
      converter_.OutputPixelFormat = Pylon::PixelType_Mono8;

      // Live processing: always hand out the newest frame and drop stale
      // ones instead of building up latency in a queue.
      camera_.StartGrabbing(Pylon::GrabStrategy_LatestImageOnly);
    } catch (const Pylon::GenericException& e) {
      throw std::runtime_error(std::string("pylon: cannot open camera: ") + e.GetDescription());
    }
  }

  ~PylonSource() override {
    try {
      if (camera_.IsGrabbing()) camera_.StopGrabbing();
      if (camera_.IsOpen()) camera_.Close();
    } catch (const Pylon::GenericException& e) {
      std::cerr << "pylon: error while closing camera: " << e.GetDescription() << '\n';
    }
  }

  bool grab(Frame& frame) override {
    if (params_.max_frames > 0 && index_ >= static_cast<std::uint64_t>(params_.max_frames)) {
      return false;
    }
    try {
      for (int attempt = 0; attempt < 10; ++attempt) {
        if (!camera_.IsGrabbing()) return false;
        camera_.RetrieveResult(params_.timeout_ms, result_, Pylon::TimeoutHandling_ThrowException);
        if (!result_->GrabSucceeded()) {
          // Incomplete buffers happen (e.g. GigE packet loss): skip, don't abort.
          std::cerr << "pylon: grab failed (0x" << std::hex << result_->GetErrorCode() << std::dec
                    << "): " << result_->GetErrorDescription() << '\n';
          continue;
        }
        toMat(frame.image);
        frame.index = index_++;
        frame.timestamp = result_->GetTimeStamp();
        frame.ground_truth.clear();
        return true;
      }
      throw std::runtime_error("pylon: too many consecutive failed grabs");
    } catch (const Pylon::GenericException& e) {
      throw std::runtime_error(std::string("pylon: ") + e.GetDescription());
    }
  }

  [[nodiscard]] std::string describe() const override {
    const Pylon::CDeviceInfo& info = camera_.GetDeviceInfo();
    return std::string(info.GetVendorName().c_str()) + " " + info.GetModelName().c_str() + " (SN " +
           info.GetSerialNumber().c_str() + ")";
  }

 private:
  void configure() {
    GenApi::INodeMap& nodemap = camera_.GetNodeMap();

    // Prefer native 8-bit mono to avoid conversions; colour cameras keep
    // their format and are converted in toMat().
    Pylon::CEnumParameter(nodemap, "PixelFormat").TrySetValue("Mono8");

    if (params_.exposure_us > 0.0) {
      Pylon::CEnumParameter(nodemap, "ExposureAuto").TrySetValue("Off");
      if (!trySetFloat(nodemap, {"ExposureTime", "ExposureTimeAbs"}, params_.exposure_us)) {
        std::cerr << "pylon: exposure time not settable on this device\n";
      }
    }
    if (params_.gain_db >= 0.0) {
      Pylon::CEnumParameter(nodemap, "GainAuto").TrySetValue("Off");
      if (!trySetFloat(nodemap, {"Gain"}, params_.gain_db)) {
        std::cerr << "pylon: gain (dB) not settable on this device\n";
      }
    }
    if (params_.emulator && !params_.emulator_images.empty()) {
      // Camera emulator: replay own images instead of the built-in test pattern.
      Pylon::CEnumParameter(nodemap, "ImageFileMode").TrySetValue("On");
      if (!Pylon::CStringParameter(nodemap, "ImageFilename")
               .TrySetValue(params_.emulator_images.c_str())) {
        std::cerr << "pylon: emulator does not accept ImageFilename\n";
      }
    }
  }

  void toMat(cv::Mat& out) {
    const auto w = static_cast<int>(result_->GetWidth());
    const auto h = static_cast<int>(result_->GetHeight());
    if (converter_.ImageHasDestinationFormat(result_)) {
      // Zero-copy view on the pylon buffer (respecting row padding), then one
      // copy so the frame stays valid after the buffer is re-queued.
      const size_t stride = static_cast<size_t>(w) + result_->GetPaddingX();
      cv::Mat(h, w, CV_8UC1, result_->GetBuffer(), stride).copyTo(out);
    } else {
      converter_.Convert(converted_, result_);
      cv::Mat(h, w, CV_8UC1, converted_.GetBuffer()).copyTo(out);
    }
  }

  PylonSourceParams params_;
  // Declaration order matters: the pylon runtime must be initialised before
  // and terminated after every other pylon object (members are destroyed in
  // reverse order).
  Pylon::PylonAutoInitTerm runtime_;
  Pylon::CInstantCamera camera_;
  Pylon::CGrabResultPtr result_;
  Pylon::CImageFormatConverter converter_;
  Pylon::CPylonImage converted_;
  std::uint64_t index_ = 0;
};

}  // namespace

bool pylonAvailable() noexcept { return true; }

std::unique_ptr<FrameSource> makePylonSource(const PylonSourceParams& params) {
  return std::make_unique<PylonSource>(params);
}

}  // namespace gcp

#else  // ------------------------------------------------------------------ stub

namespace gcp {

bool pylonAvailable() noexcept { return false; }

std::unique_ptr<FrameSource> makePylonSource(const PylonSourceParams& /*params*/) {
  throw std::runtime_error(
      "built without Basler pylon support: install the pylon SDK and configure with "
      "-DGCP_WITH_PYLON=ON");
}

}  // namespace gcp

#endif
