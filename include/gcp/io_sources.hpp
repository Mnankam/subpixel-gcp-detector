#pragma once

#include "gcp/frame_source.hpp"

#include <memory>
#include <string>
#include <vector>

namespace gcp {

/// Plays back a list of image files (e.g. a recorded UAV flight).
[[nodiscard]] std::unique_ptr<FrameSource> makeImageFileSource(std::vector<std::string> files,
                                                               bool loop = false);

/// Settings for a Basler camera opened through the pylon SDK (GenICam).
struct PylonSourceParams {
  /// Use the pylon camera emulator (needs the environment variable
  /// PYLON_CAMEMU=1). No hardware required.
  bool emulator = false;
  /// Open the camera with this serial number; empty = first camera found.
  std::string serial;
  /// Exposure time in microseconds; <= 0 keeps the camera setting.
  double exposure_us = -1.0;
  /// Gain in dB; < 0 keeps the camera setting.
  double gain_db = -1.0;
  /// Emulator only: file or folder with images the emulator should "capture"
  /// (GenICam features ImageFileMode / ImageFilename).
  std::string emulator_images;
  /// Stop after this many frames; 0 = endless.
  int max_frames = 0;
  unsigned timeout_ms = 5000;
};

/// True if the library was built with Basler pylon support (GCP_WITH_PYLON).
[[nodiscard]] bool pylonAvailable() noexcept;

/// Opens a Basler camera (or the emulator). Throws std::runtime_error if no
/// camera is found or the library was built without pylon support.
[[nodiscard]] std::unique_ptr<FrameSource> makePylonSource(const PylonSourceParams& params);

}  // namespace gcp
