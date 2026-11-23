#include "gcp/io_sources.hpp"

#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <stdexcept>

namespace gcp {

namespace {

class ImageFileSource final : public FrameSource {
 public:
  ImageFileSource(std::vector<std::string> files, bool loop)
      : files_(std::move(files)), loop_(loop) {
    if (files_.empty()) throw std::runtime_error("image file source: no files given");
  }

  bool grab(Frame& frame) override {
    while (true) {
      if (pos_ >= files_.size()) {
        if (!loop_) return false;
        pos_ = 0;
      }
      const std::string& file = files_[pos_++];
      frame.image = cv::imread(file, cv::IMREAD_GRAYSCALE);
      if (frame.image.empty()) {
        if (++failures_ > files_.size()) throw std::runtime_error("no readable image files");
        continue;  // skip unreadable files
      }
      frame.index = index_++;
      frame.timestamp =
          static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::steady_clock::now().time_since_epoch())
                                         .count());
      frame.ground_truth.clear();
      return true;
    }
  }

  [[nodiscard]] std::string describe() const override {
    return std::to_string(files_.size()) + " image file(s)" + (loop_ ? ", looping" : "");
  }

 private:
  std::vector<std::string> files_;
  bool loop_;
  size_t pos_ = 0;
  size_t failures_ = 0;
  std::uint64_t index_ = 0;
};

}  // namespace

std::unique_ptr<FrameSource> makeImageFileSource(std::vector<std::string> files, bool loop) {
  return std::make_unique<ImageFileSource>(std::move(files), loop);
}

}  // namespace gcp
