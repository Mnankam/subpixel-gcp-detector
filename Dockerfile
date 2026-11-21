# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Stage 1: build + test
# ---------------------------------------------------------------------------
FROM ubuntu:24.04 AS build

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build \
      libopencv-dev libeigen3-dev libgtest-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY CMakeLists.txt ./
COPY include ./include
COPY src ./src
COPY apps ./apps
COPY tests ./tests

RUN cmake -S . -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DGCP_WARNINGS_AS_ERRORS=ON \
 && cmake --build build \
 && ctest --test-dir build --output-on-failure \
 && cmake --install build --prefix /opt/gcp

# ---------------------------------------------------------------------------
# Stage 2: minimal runtime image (only the OpenCV runtime libraries)
# ---------------------------------------------------------------------------
FROM ubuntu:24.04 AS runtime

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libopencv-core406t64 libopencv-imgproc406t64 libopencv-imgcodecs406t64 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/gcp/bin/ /usr/local/bin/

RUN useradd --create-home --uid 10001 gcp
USER gcp
WORKDIR /data

LABEL org.opencontainers.image.title="subpixel-gcp-detector" \
      org.opencontainers.image.description="Robust sub-pixel detection of checkerboard ground control points" \
      org.opencontainers.image.licenses="MIT"

ENTRYPOINT ["gcp_detect"]
CMD ["--help"]
