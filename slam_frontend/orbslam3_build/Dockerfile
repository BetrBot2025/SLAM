# WP2: reproducible ORB-SLAM3 build environment.
#
# Pinned to ubuntu:22.04 rather than the ORB-SLAM3-era 20.04, because
# ORB_SLAM3/CMakeLists.txt requires `find_package(OpenCV 4.4)` and 20.04's
# apt-provided OpenCV is 4.2 (would need a from-source OpenCV build); 22.04's
# apt OpenCV is 4.5.4, which satisfies the check directly. GCC 11 (22.04) is
# still old enough to avoid the STL/API drift a bleeding-edge host compiler
# would hit against this ~2021-era codebase.
#
# Build: docker build -t orbslam3:22.04 -f Dockerfile .
# (run from inside WSL2 Ubuntu -- see slam_frontend/orbslam3_build/README.md)

FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git wget ca-certificates pkg-config \
    libeigen3-dev libopencv-dev \
    libgl1-mesa-dev libglew-dev \
    libboost-all-dev libssl-dev \
    libjpeg-dev libpng-dev libtiff-dev \
    && rm -rf /var/lib/apt/lists/*

# Pangolin, pinned to v0.6 -- the last tag known to match ORB-SLAM3's viewer
# API (later Pangolin releases changed headers ORB-SLAM3's System/Viewer
# code depends on). Python bindings disabled: not needed here, and one less
# thing that can fail the build.
RUN git clone --branch v0.6 --depth 1 https://github.com/stevenlovegrove/Pangolin.git /opt/Pangolin && \
    cd /opt/Pangolin && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_PANGOLIN_PYTHON=OFF && \
    make -j"$(nproc)" && make install && ldconfig

RUN git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git /opt/ORB_SLAM3
WORKDIR /opt/ORB_SLAM3

# Each Thirdparty lib as its own layer (mirrors build.sh's stages) so a fix
# to a later stage doesn't force re-running everything before it.
RUN cd Thirdparty/DBoW2 && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && make -j"$(nproc)"

RUN cd Thirdparty/g2o && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && make -j"$(nproc)"

RUN cd Thirdparty/Sophus && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && make -j"$(nproc)"

RUN cd Vocabulary && tar -xf ORBvoc.txt.tar.gz

# -- WP2 Python binding: kept as its own apt layer (rather than folded into
# the deps block above) so the already-built Pangolin/DBoW2/g2o/Sophus layers
# above stay cached instead of rebuilding from scratch for an unrelated change.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev pybind11-dev && rm -rf /var/lib/apt/lists/*

# Source patch: adds System::WasLastFrameAKeyFrame(), which vanilla
# ORB-SLAM3 has no equivalent of -- see patch_orbslam3.py's docstring.
COPY patch_orbslam3.py /opt/patch_orbslam3.py
RUN python3 /opt/patch_orbslam3.py /opt/ORB_SLAM3

# pybind11 module, added as an extra target in ORB_SLAM3's own CMakeLists.txt
# so it inherits the already-correct include/link flags for OpenCV/Eigen/
# Pangolin/DBoW2/g2o rather than re-deriving them in a standalone CMake file.
COPY orbslam3_python.cc /opt/ORB_SLAM3/bindings/orbslam3_python.cc
RUN echo '' >> CMakeLists.txt && \
    echo 'find_package(pybind11 REQUIRED)' >> CMakeLists.txt && \
    echo 'pybind11_add_module(orbslam3_python bindings/orbslam3_python.cc)' >> CMakeLists.txt && \
    echo 'target_link_libraries(orbslam3_python PRIVATE ${PROJECT_NAME})' >> CMakeLists.txt && \
    echo 'set_target_properties(orbslam3_python PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)' >> CMakeLists.txt

# -j2, not -j$(nproc): Tracking.cc/Optimizer.cc/G2oTypes.cc each pull in heavy
# Eigen+g2o template instantiation and can use several GB apiece; building
# many in parallel OOM-kills cc1plus under WSL2's default memory budget.
RUN mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && make -j2

CMD ["/bin/bash"]
