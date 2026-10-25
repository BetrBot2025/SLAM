// WP2: pybind11 binding exposing ORB-SLAM3's System to Python, matching the
// interface slam_frontend/tracking_interface.py's ORBSlam3Tracker expects.
//
// Only monocular is wired up (fisheye vs. pinhole is selected by the YAML
// settings file's Camera.type field, not a separate C++ call -- vanilla
// ORB-SLAM3 has no distinct "TrackFisheye" method; KannalaBrandt8 is just
// another GeometricCamera implementation TrackMonocular dispatches to
// internally). is_keyframe() relies on the WasLastFrameAKeyFrame() patch in
// patch_orbslam3.py -- vanilla ORB-SLAM3 has no such public accessor.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include <System.h>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace {

cv::Mat numpy_to_mat(py::array arr) {
    py::buffer_info info = arr.request();
    if (info.format != py::format_descriptor<uint8_t>::format())
        throw std::runtime_error("expected a uint8 array");
    if (info.ndim == 2)
        return cv::Mat(static_cast<int>(info.shape[0]), static_cast<int>(info.shape[1]),
                        CV_8UC1, info.ptr).clone();
    if (info.ndim == 3 && info.shape[2] == 3)
        return cv::Mat(static_cast<int>(info.shape[0]), static_cast<int>(info.shape[1]),
                        CV_8UC3, info.ptr).clone();
    throw std::runtime_error("expected a (H,W) grayscale or (H,W,3) RGB uint8 array");
}

py::array_t<float> mat4_to_numpy(const Eigen::Matrix4f& M) {
    py::array_t<float> result({4, 4});
    auto buf = result.mutable_unchecked<2>();
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            buf(i, j) = M(i, j);
    return result;
}

}  // namespace

class PyORBSLAM3 {
public:
    PyORBSLAM3(const std::string& vocab_path, const std::string& settings_path, bool use_viewer)
        : slam_(vocab_path, settings_path, ORB_SLAM3::System::MONOCULAR, use_viewer) {}

    py::array_t<float> track_monocular(py::array image, double timestamp) {
        cv::Mat im = numpy_to_mat(image);
        Sophus::SE3f Tcw = slam_.TrackMonocular(im, timestamp);
        return mat4_to_numpy(Tcw.matrix());
    }

    py::array_t<float> get_map_points() {
        std::vector<ORB_SLAM3::MapPoint*> points = slam_.GetTrackedMapPoints();
        std::vector<Eigen::Vector3f> valid;
        valid.reserve(points.size());
        for (auto* p : points) {
            if (p == nullptr || p->isBad())
                continue;
            valid.push_back(p->GetWorldPos());
        }
        py::array_t<float> result({static_cast<long>(valid.size()), 3L});
        auto buf = result.mutable_unchecked<2>();
        for (size_t i = 0; i < valid.size(); i++)
            for (int j = 0; j < 3; j++)
                buf(i, j) = valid[i](j);
        return result;
    }

    bool is_keyframe() { return slam_.WasLastFrameAKeyFrame(); }
    int tracking_state() { return slam_.GetTrackingState(); }
    void shutdown() { slam_.Shutdown(); }

private:
    ORB_SLAM3::System slam_;
};

PYBIND11_MODULE(orbslam3_python, m) {
    m.doc() = "pybind11 binding for ORB-SLAM3 (WP2)";

    py::enum_<ORB_SLAM3::Tracking::eTrackingState>(m, "TrackingState")
        .value("SYSTEM_NOT_READY", ORB_SLAM3::Tracking::SYSTEM_NOT_READY)
        .value("NO_IMAGES_YET", ORB_SLAM3::Tracking::NO_IMAGES_YET)
        .value("NOT_INITIALIZED", ORB_SLAM3::Tracking::NOT_INITIALIZED)
        .value("OK", ORB_SLAM3::Tracking::OK)
        .value("RECENTLY_LOST", ORB_SLAM3::Tracking::RECENTLY_LOST)
        .value("LOST", ORB_SLAM3::Tracking::LOST)
        .value("OK_KLT", ORB_SLAM3::Tracking::OK_KLT);

    py::class_<PyORBSLAM3>(m, "System")
        .def(py::init<const std::string&, const std::string&, bool>(),
             py::arg("vocab_path"), py::arg("settings_path"), py::arg("use_viewer") = false)
        .def("track_monocular", &PyORBSLAM3::track_monocular, py::arg("image"), py::arg("timestamp"),
             "image: (H,W) grayscale or (H,W,3) RGB uint8 array. Returns the (4,4) camera pose Tcw.")
        .def("get_map_points", &PyORBSLAM3::get_map_points,
             "Returns (N,3) float32 world positions of currently-tracked map points.")
        .def("is_keyframe", &PyORBSLAM3::is_keyframe,
             "True if the frame just processed by track_monocular() was inserted as a keyframe.")
        .def("tracking_state", &PyORBSLAM3::tracking_state)
        .def("shutdown", &PyORBSLAM3::shutdown);
}
