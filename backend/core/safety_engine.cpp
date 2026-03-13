#include <pybind11/pybind11.h>
#include <string>

namespace py = pybind11;

std::string evaluate_safety(float battery, float obstacle_dist, int gps_signal) {
    if (battery < 20.0f || gps_signal < 3) {
        return "RETURN_HOME";
    }
    if (obstacle_dist < 5.0f) {
        return "HOLD";
    }
    return "SAFE";
}

PYBIND11_MODULE(drone_safety, m) {
    m.doc() = "High-performance C++ safety engine for Drone OS";
    m.def("evaluate_safety", &evaluate_safety, "Evaluate drone safety status");
}