// File: bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h> // Magic header: Auto-converts NumPy arrays to Eigen Matrices!
#include "rsw_model.hpp"

namespace py = pybind11;

// "rsw_engine" will be the name of the Python library we import
PYBIND11_MODULE(rsw_engine, m) {
    m.doc() = "High-Performance RSW C++ Engine for Python";

    // Expose the HighPerformanceRSW class to Python
    py::class_<HighPerformanceRSW>(m, "HighPerformanceRSW")
        // Expose the constructor
        .def(py::init<const std::string&, int, double, double, double, double, int, int, int>())
        // Expose the fit function
        .def("fit", &HighPerformanceRSW::fit)
        // Expose the predict function (NEW)
        .def("predict", &HighPerformanceRSW::predict)
        // Expose the learned weights so Python can read them after training
        .def_readonly("theta", &HighPerformanceRSW::theta)
        .def_readonly("mask", &HighPerformanceRSW::mask);
}