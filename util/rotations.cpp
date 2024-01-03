#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <cmath>

using namespace std;

Eigen::Vector3f logarithm(const Eigen::Matrix4f& R) {
    assert(R.cols() == R.rows());
    assert(R.cols() == 4);

    float diag = (R(0,0) + R(1,1) + R(2,2)  - 1) / 2;
    if (diag > 1)
        diag = 1;
    else if (diag < -1)
        diag = -1;

    float theta = acos(diag);
    float sine = sin(theta);
    if (sine == 0) {
        Eigen::Vector3f zeros(0,0,0);
        return zeros;
    }

    float v1 = (R(2, 1) - R(1, 2)) / (2*sine);
    float v2 = (R(0, 2) - R(2, 0)) / (2*sine);
    float v3 = (R(1, 0) - R(0, 1)) / (2*sine);

    Eigen::Vector3f result(v1*theta, v2*theta, v3*theta);
    return result;
}

Eigen::Matrix4f exponential(const Eigen::Vector3f& v) {
    float theta = sqrt(v(0)*v(0) + v(1)*v(1) + v(2)*v(2));
    float cosine = cos(theta);
    float sine = sin(theta);

    if (theta < 0.0001) {
        return Eigen::Matrix4f::Identity();
    }

    float x = v(0) / theta;
    float y = v(1) / theta;
    float z = v(2) / theta;

    Eigen::Matrix4f m;
    m <<cosine + x*x*(1-cosine), x*y*(1-cosine) - z*sine, x*z*(1-cosine) + y*sine, 0,
        y*x*(1-cosine) + z*sine, cosine + y*y*(1-cosine), y*z*(1-cosine) - x*sine, 0,
        z*x*(1-cosine) - y*sine, z*y*(1-cosine) + x*sine, cosine + z*z*(1-cosine), 0,
        0,0,0,1;

    return m;
}

namespace py=pybind11;
PYBIND11_MODULE(rotations, m) {
    m.doc() = "Rotation formulas in cpp";
    m.def("log", &logarithm, "Logarithm");
    m.def("exp", &exponential, "Exponential");
}
