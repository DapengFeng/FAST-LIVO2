#ifndef SO3_MATH_H
#define SO3_MATH_H

#include <Eigen/Core>
#include <sophus/rotation_matrix.hpp>
#include <cmath>

#define SKEW_SYM_MATRX(v) 0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0

template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 3> NormalizeRotation(const Eigen::MatrixBase<Derived> &R_expr)
{
  return Sophus::makeRotationMatrix(R_expr);
}

template <typename T> Eigen::Matrix<T, 3, 3> Exp(const Eigen::Matrix<T, 3, 1> &ang)
{
  const T ang_norm = ang.norm();
  const Eigen::Matrix<T, 3, 3> Eye3 = Eigen::Matrix<T, 3, 3>::Identity();
  Eigen::Matrix<T, 3, 3> K;
  K << SKEW_SYM_MATRX(ang);

  if (ang_norm > T(1e-7))
  {
    const T sin_theta_by_theta = std::sin(ang_norm) / ang_norm;
    const T one_minus_cos_by_theta2 = (T(1) - std::cos(ang_norm)) / (ang_norm * ang_norm);
    return Eye3 + sin_theta_by_theta * K + one_minus_cos_by_theta2 * K * K;
  }

  return Eye3 + K + T(0.5) * K * K;
}

template <typename T, typename Ts> Eigen::Matrix<T, 3, 3> Exp(const Eigen::Matrix<T, 3, 1> &ang_vel, const Ts &dt)
{
  return Exp(Eigen::Matrix<T, 3, 1>(ang_vel * dt));
}

template <typename T> Eigen::Matrix<T, 3, 3> Exp(const T &v1, const T &v2, const T &v3)
{
  return Exp(Eigen::Matrix<T, 3, 1>(v1, v2, v3));
}

/* Logrithm of a Rotation Matrix */
template <typename T> Eigen::Matrix<T, 3, 1> Log(const Eigen::Matrix<T, 3, 3> &R)
{
  T theta = (R.trace() > 3.0 - 1e-6) ? 0.0 : std::acos(0.5 * (R.trace() - 1));
  Eigen::Matrix<T, 3, 1> K(R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1));
  return (std::abs(theta) < 0.001) ? (0.5 * K) : (0.5 * theta / std::sin(theta) * K);
}

template <typename T> Eigen::Matrix<T, 3, 1> RotMtoEuler(const Eigen::Matrix<T, 3, 3> &rot)
{
  T sy = sqrt(rot(0, 0) * rot(0, 0) + rot(1, 0) * rot(1, 0));
  bool singular = sy < 1e-6;
  T x, y, z;
  if (!singular)
  {
    x = atan2(rot(2, 1), rot(2, 2));
    y = atan2(-rot(2, 0), sy);
    z = atan2(rot(1, 0), rot(0, 0));
  }
  else
  {
    x = atan2(-rot(1, 2), rot(1, 1));
    y = atan2(-rot(2, 0), sy);
    z = 0;
  }
  Eigen::Matrix<T, 3, 1> ang(x, y, z);
  return ang;
}

#endif
