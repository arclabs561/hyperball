# Changelog

All notable changes to this crate are documented here.
Format: [Keep a Changelog](https://keepachangelog.com).

## [Unreleased]

## [0.1.7] - 2026-11-01

### Added

- `core::BallPoint`, a Poincare-ball point validated to be finite and
  strictly inside the ball (`BallPoint::new` returns `BallPointError`
  otherwise). `BallPoint::project` uses the same clamp as
  `PoincareBallCore::project`, and its `distance` and `mobius_add` give the
  same results as the slice functions.

### Changed

- `PoincareBall::new` and `PoincareBallCore::new` panic on non-finite
  curvature as well as non-positive curvature.
- `project` clamps to a radius relative to the ball size,
  `(1 / sqrt(c)) * (1 - sqrt(epsilon))`, instead of subtracting a fixed
  `1e-5`, so projected points stay inside very small and very large balls.
- The optional `innr` dependency is removed. The `simd` feature is kept as
  a no-op so dependents that enable it still build.

### Fixed

- `poincare_to_lorentz` scaled the space part by an extra `1 / sqrt(c)`, so
  for curvature other than 1 the result left the hyperboloid and the round
  trip failed. Both the slice and `ndarray` versions are fixed.
- `LorentzModel::parallel_transport` had the wrong sign on its correction
  term, so transported vectors did not keep their Minkowski norm.
- `Manifold::parallel_transport` for `PoincareBall<f64>` uses the exact
  gyration-based transport and agrees with the Lorentz model transport.
- `PoincareBall::distance` and `PoincareBallCore::distance` returned infinity
  for points near the boundary that are still inside the ball (for example
  antipodal points at the radius `project` clamps to). They now use the
  equivalent arcosh form, which stays finite.
- `Manifold::exp_map` for `PoincareBall<f64>` returned `x` unchanged for
  tangent vectors with norm below 1e-6, dropping steps whose geodesic length
  `lambda_x * ||v||` is large near the boundary. Only a zero vector is now a
  no-op.
- The README no longer claims a depth-10 binary tree needs about 1024
  Euclidean dimensions.

## [0.1.6] - 2026-04-28

### Changed

- Inlined Mobius computation in the distance hot path; eliminated `Vec<f64>`
  allocations per call. No measured benchmark; the change is mechanical
  (allocation removal, no algorithmic change).
- Bumped `innr` dependency to `0.2`.

### Fixed

- Clippy lints in the distortion example.

## [0.1.5] - 2026-04-06

Earlier releases predate this changelog.
