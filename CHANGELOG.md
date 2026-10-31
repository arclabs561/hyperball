# Changelog

All notable changes to this crate are documented here.
Format: [Keep a Changelog](https://keepachangelog.com).

## [Unreleased]

### Fixed

- `PoincareBall::distance` and `PoincareBallCore::distance` returned infinity
  for points near the boundary that are still inside the ball (for example
  antipodal points at the radius `project` clamps to). They now use the
  equivalent arcosh form, which stays finite.
- `Manifold::exp_map` for `PoincareBall<f64>` returned `x` unchanged for
  tangent vectors with norm below 1e-6, dropping steps whose geodesic length
  `lambda_x * ||v||` is large near the boundary. Only a zero vector is now a
  no-op.

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
