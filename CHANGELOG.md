# Changelog

All notable changes to this crate are documented here.
Format: [Keep a Changelog](https://keepachangelog.com).

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
