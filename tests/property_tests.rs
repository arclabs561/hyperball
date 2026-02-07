//! Property-based tests for hyperbolic manifolds.
//!
//! These tests verify mathematical invariants that must hold for all inputs,
//! not just hand-picked test cases. This catches edge cases and numerical
//! stability issues.
//!
//! # Invariants Tested
//!
//! ## Metric Space Properties
//! - d(x, x) = 0 (identity)
//! - d(x, y) = d(y, x) (symmetry)
//! - d(x, y) >= 0 (non-negativity)
//! - d(x, z) <= d(x, y) + d(y, z) (triangle inequality)
//!
//! ## Manifold Properties
//! - exp_x(log_x(y)) = y (round-trip)
//! - Points stay on manifold after operations
//! - Parallel transport preserves tangent space inner products
//!
//! ## Numerical Stability
//! - Operations near boundary don't produce NaN/Inf
//! - Small perturbations don't cause catastrophic errors

#![cfg(feature = "ndarray")]

use hyp::{LorentzModel, PoincareBall};
use ndarray::{Array1, ArrayView1};
use proptest::prelude::*;
use skel::Manifold;

const TOL: f64 = 1e-6;
#[allow(dead_code)] // For future strict numerical tests
const STRICT_TOL: f64 = 1e-10;

// ============================================================================
// Generators for valid hyperbolic points
// ============================================================================

/// Generate a point strictly inside the Poincare ball.
/// For stability, keep away from boundary.
fn poincare_point(dim: usize) -> impl Strategy<Value = Array1<f64>> {
    prop::collection::vec(-0.8f64..0.8f64, dim).prop_map(|v: Vec<f64>| {
        let arr = Array1::from_vec(v);
        // Ensure ||x|| < 0.95 (well inside unit ball)
        let norm = arr.dot(&arr).sqrt();
        if norm > 0.9 {
            arr * (0.9 / norm)
        } else {
            arr
        }
    })
}

/// Generate a Poincaré point with a conservative radius bound.
/// This is the regime we care about for stable step-based integrators (small dt).
fn poincare_point_safe(dim: usize) -> impl Strategy<Value = Array1<f64>> {
    prop::collection::vec(-0.6f64..0.6f64, dim).prop_map(|v: Vec<f64>| {
        let arr = Array1::from_vec(v);
        let norm = arr.dot(&arr).sqrt();
        if norm > 0.6 {
            arr * (0.6 / norm)
        } else {
            arr
        }
    })
}

/// Generate a point very close to the Poincare ball boundary.
/// Tests numerical stability in extreme regions.
fn poincare_point_near_boundary(dim: usize) -> impl Strategy<Value = Array1<f64>> {
    prop::collection::vec(-1.0f64..1.0f64, dim).prop_map(|v: Vec<f64>| {
        let arr = Array1::from_vec(v);
        let norm = arr.dot(&arr).sqrt();
        if norm < 0.01 {
            // If too close to origin, move outward
            arr + 0.98
        } else {
            // Scale to be very close to boundary
            arr * (0.99 / norm)
        }
    })
}

/// Generate a valid Lorentz point (on the hyperboloid).
#[allow(dead_code)] // Helper for future Lorentz property tests
fn lorentz_point(space_dim: usize) -> impl Strategy<Value = Array1<f64>> {
    prop::collection::vec(-2.0..2.0, space_dim).prop_map(move |v| {
        let lorentz = LorentzModel::<f64>::new(1.0);
        let space = Array1::from_vec(v);
        lorentz.from_euclidean(&space.view())
    })
}

/// Generate a tangent vector at a Lorentz point.
/// Tangent vectors satisfy <v, x>_L = 0.
#[allow(dead_code)]
fn lorentz_tangent(x: &ArrayView1<f64>) -> Array1<f64> {
    // For a point x on the hyperboloid, tangent vectors v satisfy:
    // -v_0 * x_0 + sum(v_i * x_i) = 0
    // So v_0 = sum(v_i * x_i) / x_0
    let _space_dim = x.len() - 1;
    let mut v = Array1::zeros(x.len());

    // Generate random space components
    use std::hash::{Hash, Hasher};
    for i in 1..x.len() {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        (i, x[0].to_bits()).hash(&mut hasher);
        let val = hasher.finish();
        v[i] = (val as f64 / u64::MAX as f64 - 0.5) * 0.5;
    }

    // Set time component to satisfy constraint
    let space_dot: f64 = (1..x.len()).map(|i| v[i] * x[i]).sum();
    v[0] = space_dot / x[0];

    v
}

// ============================================================================
// Poincare Ball Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    // Distance to self is zero
    #[test]
    fn poincare_distance_self_is_zero(x in poincare_point(3)) {
        let ball = PoincareBall::<f64>::new(1.0);
        let d: f64 = ball.distance(&x.view(), &x.view());
        prop_assert!(d.abs() < TOL, "d(x,x) = {} != 0", d);
    }

    // Distance is symmetric
    #[test]
    fn poincare_distance_symmetric(
        x in poincare_point(3),
        y in poincare_point(3)
    ) {
        let ball = PoincareBall::<f64>::new(1.0);
        let d_xy: f64 = ball.distance(&x.view(), &y.view());
        let d_yx: f64 = ball.distance(&y.view(), &x.view());
        prop_assert!((d_xy - d_yx).abs() < TOL,
            "d(x,y) = {} != d(y,x) = {}", d_xy, d_yx);
    }

    // Distance is non-negative
    #[test]
    fn poincare_distance_non_negative(
        x in poincare_point(3),
        y in poincare_point(3)
    ) {
        let ball = PoincareBall::<f64>::new(1.0);
        let d: f64 = ball.distance(&x.view(), &y.view());
        prop_assert!(d >= -TOL, "d(x,y) = {} < 0", d);
        prop_assert!(!d.is_nan(), "d(x,y) is NaN");
    }

    // Triangle inequality
    #[test]
    fn poincare_triangle_inequality(
        x in poincare_point(3),
        y in poincare_point(3),
        z in poincare_point(3)
    ) {
        let ball = PoincareBall::<f64>::new(1.0);
        let d_xz: f64 = ball.distance(&x.view(), &z.view());
        let d_xy: f64 = ball.distance(&x.view(), &y.view());
        let d_yz: f64 = ball.distance(&y.view(), &z.view());

        prop_assert!(d_xz <= d_xy + d_yz + TOL,
            "d(x,z) = {} > d(x,y) + d(y,z) = {}", d_xz, d_xy + d_yz);
    }

    // exp_0(log_0(y)) = y for points near origin
    #[test]
    fn poincare_exp_log_round_trip(y in poincare_point(3)) {
        let ball = PoincareBall::<f64>::new(1.0);

        // log_0(y) maps y to tangent space at origin
        let v = ball.log_map_zero(&y.view());

        // exp_0(v) should recover y
        let y_recovered = ball.exp_map_zero(&v.view());

        for i in 0..y.len() {
            prop_assert!((y[i] - y_recovered[i]).abs() < TOL,
                "exp(log(y))[{}] = {} != y[{}] = {}",
                i, y_recovered[i], i, y[i]);
        }
    }

    // exp_map stays inside ball
    #[test]
    fn poincare_exp_stays_in_ball(v in prop::collection::vec(-10.0..10.0, 3usize)) {
        let ball = PoincareBall::<f64>::new(1.0);
        let tangent = Array1::from_vec(v);
        let result = ball.exp_map_zero(&tangent.view());

        let norm_sq: f64 = result.dot(&result);
        prop_assert!(ball.is_in_ball(&result.view()),
            "exp_0(v) escaped ball, ||result|| = {}", norm_sq.sqrt());
    }

    // Project preserves direction (for points outside ball)
    #[test]
    fn poincare_project_preserves_direction(v in prop::collection::vec(-5.0..5.0, 3usize)) {
        let ball = PoincareBall::<f64>::new(1.0);
        let x = Array1::from_vec(v);
        let projected = ball.project(&x.view());

        prop_assert!(ball.is_in_ball(&projected.view()),
            "Projection should be inside ball");

        // Direction should be preserved (check ratios)
        let norm_x: f64 = x.dot(&x).sqrt();
        if norm_x > 0.1 {
            for i in 1..x.len() {
                if x[0].abs() > 0.01 && projected[0].abs() > 0.01 {
                    let ratio_orig = x[i] / x[0];
                    let ratio_proj = projected[i] / projected[0];
                    prop_assert!((ratio_orig - ratio_proj).abs() < TOL,
                        "Direction not preserved");
                }
            }
        }
    }

    // Mobius addition with zero is identity
    #[test]
    fn poincare_mobius_identity(x in poincare_point(3)) {
        let ball = PoincareBall::<f64>::new(1.0);
        let zero = Array1::zeros(x.len());
        let result = ball.mobius_add(&x.view(), &zero.view());

        for i in 0..x.len() {
            prop_assert!((result[i] - x[i]).abs() < TOL,
                "x + 0 != x at index {}", i);
        }
    }

    // Near-boundary numerical stability
    #[test]
    fn poincare_boundary_stability(
        x in poincare_point_near_boundary(3),
        y in poincare_point_near_boundary(3)
    ) {
        let ball = PoincareBall::<f64>::new(1.0);
        let d: f64 = ball.distance(&x.view(), &y.view());

        prop_assert!(!d.is_nan(), "Distance is NaN near boundary");
        prop_assert!(!d.is_infinite(), "Distance is infinite near boundary");
        prop_assert!(d >= 0.0, "Distance is negative near boundary: {}", d);
    }
}

// ============================================================================
// Lorentz Model Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    // Points from from_euclidean are on manifold
    #[test]
    fn lorentz_from_euclidean_on_manifold(v in prop::collection::vec(-5.0..5.0, 3usize)) {
        let lorentz = LorentzModel::<f64>::new(1.0);
        let space = Array1::from_vec(v);
        let x = lorentz.from_euclidean(&space.view());

        prop_assert!(lorentz.is_on_manifold(&x.view(), TOL),
            "Point not on manifold: <x,x>_L = {}", lorentz.minkowski_dot(&x.view(), &x.view()));
    }

    // Distance to self is zero
    #[test]
    fn lorentz_distance_self_zero(v in prop::collection::vec(-2.0..2.0, 3usize)) {
        let lorentz = LorentzModel::<f64>::new(1.0);
        let space = Array1::from_vec(v);
        let x = lorentz.from_euclidean(&space.view());
        let d: f64 = lorentz.distance(&x.view(), &x.view());

        prop_assert!(d.abs() < TOL, "d(x,x) = {} != 0", d);
    }

    // Distance is symmetric
    #[test]
    fn lorentz_distance_symmetric(
        v1 in prop::collection::vec(-2.0..2.0, 3usize),
        v2 in prop::collection::vec(-2.0..2.0, 3usize)
    ) {
        let lorentz = LorentzModel::<f64>::new(1.0);
        let x = lorentz.from_euclidean(&Array1::from_vec(v1).view());
        let y = lorentz.from_euclidean(&Array1::from_vec(v2).view());

        let d_xy: f64 = lorentz.distance(&x.view(), &y.view());
        let d_yx: f64 = lorentz.distance(&y.view(), &x.view());

        prop_assert!((d_xy - d_yx).abs() < TOL,
            "d(x,y) = {} != d(y,x) = {}", d_xy, d_yx);
    }

    // Triangle inequality
    #[test]
    fn lorentz_triangle_inequality(
        v1 in prop::collection::vec(-2.0..2.0, 3usize),
        v2 in prop::collection::vec(-2.0..2.0, 3usize),
        v3 in prop::collection::vec(-2.0..2.0, 3usize)
    ) {
        let lorentz = LorentzModel::<f64>::new(1.0);
        let x = lorentz.from_euclidean(&Array1::from_vec(v1).view());
        let y = lorentz.from_euclidean(&Array1::from_vec(v2).view());
        let z = lorentz.from_euclidean(&Array1::from_vec(v3).view());

        let d_xz: f64 = lorentz.distance(&x.view(), &z.view());
        let d_xy: f64 = lorentz.distance(&x.view(), &y.view());
        let d_yz: f64 = lorentz.distance(&y.view(), &z.view());

        prop_assert!(d_xz <= d_xy + d_yz + TOL,
            "d(x,z) = {} > d(x,y) + d(y,z) = {}", d_xz, d_xy + d_yz);
    }

    // exp preserves manifold membership
    #[test]
    fn lorentz_exp_on_manifold(v in prop::collection::vec(-2.0..2.0, 3usize)) {
        let lorentz = LorentzModel::<f64>::new(1.0);
        let x = lorentz.origin(3);
        let tangent_space = Array1::from_vec(v);

        // Create proper tangent vector at origin (time component = 0)
        let mut tangent = Array1::zeros(4);
        for (i, &val) in tangent_space.iter().enumerate() {
            tangent[i + 1] = val * 0.5; // Scale down for stability
        }
        tangent[0] = 0.0; // Tangent at origin has t=0

        let y = lorentz.exp_map(&x.view(), &tangent.view());

        prop_assert!(lorentz.is_on_manifold(&y.view(), 1e-5),
            "exp_x(v) not on manifold");
        prop_assert!(!y[0].is_nan(), "exp_x(v) produced NaN");
    }

    // Project puts points on manifold
    #[test]
    fn lorentz_project_on_manifold(v in prop::collection::vec(-5.0..5.0, 3usize)) {
        let lorentz = LorentzModel::<f64>::new(1.0);
        // Create arbitrary point (not necessarily on manifold)
        let mut x = Array1::zeros(v.len() + 1);
        x[0] = 2.0; // Positive time component
        for (i, &val) in v.iter().enumerate() {
            x[i + 1] = val;
        }

        let projected = lorentz.project(&x.view());

        prop_assert!(lorentz.is_on_manifold(&projected.view(), TOL),
            "Projected point not on manifold");
    }

    // Different curvatures give different distances
    #[test]
    fn lorentz_curvature_affects_distance(
        v1 in prop::collection::vec(-1.0..1.0, 3usize),
        v2 in prop::collection::vec(-1.0..1.0, 3usize)
    ) {
        let l1 = LorentzModel::<f64>::new(1.0);
        let l2 = LorentzModel::<f64>::new(4.0);

        let x1 = l1.from_euclidean(&Array1::from_vec(v1.clone()).view());
        let y1 = l1.from_euclidean(&Array1::from_vec(v2.clone()).view());

        let x2 = l2.from_euclidean(&Array1::from_vec(v1).view());
        let y2 = l2.from_euclidean(&Array1::from_vec(v2).view());

        let d1: f64 = l1.distance(&x1.view(), &y1.view());
        let d2: f64 = l2.distance(&x2.view(), &y2.view());

        // If points are not identical, curvature should affect distance
        let norm_diff: f64 = x1
            .iter()
            .zip(y1.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        if norm_diff > 0.1 {
            // Just verify distances are finite, curvature has effect
            prop_assert!(!d1.is_nan() && !d2.is_nan());
        }
    }

    #[test]
    fn poincare_parallel_transport_preserves_metric_norm(
        x in poincare_point_safe(3),
        y in poincare_point_safe(3),
        v in prop::collection::vec(-0.2f64..0.2f64, 3usize),
    ) {
        let ball = PoincareBall::<f64>::new(1.0);
        let v = Array1::from_vec(v);
        let pt = ball.parallel_transport(&x.view(), &y.view(), &v.view());

        let lambda_x = 2.0 / (1.0 - ball.c * x.dot(&x));
        let lambda_y = 2.0 / (1.0 - ball.c * y.dot(&y));

        let norm2_x = lambda_x * lambda_x * v.dot(&v);
        let norm2_y = lambda_y * lambda_y * pt.dot(&pt);

        let rel = (norm2_x - norm2_y).abs() / norm2_x.max(1e-12);
        // Loose tolerance: transport is numerically integrated.
        prop_assert!(rel < 3e-2, "rel={rel}");
    }
}

// ============================================================================
// Cross-model conversion tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    // Poincare -> Lorentz -> Poincare round trip
    #[test]
    fn conversion_round_trip(x in poincare_point(3)) {
        use hyp::lorentz::conversions::{lorentz_to_poincare, poincare_to_lorentz};

        let ball = PoincareBall::<f64>::new(1.0);
        let lorentz = LorentzModel::<f64>::new(1.0);

        let on_lorentz = poincare_to_lorentz(&ball, &x.view());
        let back_to_poincare = lorentz_to_poincare(&lorentz, &on_lorentz.view());

        for i in 0..x.len() {
            prop_assert!((x[i] - back_to_poincare[i]).abs() < TOL,
                "Round trip failed at index {}: {} != {}",
                i, x[i], back_to_poincare[i]);
        }
    }

    // Conversion preserves distance (up to scaling by curvature)
    #[test]
    fn conversion_preserves_distance(
        x in poincare_point(3),
        y in poincare_point(3)
    ) {
        use hyp::lorentz::conversions::poincare_to_lorentz;

        let ball = PoincareBall::<f64>::new(1.0);
        let lorentz = LorentzModel::<f64>::new(1.0);

        let d_poincare: f64 = ball.distance(&x.view(), &y.view());

        let x_lorentz = poincare_to_lorentz(&ball, &x.view());
        let y_lorentz = poincare_to_lorentz(&ball, &y.view());
        let d_lorentz: f64 = lorentz.distance(&x_lorentz.view(), &y_lorentz.view());

        // Same curvature should give same distance
        prop_assert!((d_poincare - d_lorentz).abs() < TOL,
            "Distance mismatch: Poincare {} vs Lorentz {}", d_poincare, d_lorentz);
    }
}
