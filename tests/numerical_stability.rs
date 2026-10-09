//! Numerical stability tests for hyperbolic operations.
//!
//! These tests verify that operations remain stable at:
//! - The origin (curvature singularity point)
//! - Near the boundary (where distances explode)
//! - With extreme curvatures
//! - With very small/large inputs

#![cfg(feature = "ndarray")]

use hyperball::{LorentzModel, PoincareBall};
use ndarray::Array1;

const TOL: f64 = 1e-6;

// =============================================================================
// Origin behavior
// =============================================================================

#[test]
fn poincare_origin_is_fixed_point() {
    let ball = PoincareBall::<f64>::new(1.0);
    let origin = Array1::zeros(3);

    // Distance from origin to itself is 0
    let d: f64 = ball.distance(&origin.view(), &origin.view());
    assert!(
        d.abs() < TOL,
        "Origin distance to itself should be 0, got {}",
        d
    );

    // Mobius add with zero is identity
    let v = Array1::from_vec(vec![0.1, 0.2, 0.3]);
    let result = ball.mobius_add(&origin.view(), &v.view());
    for i in 0..3 {
        assert!(
            (result[i] - v[i]).abs() < TOL,
            "0 + v should equal v at index {}",
            i
        );
    }
}

#[test]
fn poincare_exp_log_at_origin() {
    let ball = PoincareBall::<f64>::new(1.0);

    // Test various tangent vectors at origin
    let tangents = [
        vec![0.1, 0.0, 0.0],
        vec![0.0, 0.1, 0.0],
        vec![0.1, 0.1, 0.1],
        vec![0.5, 0.5, 0.5],
    ];

    for t in &tangents {
        let tangent = Array1::from_vec(t.clone());

        // exp then log should recover the tangent
        let point = ball.exp_map_zero(&tangent.view());
        let recovered = ball.log_map_zero(&point.view());

        for i in 0..3 {
            assert!(
                (tangent[i] - recovered[i]).abs() < TOL,
                "log(exp(v)) != v at index {}: {} vs {}",
                i,
                tangent[i],
                recovered[i]
            );
        }
    }
}

#[test]
fn lorentz_origin_properties() {
    let lorentz = LorentzModel::<f64>::new(1.0);
    let origin = lorentz.origin(3);

    // Origin should be on manifold
    assert!(
        lorentz.is_on_manifold(&origin.view(), TOL),
        "Origin should be on manifold"
    );

    // Origin's time component should be 1/sqrt(K)
    let expected_t = 1.0; // For K=1
    assert!(
        (origin[0] - expected_t).abs() < TOL,
        "Origin time component should be {}, got {}",
        expected_t,
        origin[0]
    );

    // Space components should be 0
    for i in 1..origin.len() {
        assert!(
            origin[i].abs() < TOL,
            "Origin space component {} should be 0, got {}",
            i,
            origin[i]
        );
    }
}

// =============================================================================
// Boundary behavior (Poincare ball)
// =============================================================================

#[test]
fn poincare_near_boundary_finite_distance() {
    let ball = PoincareBall::new(1.0);

    // Create points very close to boundary
    let near_boundary = |scale: f64| -> Array1<f64> {
        let mut v = Array1::zeros(3);
        v[0] = scale;
        v
    };

    let scales = [0.9, 0.95, 0.99, 0.999];

    for &s1 in &scales {
        for &s2 in &scales {
            let p1 = near_boundary(s1);
            let p2 = near_boundary(s2);

            let d = ball.distance(&p1.view(), &p2.view());

            assert!(
                !d.is_nan(),
                "Distance is NaN for points at scales {}, {}",
                s1,
                s2
            );
            assert!(
                !d.is_infinite(),
                "Distance is infinite for points at scales {}, {}",
                s1,
                s2
            );
            assert!(d >= 0.0, "Distance is negative: {}", d);
        }
    }
}

#[test]
fn poincare_project_brings_inside() {
    let ball = PoincareBall::new(1.0);

    // Points outside the ball
    let outside_points = [
        vec![2.0, 0.0, 0.0],
        vec![0.0, 1.5, 0.0],
        vec![1.0, 1.0, 1.0],
        vec![10.0, 10.0, 10.0],
    ];

    for p in &outside_points {
        let point = Array1::from_vec(p.clone());
        let projected = ball.project(&point.view());

        assert!(
            ball.is_in_ball(&projected.view()),
            "Projected point should be inside ball: {:?}",
            projected
        );

        let norm_sq: f64 = projected.iter().map(|x| x * x).sum();
        assert!(
            norm_sq < 1.0,
            "Projected point norm^2 {} should be < 1",
            norm_sq
        );
    }
}

// =============================================================================
// Curvature effects
// =============================================================================

#[test]
fn different_curvatures_affect_distances() {
    // Different curvatures should produce different (but valid) distances
    let p1 = Array1::from_vec(vec![0.1, 0.0, 0.0]);
    let p2 = Array1::from_vec(vec![0.3, 0.0, 0.0]);

    let ball_k1 = PoincareBall::<f64>::new(1.0);
    let ball_k4 = PoincareBall::<f64>::new(4.0);

    let d1: f64 = ball_k1.distance(&p1.view(), &p2.view());
    let d4: f64 = ball_k4.distance(&p1.view(), &p2.view());

    // Both distances should be finite and non-negative
    assert!(!d1.is_nan() && !d1.is_infinite() && d1 >= 0.0);
    assert!(!d4.is_nan() && !d4.is_infinite() && d4 >= 0.0);

    // Different curvatures should give different distances
    // (unless points are the same, which they're not)
    assert!(
        (d1 - d4).abs() > 1e-10,
        "Different curvatures should give different distances: {} vs {}",
        d1,
        d4
    );
}

#[test]
fn lorentz_curvature_effects() {
    let l1 = LorentzModel::<f64>::new(1.0);
    let l4 = LorentzModel::<f64>::new(4.0);

    // Create corresponding points
    let space = Array1::from_vec(vec![0.2, 0.1, 0.0]);
    let x1 = l1.from_euclidean(&space.view());
    let x4 = l4.from_euclidean(&space.view());

    // Both should be on their respective manifolds
    assert!(l1.is_on_manifold(&x1.view(), TOL));
    assert!(l4.is_on_manifold(&x4.view(), TOL));
}

// =============================================================================
// Extreme values
// =============================================================================

#[test]
fn poincare_handles_zero_vector() {
    let ball = PoincareBall::<f64>::new(1.0);
    let zero = Array1::zeros(3);
    let nonzero = Array1::from_vec(vec![0.5, 0.0, 0.0]);

    // Distance from zero to nonzero should be finite
    let d: f64 = ball.distance(&zero.view(), &nonzero.view());
    assert!(!d.is_nan(), "Distance from origin should not be NaN");
    assert!(
        !d.is_infinite(),
        "Distance from origin should not be infinite"
    );
}

#[test]
fn poincare_handles_very_small_vectors() {
    let ball = PoincareBall::<f64>::new(1.0);

    let tiny = Array1::from_vec(vec![1e-10, 0.0, 0.0]);
    let also_tiny = Array1::from_vec(vec![0.0, 1e-10, 0.0]);

    let d: f64 = ball.distance(&tiny.view(), &also_tiny.view());

    assert!(
        !d.is_nan(),
        "Distance between tiny vectors should not be NaN"
    );
    assert!(d >= 0.0, "Distance should be non-negative");
}

#[test]
fn lorentz_handles_large_space_components() {
    let lorentz = LorentzModel::<f64>::new(1.0);

    // Large Euclidean coordinates
    let large = Array1::from_vec(vec![100.0, 50.0, 25.0]);
    let x = lorentz.from_euclidean(&large.view());

    assert!(!x[0].is_nan(), "Time component should not be NaN");
    assert!(!x[0].is_infinite(), "Time component should not be infinite");
    assert!(x[0] > 0.0, "Time component should be positive");

    assert!(
        lorentz.is_on_manifold(&x.view(), 1e-4),
        "Point from large Euclidean should be on manifold"
    );
}

// =============================================================================
// Inverse operations
// =============================================================================

#[test]
fn poincare_mobius_with_self() {
    let ball = PoincareBall::<f64>::new(1.0);

    let x = Array1::from_vec(vec![0.3, 0.2, 0.1]);

    // x + x should be 2x in tangent space direction (approximately for small x)
    let result = ball.mobius_add(&x.view(), &x.view());

    // Result should still be in the ball
    assert!(ball.is_in_ball(&result.view()), "x + x should stay in ball");

    // For small x, x + x should point in same direction
    let dot_original: f64 = x.iter().sum();
    let dot_result: f64 = result.iter().sum();

    // Both should have same sign (same quadrant)
    assert!(
        dot_original.signum() == dot_result.signum() || dot_original.abs() < TOL,
        "x + x should preserve direction"
    );
}

#[test]
fn lorentz_from_euclidean_invertible() {
    let lorentz = LorentzModel::new(1.0);

    let space_x = Array1::from_vec(vec![0.2, 0.1, 0.0]);
    let space_y = Array1::from_vec(vec![-0.1, 0.3, 0.1]);

    let x = lorentz.from_euclidean(&space_x.view());
    let y = lorentz.from_euclidean(&space_y.view());

    // Both should be on manifold
    assert!(lorentz.is_on_manifold(&x.view(), TOL));
    assert!(lorentz.is_on_manifold(&y.view(), TOL));

    // Distance should be symmetric
    let d_xy = lorentz.distance(&x.view(), &y.view());
    let d_yx = lorentz.distance(&y.view(), &x.view());

    assert!(
        (d_xy - d_yx).abs() < TOL,
        "Distance should be symmetric: {} vs {}",
        d_xy,
        d_yx
    );
}

// =============================================================================
// Higher dimensions
// =============================================================================

#[test]
fn poincare_high_dimensional() {
    let ball = PoincareBall::new(1.0);
    let dim = 100;

    // Create two random-ish points
    let mut x = Array1::zeros(dim);
    let mut y = Array1::zeros(dim);
    for i in 0..dim {
        x[i] = 0.01 * (i as f64).sin();
        y[i] = 0.01 * (i as f64).cos();
    }

    let d = ball.distance(&x.view(), &y.view());

    assert!(!d.is_nan(), "High-dim distance should not be NaN");
    assert!(!d.is_infinite(), "High-dim distance should not be infinite");
    assert!(d >= 0.0, "Distance should be non-negative");
}

#[test]
fn lorentz_high_dimensional() {
    let lorentz = LorentzModel::new(1.0);
    let space_dim = 100;

    let mut space = Array1::zeros(space_dim);
    for i in 0..space_dim {
        space[i] = 0.1 * (i as f64).sin();
    }

    let x = lorentz.from_euclidean(&space.view());

    assert!(
        lorentz.is_on_manifold(&x.view(), 1e-5),
        "High-dim point should be on manifold"
    );

    // Distance to origin should be finite
    let origin = lorentz.origin(space_dim);
    let d = lorentz.distance(&x.view(), &origin.view());

    assert!(!d.is_nan(), "High-dim distance should not be NaN");
    assert!(!d.is_infinite(), "High-dim distance should not be infinite");
}

// =============================================================================
// f32 near the boundary
// =============================================================================

/// A point at relative radius 1 - 1e-6 is representable in f32 (machine
/// epsilon about 6e-8), so log_0 then exp_0 must return it, and the
/// distance from the origin must match 2/sqrt(c) * artanh(sqrt(c) |x|).
#[test]
fn poincare_f32_round_trip_at_radius_one_minus_1e6() {
    for &c in &[1.0f32, 2.0, 0.5] {
        let ball = PoincareBall::<f32>::new(c);
        let r = (1.0 - 1e-6) / c.sqrt();
        let x = Array1::from_vec(vec![0.6 * r, 0.8 * r]);

        let v = ball.log_map_zero(&x.view());
        assert!(v.iter().all(|t| t.is_finite()), "c={c}: log_0 = {v:?}");
        let back = ball.exp_map_zero(&v.view());
        for i in 0..2 {
            assert!(
                (back[i] - x[i]).abs() < 1e-6,
                "c={c}: x={x:?} back={back:?}"
            );
        }
        assert!(ball.is_in_ball(&back.view()), "c={c}: left the ball");

        // Expected value in f64 from the f32 point actually stored. Near the
        // boundary artanh has slope 1/(1 - z^2), about 5e5 here, so f32
        // rounding (eps about 6e-8) inside the library moves d by up to about
        // 0.03 on a value near 10: a relative tolerance of 1e-2.
        let origin = Array1::<f32>::zeros(2);
        let d = ball.distance(&origin.view(), &x.view());
        let norm = ((x[0] as f64).powi(2) + (x[1] as f64).powi(2)).sqrt();
        let sqrt_c = (c as f64).sqrt();
        let expected = 2.0 / sqrt_c * (sqrt_c * norm).atanh();
        assert!(
            ((d as f64) - expected).abs() < 1e-2 * expected,
            "c={c}: d={d} expected={expected}"
        );
    }
}
