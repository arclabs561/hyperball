//! Pins the agreement measured in `docs/rotor-experiment.md` so the rotor path
//! cannot drift away from the exp/log path silently.
//!
//! Bounds are ~100x the values recorded there, which keeps the test off the
//! noise floor while still catching a sign error, a wrong rapidity, or a
//! dropped normalization.
#![cfg(feature = "rotor-experiment")]

use hyperball::lorentz::LorentzModel;
use hyperball::rotor_experiment::{
    apply_rotor, boost_from_tangent, point_from_mv, rotor_unit_drift, translation_rotor, Minkowski,
};
use ndarray::{Array1, ArrayView1};

const D: usize = 3;
type Rotor = Minkowski<D>;

/// Measured: 1.3e-13 for the boost path, 6.5e-14 for translation.
const AGREEMENT_BOUND: f64 = 1e-11;
/// Measured: 7.3e-15.
const ROTOR_UNIT_BOUND: f64 = 1e-13;
/// Measured: 1.9e-14 (rotor) and 1.2e-13 (exp).
const NORM_DRIFT_BOUND: f64 = 1e-11;

/// Unit tangent at `x`: `u + c<u,x>_L x` removes the normal component, then the
/// Minkowski norm normalizes.
fn tangent(x: &Array1<f64>, space: &[f64], c: f64) -> Array1<f64> {
    let mut u = Array1::zeros(D + 1);
    for (i, value) in space.iter().enumerate() {
        u[i + 1] = *value;
    }
    let inner = -u[0] * x[0] + u.slice(ndarray::s![1..]).dot(&x.slice(ndarray::s![1..]));
    let mut out = u.clone();
    for i in 0..=D {
        out[i] += c * inner * x[i];
    }
    let norm = out.iter().skip(1).map(|v| v * v).sum::<f64>().sqrt();
    out.mapv(|value| value / norm)
}

fn scaled_diff(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs() / x.abs().max(1.0))
        .fold(0.0, f64::max)
}

fn norm_residual(lorentz: &LorentzModel<f64>, p: &ArrayView1<f64>) -> f64 {
    let target = -1.0 / lorentz.c;
    let scale: f64 = p.iter().map(|x| x * x).sum();
    (lorentz.minkowski_dot(p, p) - target).abs() / scale.max(target.abs())
}

/// Points, directions and rapidities that cover the measured range, including
/// the 1e-8 and near-pi ends.
fn sweep() -> Vec<(f64, Array1<f64>, Array1<f64>, f64)> {
    let rapidities = [
        1e-6,
        1e-4,
        0.01,
        0.5,
        1.0,
        2.0,
        std::f64::consts::PI - 1e-9,
        std::f64::consts::PI,
        std::f64::consts::PI + 1e-9,
        6.0,
    ];
    let mut cases = Vec::new();
    for c in [1.0, 2.5] {
        let lorentz = LorentzModel::<f64>::new(c);
        for point in [
            vec![0.0, 0.0, 0.0],
            vec![0.4, -0.2, 0.1],
            vec![-0.9, 0.6, 0.3],
            vec![1.2, -1.1, 0.7],
        ] {
            let x = lorentz.from_euclidean(&ArrayView1::from(&point));
            let v = tangent(&x, &[0.3, -0.5, 0.8], c);
            for &rapidity in &rapidities {
                cases.push((c, x.clone(), v.clone(), rapidity));
            }
        }
    }
    cases
}

#[test]
fn rotor_boost_matches_exp_map() {
    let mut worst = 0.0f64;
    let mut worst_norm = 0.0f64;
    let mut worst_unit = 0.0f64;
    for (c, x, v, rapidity) in sweep() {
        let lorentz = LorentzModel::<f64>::new(c);
        let v = v.mapv(|value| value * rapidity);

        let exp_point = lorentz.exp_map(&x.view(), &v.view());
        let rotor: Rotor = boost_from_tangent(&lorentz, &x.view(), &v.view());
        let rotor_point = apply_rotor(&rotor, &x.view());

        worst = worst.max(scaled_diff(&exp_point, &rotor_point));
        worst_norm = worst_norm.max(norm_residual(&lorentz, &rotor_point.view()));
        worst_unit = worst_unit.max(rotor_unit_drift(&rotor));
    }
    assert!(
        worst <= AGREEMENT_BOUND,
        "rotor and exp_map diverged: {worst:.3e} > {AGREEMENT_BOUND:.1e}"
    );
    assert!(
        worst_norm <= NORM_DRIFT_BOUND,
        "rotor left the hyperboloid: {worst_norm:.3e}"
    );
    assert!(
        worst_unit <= ROTOR_UNIT_BOUND,
        "rotor is not unit: |R R~ - 1| = {worst_unit:.3e}"
    );
}

#[test]
fn rotor_translation_and_transport_match_the_closed_forms() {
    let mut worst = 0.0f64;
    for (c, x, _, _) in sweep().into_iter().step_by(4) {
        let lorentz = LorentzModel::<f64>::new(c);
        for target in [vec![0.5, 0.4, -0.3], vec![-0.7, 0.2, 0.9]] {
            let y = lorentz.from_euclidean(&ArrayView1::from(&target));
            let log_xy = lorentz.log_map(&x.view(), &y.view());
            let exp_point = lorentz.exp_map(&x.view(), &log_xy.view());

            let rotor: Rotor = translation_rotor(&lorentz, &x.view(), &y.view());
            let rotor_point = apply_rotor(&rotor, &x.view());

            worst = worst.max(scaled_diff(&exp_point, &rotor_point));
            assert!(
                scaled_diff(&rotor_point, &y) <= AGREEMENT_BOUND,
                "translation rotor missed its target: {:e}",
                scaled_diff(&rotor_point, &y)
            );

            let w = tangent(&x, &[0.6, 0.1, -0.4], c);
            let closed_form = lorentz.parallel_transport(&x.view(), &y.view(), &w.view());
            let via_rotor = apply_rotor(&rotor, &w.view());
            worst = worst.max(scaled_diff(&closed_form, &via_rotor));
        }
    }
    assert!(
        worst <= AGREEMENT_BOUND,
        "rotor translation/transport diverged: {worst:.3e}"
    );
}

/// `exp_map` zeroes boosts below `|v| = sqrt(1e-15)`; the rotor does not. Pin
/// that documented asymmetry rather than letting it appear as a regression.
#[test]
fn exp_map_clamps_the_smallest_boost_the_rotor_applies() {
    let lorentz = LorentzModel::<f64>::new(1.0);
    let x = lorentz.from_euclidean(&ArrayView1::from(&[0.4, -0.2, 0.3]));
    let v = tangent(&x, &[0.3, 0.8, -0.5], 1.0).mapv(|value| value * 1e-8);

    let exp_point = lorentz.exp_map(&x.view(), &v.view());
    assert_eq!(
        exp_point, x,
        "exp_map's small-vector guard is expected to clamp"
    );

    let rotor: Rotor = boost_from_tangent(&lorentz, &x.view(), &v.view());
    let rotor_point = apply_rotor(&rotor, &x.view());
    let moved = scaled_diff(&x, &rotor_point);
    assert!(
        (1e-9..1e-7).contains(&moved),
        "rotor should move the point by ~1e-8, moved {moved:.3e}"
    );
    let _ = point_from_mv(&rotor);
}
