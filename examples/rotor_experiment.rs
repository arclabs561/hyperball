//! Rotor boosts vs the exp/log path for the Lorentz model: agreement, norm
//! drift, and per-operation cost.
//!
//! Measurement only. Nothing here is on the default code path; see
//! `docs/rotor-experiment.md` for the recorded numbers and the verdict.
//!
//! Run: cargo run --release --features rotor-experiment --example rotor_experiment

use hyperball::lorentz::LorentzModel;
use hyperball::rotor_experiment::{
    apply_rotor, axis_boost_rotor, axis_rotor, boost_from_tangent, boost_rotor, mv_from_point,
    point_from_mv, point_to_vector, precomputed_rotor, rotor_unit_drift, translation_rotor,
    vector_to_point, Minkowski,
};
use ndarray::{Array1, ArrayView1};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::hint::black_box;
use std::time::Instant;

/// Space dimension of the benchmarked hyperboloid; the Clifford algebra is
/// `Cl(3, 1)` with 16 blades.
const D: usize = 3;

type Rotor = Minkowski<D>;

/// Fixed rapidities that stress both ends of the range: below the closed forms'
/// useful resolution, at pi, and far past it.
const EDGE_RAPIDITIES: [f64; 8] = [
    1e-8,
    1e-6,
    1e-4,
    std::f64::consts::PI - 1e-9,
    std::f64::consts::PI,
    std::f64::consts::PI + 1e-9,
    6.0,
    12.0,
];

struct Case {
    x: Array1<f64>,
    /// Tangent direction at `x`, unit in the Minkowski form.
    v: Array1<f64>,
    /// Second point, target of the geodesic translation.
    y: Array1<f64>,
    /// Tangent vector at `x` to transport.
    w: Array1<f64>,
    /// Boost magnitude applied along `v`.
    rapidity: f64,
    c: f64,
}

fn random_point(rng: &mut ChaCha8Rng, lorentz: &LorentzModel<f64>) -> Array1<f64> {
    let coords: Vec<f64> = (0..D).map(|_| rng.random_range(-1.5..1.5f64)).collect();
    lorentz.from_euclidean(&ArrayView1::from(&coords))
}

/// Unit tangent at `x`: project a random ambient vector onto the tangent space
/// (`u + c<u,x>x`) and normalize in the Minkowski form.
fn random_tangent(
    rng: &mut ChaCha8Rng,
    lorentz: &LorentzModel<f64>,
    x: &Array1<f64>,
) -> Array1<f64> {
    let mut u = Array1::zeros(D + 1);
    for i in 1..=D {
        u[i] = rng.random_range(-1.0..1.0f64);
    }
    let inner = lorentz.minkowski_dot(&u.view(), &x.view());
    let mut tangent = u.clone();
    for i in 0..=D {
        tangent[i] += lorentz.c * inner * x[i];
    }
    let norm = lorentz
        .minkowski_dot(&tangent.view(), &tangent.view())
        .sqrt();
    tangent.mapv(|value| value / norm)
}

fn build_cases() -> Vec<Case> {
    let mut rng = ChaCha8Rng::seed_from_u64(0x5eed_2026);
    let mut cases = Vec::new();
    for c in [1.0, 2.5] {
        let lorentz = LorentzModel::<f64>::new(c);
        for _ in 0..600 {
            let x = random_point(&mut rng, &lorentz);
            cases.push(Case {
                v: random_tangent(&mut rng, &lorentz, &x),
                w: random_tangent(&mut rng, &lorentz, &x),
                y: random_point(&mut rng, &lorentz),
                rapidity: rng.random_range(0.0..4.0f64),
                x,
                c,
            });
        }
        for &rapidity in &EDGE_RAPIDITIES {
            let x = random_point(&mut rng, &lorentz);
            cases.push(Case {
                v: random_tangent(&mut rng, &lorentz, &x),
                w: random_tangent(&mut rng, &lorentz, &x),
                y: random_point(&mut rng, &lorentz),
                rapidity,
                x,
                c,
            });
        }
    }
    cases
}

/// `|<p,p>_L + 1/c|` relative to the magnitude of the terms that produce it, so
/// a large-rapidity point is not scored by the cancellation inside the check.
fn norm_residual(lorentz: &LorentzModel<f64>, p: &ArrayView1<f64>) -> f64 {
    let target = -1.0 / lorentz.c;
    let inner = lorentz.minkowski_dot(p, p);
    // Normalize by the magnitude of the terms that form the constraint, not by
    // its value: at rapidity 12 the coordinates reach 1e7 and the subtraction
    // `-t^2 + |s|^2` loses every digit of a 1/c target either way.
    let scale = p.iter().map(|x| x * x).sum::<f64>();
    (inner - target).abs() / scale.max(target.abs())
}

/// `max_i |a_i - b_i|`, in absolute coordinates.
fn max_abs_diff(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// `max_i |a_i - b_i| / max(1, |a_i|)`: the scale-free disagreement, so
/// coordinates of order `cosh(rapidity)` do not swamp the comparison.
fn max_scaled_diff(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs() / x.abs().max(1.0))
        .fold(0.0, f64::max)
}

/// Worst agreement and norm residual for a boost-like operation.
#[derive(Default)]
struct Agreement {
    max_abs: f64,
    max_scaled: f64,
    norm_drift_exp: f64,
    norm_drift_rotor: f64,
    rotor_drift: f64,
}

impl Agreement {
    fn record(
        &mut self,
        lorentz: &LorentzModel<f64>,
        exp_point: &Array1<f64>,
        rotor: &Rotor,
        rotor_point: &Array1<f64>,
    ) {
        self.max_abs = self.max_abs.max(max_abs_diff(exp_point, rotor_point));
        self.max_scaled = self.max_scaled.max(max_scaled_diff(exp_point, rotor_point));
        self.norm_drift_exp = self
            .norm_drift_exp
            .max(norm_residual(lorentz, &exp_point.view()));
        self.norm_drift_rotor = self
            .norm_drift_rotor
            .max(norm_residual(lorentz, &rotor_point.view()));
        self.rotor_drift = self.rotor_drift.max(rotor_unit_drift(rotor));
    }
}

fn main() {
    let cases = build_cases();
    println!("rotor-experiment: Cl({D}, 1), {} cases\n", cases.len());

    // --- Agreement ------------------------------------------------------
    let mut origin_boost = Agreement::default();
    let mut point_boost = Agreement::default();
    let mut translate = Agreement::default();
    let mut translate_to_target = 0.0f64;
    let mut general_vs_exp = 0.0f64;
    let mut general_vs_sandwich = 0.0f64;
    let mut general_cases = 0usize;
    let mut transport = 0.0f64;
    let mut clamped = (0.0f64, 0.0f64);
    for case in &cases {
        let lorentz = LorentzModel::<f64>::new(case.c);
        let origin = lorentz.origin(D);
        let v_origin = {
            let mut t = case.v.mapv(|value| value * case.rapidity);
            t[0] = 0.0;
            let space_norm = t.iter().skip(1).map(|x| x * x).sum::<f64>().sqrt();
            t.mapv(|value| value / space_norm * case.rapidity)
        };
        let v = case.v.mapv(|value| value * case.rapidity);

        // `exp_map` treats `v_norm_sq < 1e-15` (|v| < 3.2e-8) as zero and
        // returns `x` unchanged, while the rotor still applies the boost. Probe
        // that case once and keep it out of the headline agreement.
        if case.rapidity < 1e-6 {
            let tiny = case.v.mapv(|value| value * 1e-8);
            let exp_point = lorentz.exp_map(&case.x.view(), &tiny.view());
            let rotor: Rotor = boost_from_tangent(&lorentz, &case.x.view(), &tiny.view());
            let rotor_point = apply_rotor(&rotor, &case.x.view());
            clamped.0 = clamped.0.max(max_scaled_diff(&exp_point, &rotor_point));
            clamped.1 = clamped.1.max(max_scaled_diff(&exp_point, &case.x));
            continue;
        }

        // Origin boost.
        let exp_point = lorentz.exp_map(&origin.view(), &v_origin.view());
        let rotor = boost_from_tangent(&lorentz, &origin.view(), &v_origin.view());
        origin_boost.record(
            &lorentz,
            &exp_point,
            &rotor,
            &apply_rotor(&rotor, &origin.view()),
        );

        // Boost at a general point.
        let exp_point = lorentz.exp_map(&case.x.view(), &v.view());
        let rotor = boost_from_tangent(&lorentz, &case.x.view(), &v.view());
        point_boost.record(
            &lorentz,
            &exp_point,
            &rotor,
            &apply_rotor(&rotor, &case.x.view()),
        );

        // Geodesic translation x -> y.
        let log_xy = lorentz.log_map(&case.x.view(), &case.y.view());
        let exp_point = lorentz.exp_map(&case.x.view(), &log_xy.view());
        let rotor = translation_rotor(&lorentz, &case.x.view(), &case.y.view());
        let rotor_point = apply_rotor(&rotor, &case.x.view());
        translate.record(&lorentz, &exp_point, &rotor, &rotor_point);
        translate_to_target = translate_to_target.max(max_abs_diff(&rotor_point, &case.y));
        // The same translation through the precomputed action of a
        // general-plane rotor (geomalg's `Rotor::from_even`).
        let general = precomputed_rotor(rotor.clone()).expect("translation rotor is unit");
        let general_point =
            vector_to_point::<D>(&general.apply_vector(&point_to_vector::<D>(&case.x.view())));
        general_vs_exp = general_vs_exp.max(max_scaled_diff(&exp_point, &general_point));
        general_vs_sandwich =
            general_vs_sandwich.max(max_scaled_diff(&rotor_point, &general_point));
        general_cases += 1;

        // Parallel transport of a tangent vector along x -> y.
        let closed_form =
            lorentz.parallel_transport(&case.x.view(), &case.y.view(), &case.w.view());
        transport = transport.max(max_abs_diff(
            &closed_form,
            &apply_rotor(&rotor, &case.w.view()),
        ));
    }

    // Axis-aligned `Cl::rotor` constructor against the explicit bivector.
    let mut constructor_agreement = 0.0f64;
    for axis in 0..D {
        for &rapidity in &EDGE_RAPIDITIES {
            let time_axis = Minkowski::<D>::basis(D);
            let mut dir = vec![0.0; D + 1];
            dir[axis] = 1.0;
            let via_constructor = axis_boost_rotor::<D>(axis, rapidity);
            let via_bivector = boost_rotor(&time_axis, &Minkowski::<D>::vector(&dir), rapidity);
            constructor_agreement = constructor_agreement.max(max_abs_diff(
                &point_from_mv(&via_constructor),
                &point_from_mv(&via_bivector),
            ));
        }
    }

    println!("agreement: max |exp - rotor| over all components, absolute and scaled");
    println!("  {:<22} {:>12} {:>12}", "operation", "absolute", "scaled");
    println!(
        "  {:<22} {:>12.3e} {:>12.3e}",
        "origin boost", origin_boost.max_abs, origin_boost.max_scaled
    );
    println!(
        "  {:<22} {:>12.3e} {:>12.3e}",
        "boost at point", point_boost.max_abs, point_boost.max_scaled
    );
    println!(
        "  {:<22} {:>12.3e} {:>12.3e}",
        "translate x -> y", translate.max_abs, translate.max_scaled
    );
    println!(
        "  {:<22} {:>12.3e} {:>12.3e}",
        "translate reaches y", translate_to_target, translate_to_target
    );
    println!(
        "  {:<22} {:>12.3e} {:>12.3e}",
        "parallel transport", transport, transport
    );
    println!(
        "  {:<22} {:>12.3e} {:>12.3e}",
        "Cl::rotor vs bivector", constructor_agreement, constructor_agreement
    );

    println!("\nnote: one case at rapidity 1e-8 per curvature, where exp_map's guard",);
    println!(
        "      (v_norm_sq < 1e-15) returns x unchanged: exp-vs-rotor {:.3e}, exp-vs-x {:.3e}",
        clamped.0, clamped.1
    );
    println!("      (scaled); excluded from the table above.");

    println!("\nMinkowski-norm residual |<p,p>_L + 1/c|, relative to coordinate scale");
    println!(
        "  origin boost   exp {:.3e}  rotor {:.3e}",
        origin_boost.norm_drift_exp, origin_boost.norm_drift_rotor
    );
    println!(
        "  boost at point exp {:.3e}  rotor {:.3e}",
        point_boost.norm_drift_exp, point_boost.norm_drift_rotor
    );
    println!(
        "  translate      exp {:.3e}  rotor {:.3e}",
        translate.norm_drift_exp, translate.norm_drift_rotor
    );
    println!(
        "  rotor unit drift |R R~ - 1| {:.3e}",
        point_boost.rotor_drift.max(translate.rotor_drift)
    );

    // --- Precomputed action (geomalg::Rotor) -----------------------------
    //
    // `Rotor::new` indexes two basis vectors, so the rotor exists only for a
    // coordinate plane. Agreement with `exp_map` is still exact and not merely
    // "the same matrix": a point on the `(e(D), e(axis))` hyperbola boosted
    // along `e(axis)` is again a coordinate-plane boost, of rapidity
    // `offset + rapidity`.
    let axis_lorentz = LorentzModel::<f64>::new(1.0);
    let mut axis_rapidities = EDGE_RAPIDITIES.to_vec();
    axis_rapidities.extend([0.01, 0.5, 2.0, 3.9]);
    let mut precomputed_vs_exp = 0.0f64;
    let mut precomputed_clamp = 0.0f64;
    let mut precomputed_configurations = 0usize;
    let mut precomputed_vs_sandwich = 0.0f64;
    let mut precomputed_norm = 0.0f64;
    for axis in 0..D {
        for &offset in &[0.0f64, 0.5, 1.0, 2.0] {
            for &rapidity in &axis_rapidities {
                let mut x = Array1::zeros(D + 1);
                x[0] = offset.cosh();
                x[axis + 1] = offset.sinh();
                // Tangent at `x` of Minkowski norm `rapidity`.
                let mut v = Array1::zeros(D + 1);
                v[0] = offset.sinh() * rapidity;
                v[axis + 1] = offset.cosh() * rapidity;

                let rotor = axis_rotor::<D>(axis, rapidity);
                let flat = point_to_vector::<D>(&x.view());
                let single = vector_to_point::<D>(&rotor.apply_vector(&flat));
                let sandwich = point_from_mv(&rotor.apply(&mv_from_point::<D>(&x.view())));
                let exp_point = axis_lorentz.exp_map(&x.view(), &v.view());

                // `exp_map` clamps below |v| = 3.2e-8 and returns `x`; the
                // rotor still boosts. That is the same documented asymmetry as
                // the general table, so it is reported apart, not averaged in.
                if rapidity < 1e-6 {
                    precomputed_clamp = precomputed_clamp.max(max_scaled_diff(&exp_point, &single));
                } else {
                    precomputed_vs_exp =
                        precomputed_vs_exp.max(max_scaled_diff(&exp_point, &single));
                    precomputed_configurations += 1;
                }
                precomputed_vs_sandwich =
                    precomputed_vs_sandwich.max(max_scaled_diff(&sandwich, &single));
                precomputed_norm =
                    precomputed_norm.max(norm_residual(&axis_lorentz, &single.view()));
            }
        }
    }

    // Batched action must reproduce the single call: `apply_vectors` is the
    // same kernel in a loop, so this has to be bitwise, not approximate.
    const BATCH: usize = 1000;
    let mut batch_rng = ChaCha8Rng::seed_from_u64(0xb00b);
    let batch: Vec<Array1<f64>> = (0..BATCH)
        .map(|_| random_point(&mut batch_rng, &axis_lorentz))
        .collect();
    let mut input = Vec::with_capacity(BATCH * (D + 1));
    for point in &batch {
        input.extend_from_slice(&point_to_vector::<D>(&point.view()));
    }
    let mut output = vec![0.0; input.len()];
    let batch_rotor = axis_rotor::<D>(0, 0.7);
    batch_rotor.apply_vectors(&input, &mut output);
    let mut batch_mismatch = 0.0f64;
    let mut batch_norm = 0.0f64;
    for (chunk_in, chunk_out) in input.chunks_exact(D + 1).zip(output.chunks_exact(D + 1)) {
        let single = batch_rotor.apply_vector(chunk_in);
        for (a, b) in chunk_out.iter().zip(&single) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "batched action differs from single"
            );
            batch_mismatch = batch_mismatch.max((a - b).abs());
        }
        let point = vector_to_point::<D>(chunk_out);
        batch_norm = batch_norm.max(norm_residual(&axis_lorentz, &point.view()));
    }

    println!("\nprecomputed action (geomalg::Rotor, coordinate planes only)");
    println!(
        "  apply_vector vs exp_map    {:.3e} scaled ({precomputed_configurations} configurations)",
        precomputed_vs_exp
    );
    println!("  apply_vector vs sandwich   {precomputed_vs_sandwich:.3e} scaled");
    println!("  apply_vector norm residual {precomputed_norm:.3e}");
    println!("  rapidity 1e-8 (exp_map clamps, rotor does not) {precomputed_clamp:.3e} scaled");
    println!(
        "  batch of {BATCH} vs single call   {batch_mismatch:.3e} (bitwise equal), norm {batch_norm:.3e}"
    );
    println!("\nprecomputed action at general points (geomalg::Rotor::from_even)");
    println!(
        "  translate x->y vs exp_map  {general_vs_exp:.3e} scaled ({general_cases} configurations)"
    );
    println!("  translate x->y vs sandwich {general_vs_sandwich:.3e} scaled");

    // --- Cost ------------------------------------------------------------
    let lorentz = LorentzModel::<f64>::new(1.0);
    let x = random_point(&mut ChaCha8Rng::seed_from_u64(7), &lorentz);
    let y = random_point(&mut ChaCha8Rng::seed_from_u64(11), &lorentz);
    let z = random_point(&mut ChaCha8Rng::seed_from_u64(13), &lorentz);
    let probe = random_point(&mut ChaCha8Rng::seed_from_u64(23), &lorentz);
    let mut rng = ChaCha8Rng::seed_from_u64(17);
    let origin = lorentz.origin(D);
    let v = random_tangent(&mut rng, &lorentz, &x);
    let w = random_tangent(&mut rng, &lorentz, &x);

    let rotor_at_x: Rotor = boost_from_tangent(&lorentz, &x.view(), &v.view());
    let rot_xy: Rotor = translation_rotor(&lorentz, &x.view(), &y.view());
    let log_xy = lorentz.log_map(&x.view(), &y.view());

    let translate_exp = |from: &Array1<f64>, to: &Array1<f64>, p: &Array1<f64>| -> Array1<f64> {
        let v = lorentz.log_map(&from.view(), &to.view());
        let moved = lorentz.parallel_transport(&from.view(), &p.view(), &v.view());
        lorentz.exp_map(&p.view(), &moved.view())
    };

    // Only the origin boost exists on all three paths: `Rotor::new` needs a
    // coordinate plane. `None` marks a path that cannot be built at all.
    let origin_rapidity = 0.7;
    let mut v_axis = Array1::zeros(D + 1);
    v_axis[1] = origin_rapidity;
    let axis_even: Rotor = boost_rotor(
        &Minkowski::<D>::basis(D),
        &Minkowski::<D>::basis(1),
        origin_rapidity,
    );
    let precomputed_origin = axis_rotor::<D>(1, origin_rapidity);
    let flat_origin = point_to_vector::<D>(&origin.view());

    // Each distinct expression is timed exactly once, so two rows can never
    // disagree about the cost of the same code.
    let per_point = |total: f64| total / batch.len() as f64;

    let exp_origin = bench(|| lorentz.exp_map(&origin.view(), &v_axis.view()));
    let exp_point = bench(|| lorentz.exp_map(&x.view(), &v.view()));
    let exp_log_shared = bench(|| lorentz.exp_map(&x.view(), &log_xy.view()));
    let exp_translate = bench(|| translate_exp(&x, &y, &x));
    let exp_transport = bench(|| lorentz.parallel_transport(&x.view(), &y.view(), &w.view()));
    let exp_compose = bench(|| translate_exp(&y, &z, &translate_exp(&x, &y, &probe)));
    let exp_batch = per_point(bench(|| {
        for point in &batch {
            black_box(lorentz.exp_map(&origin.view(), &v_axis.view()));
            black_box(point);
        }
    }));

    let sandwich_origin = bench(|| apply_rotor(&axis_even, &origin.view()));
    let sandwich_origin_built = bench(|| {
        let r: Rotor = boost_from_tangent(&lorentz, &origin.view(), &v_axis.view());
        apply_rotor(&r, &origin.view())
    });
    let sandwich_point = bench(|| apply_rotor(&rotor_at_x, &x.view()));
    let sandwich_point_built = bench(|| {
        let r: Rotor = boost_from_tangent(&lorentz, &x.view(), &v.view());
        apply_rotor(&r, &x.view())
    });
    let sandwich_log_shared = bench(|| apply_rotor(&rot_xy, &x.view()));
    let sandwich_translate = bench(|| {
        let r: Rotor = translation_rotor(&lorentz, &x.view(), &y.view());
        apply_rotor(&r, &x.view())
    });
    let sandwich_transport = bench(|| apply_rotor(&rot_xy, &w.view()));
    let sandwich_compose = bench(|| {
        let first: Rotor = translation_rotor(&lorentz, &x.view(), &y.view());
        let second: Rotor = translation_rotor(&lorentz, &y.view(), &z.view());
        apply_rotor(&second.geometric(&first), &probe.view())
    });
    let sandwich_batch = per_point(bench(|| {
        for point in &batch {
            black_box(apply_rotor(&axis_even, &point.view()));
        }
    }));
    let sandwich_batch_built = per_point(bench(|| {
        let r: Rotor = boost_from_tangent(&lorentz, &origin.view(), &v_axis.view());
        for point in &batch {
            black_box(apply_rotor(&r, &point.view()));
        }
    }));

    let build = bench(|| axis_rotor::<D>(1, origin_rapidity));
    let pre_origin = bench(|| precomputed_origin.apply_vector(&flat_origin));
    let pre_origin_built = bench(|| {
        let r = axis_rotor::<D>(1, origin_rapidity);
        r.apply_vector(&flat_origin)
    });
    let flat_x = point_to_vector::<D>(&x.view());
    let pre_at_x = precomputed_rotor(rotor_at_x.clone()).expect("unit rotor");
    let pre_xy = precomputed_rotor(rot_xy.clone()).expect("unit rotor");
    let pre_point = bench(|| pre_at_x.apply_vector(&flat_x));
    let pre_point_built = bench(|| {
        let r: Rotor = boost_from_tangent(&lorentz, &x.view(), &v.view());
        precomputed_rotor(r)
            .expect("unit rotor")
            .apply_vector(&flat_x)
    });
    let pre_log_shared = bench(|| pre_xy.apply_vector(&flat_x));
    let pre_translate = bench(|| {
        let r: Rotor = translation_rotor(&lorentz, &x.view(), &y.view());
        precomputed_rotor(r)
            .expect("unit rotor")
            .apply_vector(&flat_x)
    });
    let pre_batch = per_point(bench(|| {
        precomputed_origin.apply_vectors(&input, &mut output);
        black_box(&output);
    }));
    let pre_batch_built = per_point(bench(|| {
        let r = axis_rotor::<D>(1, origin_rapidity);
        r.apply_vectors(&input, &mut output);
    }));

    let timings: Vec<(&str, f64, f64, Option<f64>)> = vec![
        (
            "boost of origin (rotor prebuilt)",
            exp_origin,
            sandwich_origin,
            Some(pre_origin),
        ),
        (
            "boost of origin (rotor built)",
            exp_origin,
            sandwich_origin_built,
            Some(pre_origin_built),
        ),
        (
            "boost of point (rotor prebuilt)",
            exp_point,
            sandwich_point,
            Some(pre_point),
        ),
        (
            "boost of point (rotor built)",
            exp_point,
            sandwich_point_built,
            Some(pre_point_built),
        ),
        (
            "translate x->y (log shared)",
            exp_log_shared,
            sandwich_log_shared,
            Some(pre_log_shared),
        ),
        (
            "translate x->y (rotor built)",
            exp_translate,
            sandwich_translate,
            Some(pre_translate),
        ),
        (
            "parallel transport (rotor prebuilt)",
            exp_transport,
            sandwich_transport,
            None,
        ),
        (
            "compose 2 translations",
            exp_compose,
            sandwich_compose,
            None,
        ),
        (
            "1000 pts, rotor prebuilt (per pt)",
            exp_batch,
            sandwich_batch,
            Some(pre_batch),
        ),
        (
            "1000 pts, rotor built once (per pt)",
            exp_batch,
            sandwich_batch_built,
            Some(pre_batch_built),
        ),
    ];

    println!("\nns/op (best of 7 rounds of >=80 ms each, release build)");
    println!(
        "  {:<40} {:>9} {:>9} {:>11}",
        "operation", "exp/log", "sandwich", "precomputed"
    );
    for (label, exp_ns, sandwich_ns, precomputed_ns) in &timings {
        match precomputed_ns {
            Some(pre) => println!(
                "  {:<40} {:>9.1} {:>9.1} {:>11.1}",
                label, exp_ns, sandwich_ns, pre
            ),
            None => println!(
                "  {:<40} {:>9.1} {:>9.1} {:>11}",
                label, exp_ns, sandwich_ns, "-"
            ),
        }
    }
    println!("  {:<40} {:>31.1}", "Rotor::new (one rotor)", build);

    println!(
        "\nprecomputed vs exp/log: single {:.1}x, batch of {} {:.1}x per point",
        exp_origin / pre_origin,
        batch.len(),
        exp_batch / pre_batch
    );
    println!(
        "breakeven for one rotor: build {build:.0} ns / ({exp_origin:.1} - {pre_origin:.1}) ns = \
         {:.0} points",
        build / (exp_origin - pre_origin)
    );

    // Where each path's time goes.
    let mv_x = mv_from_point::<D>(&x.view());
    let mv_probe = mv_from_point::<D>(&probe.view());
    let mut breakdown: Vec<(&str, f64)> = vec![
        (
            "mv_from_point (alloc + fill)",
            bench(|| mv_from_point::<D>(&x.view())),
        ),
        (
            "point_from_mv (read out)",
            bench(|| point_from_mv(&mv_probe)),
        ),
        (
            "one geometric product",
            bench(|| rotor_at_x.geometric(&rot_xy)),
        ),
        (
            "sandwich, no conversions",
            bench(|| rotor_at_x.sandwich(&mv_probe)),
        ),
        // Already measured for the table; reused so the breakdown cannot
        // disagree with it.
        ("sandwich + conversions", sandwich_point),
        ("Rotor::new (build, one rotor)", build),
        (
            "Rotor::apply (sandwich, prebuilt)",
            bench(|| precomputed_origin.apply(&mv_probe)),
        ),
        ("Rotor::apply_vector (single)", pre_origin),
        ("Rotor::apply_vectors (per point)", pre_batch),
    ];
    println!("\nbreakdown (ns/op)");
    for (label, ns) in &breakdown {
        println!("  {:<40} {:>9.1}", label, ns);
    }
    breakdown.clear();
    let _ = black_box((mv_x, mv_probe, &output));
}

/// Best-of-seven wall-clock time per call, with a calibration pass to size the
/// batch.
fn bench<F, R>(mut f: F) -> f64
where
    F: FnMut() -> R,
{
    let mut iters = 1usize;
    while iters < (1 << 22) {
        let start = Instant::now();
        for _ in 0..iters {
            black_box(f());
        }
        if start.elapsed().as_secs_f64() > 0.08 {
            break;
        }
        iters *= 2;
    }
    let mut samples = Vec::with_capacity(7);
    for _ in 0..7 {
        let start = Instant::now();
        for _ in 0..iters {
            black_box(f());
        }
        samples.push(start.elapsed().as_secs_f64() * 1e9 / iters as f64);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).expect("no NaN timings"));
    // Best of seven: the minimum is the least contaminated by scheduler noise on
    // a shared machine, which matters more here than the tail.
    samples[0]
}
