//! Rotor boosts vs the exp/log path for the Lorentz model: agreement, norm
//! drift, and per-operation cost.
//!
//! Measurement only. Nothing here is on the default code path; see
//! `docs/rotor-experiment.md` for the recorded numbers and the verdict.
//!
//! Run: cargo run --release --features rotor-experiment --example rotor_experiment

use hyperball::lorentz::LorentzModel;
use hyperball::rotor_experiment::{
    apply_rotor, axis_boost_rotor, boost_from_tangent, boost_rotor, mv_from_point, point_from_mv,
    rotor_unit_drift, translation_rotor, Minkowski,
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

    // --- Cost ------------------------------------------------------------
    let lorentz = LorentzModel::<f64>::new(1.0);
    let x = random_point(&mut ChaCha8Rng::seed_from_u64(7), &lorentz);
    let y = random_point(&mut ChaCha8Rng::seed_from_u64(11), &lorentz);
    let z = random_point(&mut ChaCha8Rng::seed_from_u64(13), &lorentz);
    let probe = random_point(&mut ChaCha8Rng::seed_from_u64(23), &lorentz);
    let mut rng = ChaCha8Rng::seed_from_u64(17);
    let origin = lorentz.origin(D);
    let v_origin = {
        let mut t = random_tangent(&mut rng, &lorentz, &origin);
        t[0] = 0.0;
        t
    };
    let v = random_tangent(&mut rng, &lorentz, &x);
    let w = random_tangent(&mut rng, &lorentz, &x);

    let rotor_origin: Rotor = boost_from_tangent(&lorentz, &origin.view(), &v_origin.view());
    let rotor_at_x: Rotor = boost_from_tangent(&lorentz, &x.view(), &v.view());
    let rot_xy: Rotor = translation_rotor(&lorentz, &x.view(), &y.view());
    let log_xy = lorentz.log_map(&x.view(), &y.view());

    let translate_exp = |from: &Array1<f64>, to: &Array1<f64>, p: &Array1<f64>| -> Array1<f64> {
        let v = lorentz.log_map(&from.view(), &to.view());
        let moved = lorentz.parallel_transport(&from.view(), &p.view(), &v.view());
        lorentz.exp_map(&p.view(), &moved.view())
    };

    let mut timings: Vec<(&str, f64, f64)> = Vec::new();

    let exp_ns = bench(|| lorentz.exp_map(&origin.view(), &v_origin.view()));
    let rotor_ns = bench(|| apply_rotor(&rotor_origin, &origin.view()));
    timings.push(("boost of origin (rotor prebuilt)", exp_ns, rotor_ns));

    let exp_ns = bench(|| lorentz.exp_map(&x.view(), &v.view()));
    let rotor_ns = bench(|| apply_rotor(&rotor_at_x, &x.view()));
    timings.push(("boost of point (rotor prebuilt)", exp_ns, rotor_ns));

    let exp_ns = bench(|| lorentz.exp_map(&x.view(), &v.view()));
    let rotor_ns = bench(|| {
        let r: Rotor = boost_from_tangent(&lorentz, &x.view(), &v.view());
        apply_rotor(&r, &x.view())
    });
    timings.push(("boost of point (rotor built)", exp_ns, rotor_ns));

    let exp_ns = bench(|| lorentz.exp_map(&x.view(), &log_xy.view()));
    let rotor_ns = bench(|| apply_rotor(&rot_xy, &x.view()));
    timings.push(("translate x->y (log shared)", exp_ns, rotor_ns));

    let exp_ns = bench(|| translate_exp(&x, &y, &x));
    let rotor_ns = bench(|| {
        let r: Rotor = translation_rotor(&lorentz, &x.view(), &y.view());
        apply_rotor(&r, &x.view())
    });
    timings.push(("translate x->y (rotor built)", exp_ns, rotor_ns));

    let exp_ns = bench(|| lorentz.parallel_transport(&x.view(), &y.view(), &w.view()));
    let rotor_ns = bench(|| apply_rotor(&rot_xy, &w.view()));
    timings.push(("parallel transport (rotor prebuilt)", exp_ns, rotor_ns));

    let exp_ns = bench(|| translate_exp(&y, &z, &translate_exp(&x, &y, &probe)));
    let rotor_ns = bench(|| {
        let first: Rotor = translation_rotor(&lorentz, &x.view(), &y.view());
        let second: Rotor = translation_rotor(&lorentz, &y.view(), &z.view());
        apply_rotor(&second.geometric(&first), &probe.view())
    });
    timings.push(("compose 2 translations", exp_ns, rotor_ns));

    println!("\nns/op (best of 7 rounds of >=80 ms each, release build)");
    println!(
        "  {:<38} {:>11} {:>11} {:>7}",
        "operation", "exp/log ns", "rotor ns", "ratio"
    );
    for (label, exp_ns, rotor_ns) in &timings {
        println!(
            "  {:<38} {:>11.1} {:>11.1} {:>6.1}x",
            label,
            exp_ns,
            rotor_ns,
            rotor_ns / exp_ns
        );
    }
    // Where the rotor time goes: the sandwich on already-built multivectors, or
    // the heap-allocated conversions around it.
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
        (
            "sandwich + conversions",
            bench(|| apply_rotor(&rotor_at_x, &x.view())),
        ),
    ];
    println!("\nrotor path breakdown (ns/op)");
    for (label, ns) in &breakdown {
        println!("  {:<38} {:>11.1}", label, ns);
    }
    breakdown.clear();
    let _ = black_box((mv_x, mv_probe));
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
