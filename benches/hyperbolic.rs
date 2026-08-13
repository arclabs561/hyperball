//! Hyperbolic geometry benchmarks.
//!
//! Dimensions measured:
//! - CPU time for various operations
//! - Scaling with dimensionality
//! - Numerical precision (f32 vs f64)
//! - Precision degradation near boundary

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use hyperball::PoincareBall as LibraryPoincareBall;
use ndarray::Array1;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use skel::Manifold;

// Poincare Ball operations (standalone for benchmarking)
struct PoincareBall {
    c: f64,
}

impl PoincareBall {
    fn new(c: f64) -> Self {
        Self { c }
    }

    fn mobius_add(&self, x: &[f64], y: &[f64]) -> Vec<f64> {
        let x_norm_sq: f64 = x.iter().map(|v| v * v).sum();
        let y_norm_sq: f64 = y.iter().map(|v| v * v).sum();
        let xy: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();

        let c = self.c;
        let denom = 1.0 + 2.0 * c * xy + c * c * x_norm_sq * y_norm_sq;

        let coeff1 = 1.0 + 2.0 * c * xy + c * y_norm_sq;
        let coeff2 = 1.0 - c * x_norm_sq;

        x.iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| (coeff1 * xi + coeff2 * yi) / denom)
            .collect()
    }

    fn distance(&self, x: &[f64], y: &[f64]) -> f64 {
        let neg_x: Vec<f64> = x.iter().map(|v| -v).collect();
        let diff = self.mobius_add(&neg_x, y);
        let diff_norm: f64 = diff.iter().map(|v| v * v).sum::<f64>().sqrt();
        let c_sqrt = self.c.sqrt();

        2.0 / c_sqrt * (c_sqrt * diff_norm).atanh()
    }

    fn exp_map_zero(&self, v: &[f64]) -> Vec<f64> {
        let v_norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if v_norm < 1e-15 {
            return v.to_vec();
        }
        let c_sqrt = self.c.sqrt();
        let scale = (c_sqrt * v_norm).tanh() / (c_sqrt * v_norm);
        v.iter().map(|&x| x * scale).collect()
    }

    fn log_map_zero(&self, y: &[f64]) -> Vec<f64> {
        let y_norm: f64 = y.iter().map(|x| x * x).sum::<f64>().sqrt();
        if y_norm < 1e-15 {
            return y.to_vec();
        }
        let c_sqrt = self.c.sqrt();
        let scale = (c_sqrt * y_norm).atanh() / (c_sqrt * y_norm);
        y.iter().map(|&x| x * scale).collect()
    }
}

// Lorentz model operations
struct LorentzModel {
    c: f64,
}

impl LorentzModel {
    fn new(c: f64) -> Self {
        Self { c }
    }

    fn minkowski_dot(&self, x: &[f64], y: &[f64]) -> f64 {
        -x[0] * y[0]
            + x[1..]
                .iter()
                .zip(y[1..].iter())
                .map(|(a, b)| a * b)
                .sum::<f64>()
    }

    fn distance(&self, x: &[f64], y: &[f64]) -> f64 {
        let inner = self.minkowski_dot(x, y);
        let arg = -self.c * inner;

        // Numerical stability near arg=1
        if arg < 1.0 + 1e-7 {
            if arg <= 1.0 {
                return 0.0;
            }
            let eps = arg - 1.0;
            return (2.0 * eps).sqrt() / self.c.sqrt();
        }
        arg.acosh() / self.c.sqrt()
    }

    fn euclidean_to_hyperboloid(&self, v: &[f64]) -> Vec<f64> {
        let space_norm_sq: f64 = v.iter().map(|x| x * x).sum();
        let t = (space_norm_sq + 1.0 / self.c).sqrt();

        let mut result = vec![t];
        result.extend_from_slice(v);
        result
    }

    fn exp_map(&self, x: &[f64], v: &[f64]) -> Vec<f64> {
        let v_norm_sq = self.minkowski_dot(v, v);
        if v_norm_sq < 1e-15 {
            return x.to_vec();
        }
        let v_norm = v_norm_sq.sqrt();
        let c_sqrt = self.c.sqrt();

        let cosh_term = (c_sqrt * v_norm).cosh();
        let sinh_term = (c_sqrt * v_norm).sinh() / (c_sqrt * v_norm);

        x.iter()
            .zip(v.iter())
            .map(|(&xi, &vi)| xi * cosh_term + vi * sinh_term)
            .collect()
    }
}

fn random_point_in_ball(dim: usize, max_norm: f64, rng: &mut StdRng) -> Vec<f64> {
    let mut v: Vec<f64> = (0..dim).map(|_| rng.random::<f64>() * 2.0 - 1.0).collect();
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    let target_norm = rng.random::<f64>() * max_norm;
    for x in &mut v {
        *x *= target_norm / norm;
    }
    v
}

fn random_point_at_radius(dim: usize, radius: f64, rng: &mut StdRng) -> Vec<f64> {
    let mut v: Vec<f64> = (0..dim).map(|_| rng.random::<f64>() * 2.0 - 1.0).collect();
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    for x in &mut v {
        *x *= radius / norm;
    }
    v
}

fn random_lorentz_point(dim: usize, rng: &mut StdRng) -> Vec<f64> {
    let lorentz = LorentzModel::new(1.0);
    let euclidean: Vec<f64> = (0..dim).map(|_| rng.random::<f64>() * 2.0 - 1.0).collect();
    lorentz.euclidean_to_hyperboloid(&euclidean)
}

/// Benchmark Poincare distance computation.
fn bench_poincare_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("poincare_distance");

    let ball = PoincareBall::new(1.0);
    let mut rng = StdRng::seed_from_u64(42);

    for dim in [2, 8, 32, 64, 128, 256] {
        let x = random_point_in_ball(dim, 0.9, &mut rng);
        let y = random_point_in_ball(dim, 0.9, &mut rng);

        group.throughput(Throughput::Elements(dim as u64));

        group.bench_with_input(BenchmarkId::new("dim", dim), &dim, |b, _| {
            b.iter(|| ball.distance(black_box(&x), black_box(&y)))
        });
    }

    group.finish();
}

/// Benchmark Lorentz distance computation.
fn bench_lorentz_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("lorentz_distance");

    let lorentz = LorentzModel::new(1.0);
    let mut rng = StdRng::seed_from_u64(42);

    for dim in [2, 8, 32, 64, 128, 256] {
        let x = random_lorentz_point(dim, &mut rng);
        let y = random_lorentz_point(dim, &mut rng);

        group.throughput(Throughput::Elements(dim as u64));

        group.bench_with_input(BenchmarkId::new("dim", dim), &dim, |b, _| {
            b.iter(|| lorentz.distance(black_box(&x), black_box(&y)))
        });
    }

    group.finish();
}

/// Benchmark Mobius addition.
fn bench_mobius_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("mobius_add");

    let ball = PoincareBall::new(1.0);
    let mut rng = StdRng::seed_from_u64(42);

    for dim in [2, 8, 32, 64, 128] {
        let x = random_point_in_ball(dim, 0.9, &mut rng);
        let y = random_point_in_ball(dim, 0.9, &mut rng);

        group.throughput(Throughput::Elements(dim as u64));

        group.bench_with_input(BenchmarkId::new("dim", dim), &dim, |b, _| {
            b.iter(|| ball.mobius_add(black_box(&x), black_box(&y)))
        });
    }

    group.finish();
}

/// Benchmark the library's closed-form Poincaré parallel transport.
fn bench_poincare_parallel_transport(c: &mut Criterion) {
    let mut group = c.benchmark_group("poincare_parallel_transport");
    let ball = LibraryPoincareBall::new(1.0);
    let mut rng = StdRng::seed_from_u64(42);

    for dim in [2, 32, 256] {
        for (regime, radius) in [("interior", 0.75), ("near_boundary", 0.99)] {
            let x = Array1::from_vec(random_point_at_radius(dim, radius, &mut rng));
            let y = Array1::from_vec(random_point_at_radius(dim, radius, &mut rng));
            let v = Array1::from_vec((0..dim).map(|_| rng.random::<f64>() * 2.0 - 1.0).collect());

            group.throughput(Throughput::Elements(dim as u64));
            group.bench_with_input(BenchmarkId::new(regime, dim), &dim, |b, _| {
                b.iter(|| {
                    ball.parallel_transport(
                        black_box(&x.view()),
                        black_box(&y.view()),
                        black_box(&v.view()),
                    )
                })
            });
        }
    }

    group.finish();
}

/// Benchmark exponential map.
fn bench_exp_map(c: &mut Criterion) {
    let mut group = c.benchmark_group("exp_map");

    let ball = PoincareBall::new(1.0);
    let lorentz = LorentzModel::new(1.0);
    let mut rng = StdRng::seed_from_u64(42);

    for dim in [8, 32, 64, 128] {
        let v: Vec<f64> = (0..dim).map(|_| rng.random::<f64>() * 0.5).collect();

        group.bench_with_input(BenchmarkId::new("poincare", dim), &dim, |b, _| {
            b.iter(|| ball.exp_map_zero(black_box(&v)))
        });

        let x = random_lorentz_point(dim, &mut rng);
        let tangent: Vec<f64> = std::iter::once(0.0)
            .chain((0..dim).map(|_| rng.random::<f64>() * 0.3))
            .collect();

        group.bench_with_input(BenchmarkId::new("lorentz", dim), &dim, |b, _| {
            b.iter(|| lorentz.exp_map(black_box(&x), black_box(&tangent)))
        });
    }

    group.finish();
}

/// Benchmark batch distance computation.
fn bench_batch_distances(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_distances");

    let ball = PoincareBall::new(1.0);
    let lorentz = LorentzModel::new(1.0);
    let mut rng = StdRng::seed_from_u64(42);

    let dim = 64;
    let n_points = 100;

    let poincare_points: Vec<Vec<f64>> = (0..n_points)
        .map(|_| random_point_in_ball(dim, 0.9, &mut rng))
        .collect();
    let lorentz_points: Vec<Vec<f64>> = (0..n_points)
        .map(|_| random_lorentz_point(dim, &mut rng))
        .collect();

    group.throughput(Throughput::Elements((n_points * (n_points - 1) / 2) as u64));

    group.bench_function("poincare_all_pairs", |b| {
        b.iter(|| {
            let mut sum = 0.0;
            for i in 0..n_points {
                for j in (i + 1)..n_points {
                    sum += ball.distance(&poincare_points[i], &poincare_points[j]);
                }
            }
            sum
        })
    });

    group.bench_function("lorentz_all_pairs", |b| {
        b.iter(|| {
            let mut sum = 0.0;
            for i in 0..n_points {
                for j in (i + 1)..n_points {
                    sum += lorentz.distance(&lorentz_points[i], &lorentz_points[j]);
                }
            }
            sum
        })
    });

    group.finish();
}

/// Measure numerical precision vs distance from origin.
fn bench_precision_vs_radius(c: &mut Criterion) {
    let mut group = c.benchmark_group("precision_vs_radius");
    group.sample_size(20);

    let ball = PoincareBall::new(1.0);

    eprintln!("\n=== Precision vs Radius (Poincare Ball) ===");
    eprintln!(
        "{:>10} {:>15} {:>15}",
        "radius", "exp_log_error", "dist_self"
    );

    for radius in [0.1, 0.5, 0.9, 0.99, 0.999] {
        let dim = 64;
        let mut rng = StdRng::seed_from_u64(42);

        // Generate point at target radius
        let mut v: Vec<f64> = (0..dim).map(|_| rng.random::<f64>() * 2.0 - 1.0).collect();
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        for x in &mut v {
            *x *= radius / norm;
        }

        // Test exp/log round trip
        let on_manifold = ball.exp_map_zero(&v);
        let recovered = ball.log_map_zero(&on_manifold);

        let error: f64 = v
            .iter()
            .zip(recovered.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            / dim as f64;

        // Test distance to self
        let dist_self = ball.distance(&v, &v);

        eprintln!("{:>10.3} {:>15.2e} {:>15.2e}", radius, error, dist_self);

        group.bench_with_input(
            BenchmarkId::new("exp_log_radius", format!("{:.2}", radius)),
            &radius,
            |b, _| {
                b.iter(|| {
                    let m = ball.exp_map_zero(black_box(&v));
                    ball.log_map_zero(black_box(&m))
                })
            },
        );
    }

    group.finish();
}

/// Compare Poincare vs Lorentz performance.
fn bench_model_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_comparison");

    let ball = PoincareBall::new(1.0);
    let lorentz = LorentzModel::new(1.0);
    let mut rng = StdRng::seed_from_u64(42);

    let dim = 64;
    let x_poincare = random_point_in_ball(dim, 0.9, &mut rng);
    let y_poincare = random_point_in_ball(dim, 0.9, &mut rng);

    let x_lorentz = random_lorentz_point(dim, &mut rng);
    let y_lorentz = random_lorentz_point(dim, &mut rng);

    group.bench_function("poincare_distance_d64", |b| {
        b.iter(|| ball.distance(black_box(&x_poincare), black_box(&y_poincare)))
    });

    group.bench_function("lorentz_distance_d64", |b| {
        b.iter(|| lorentz.distance(black_box(&x_lorentz), black_box(&y_lorentz)))
    });

    // Print actual distances for reference
    let d_poincare = ball.distance(&x_poincare, &y_poincare);
    let d_lorentz = lorentz.distance(&x_lorentz, &y_lorentz);
    eprintln!("\n=== Model Comparison ===");
    eprintln!("Poincare distance: {:.6}", d_poincare);
    eprintln!("Lorentz distance: {:.6}", d_lorentz);

    group.finish();
}

criterion_group!(
    benches,
    bench_poincare_distance,
    bench_lorentz_distance,
    bench_mobius_add,
    bench_poincare_parallel_transport,
    bench_exp_map,
    bench_batch_distances,
    bench_precision_vs_radius,
    bench_model_comparison,
);
criterion_main!(benches);
