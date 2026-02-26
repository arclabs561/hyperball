//! Backend-agnostic hyperbolic primitives.
//!
//! This module deliberately avoids committing to a tensor/array backend.
//! Everything operates on slices and returns `Vec<T>`.
//!
//! The ndarray-based API remains available behind the `ndarray` feature.
//!
//! ## References (for rationale and cross-checking formulas)
//!
//! - Nickel & Kiela (2017): *Poincaré Embeddings for Learning Hierarchical Representations*.
//! - Ganea, Bécigneul, Hofmann (2018): *Hyperbolic Neural Networks* (Poincaré + Lorentz tooling).
//! - Gromov (1987): *Hyperbolic groups* (δ-hyperbolicity; four-point conditions).
//!
//! Notes:
//! - The `diagnostics` submodule is intentionally “small n only” (\(O(n^4)\) exact δ) and exists
//!   to validate whether a dataset is plausibly tree-like before committing to heavier machinery.

use num_traits::{Float, FromPrimitive};

/// Poincaré ball operations on slice inputs.
#[derive(Debug, Clone, Copy)]
pub struct PoincareBallCore<T> {
    /// Curvature parameter (c > 0)
    pub c: T,
}

impl<T> PoincareBallCore<T>
where
    T: Float + FromPrimitive,
{
    /// Create a Poincare ball with curvature `c` (must be positive).
    pub fn new(c: T) -> Self {
        assert!(c > T::zero(), "curvature must be positive");
        Self { c }
    }

    /// Möbius addition on the Poincaré ball.
    ///
    /// Returns a newly-allocated vector.
    pub fn mobius_add(&self, x: &[T], y: &[T]) -> Vec<T> {
        assert_eq!(x.len(), y.len());
        let c = self.c;
        let one = T::one();
        let two = T::from_f64(2.0).unwrap();

        let x_norm_sq = dot(x, x);
        let y_norm_sq = dot(y, y);
        let xy = dot(x, y);

        let denom = one + two * c * xy + c * c * x_norm_sq * y_norm_sq;
        let s1 = one + two * c * xy + c * y_norm_sq;
        let s2 = one - c * x_norm_sq;

        let mut out = vec![T::zero(); x.len()];
        for i in 0..x.len() {
            out[i] = (x[i] * s1 + y[i] * s2) / denom;
        }
        out
    }

    /// Hyperbolic distance on the Poincaré ball.
    pub fn distance(&self, x: &[T], y: &[T]) -> T {
        assert_eq!(x.len(), y.len());

        // diff = (-x) ⊕ y
        let mut neg_x = vec![T::zero(); x.len()];
        for i in 0..x.len() {
            neg_x[i] = -x[i];
        }
        let diff = self.mobius_add(&neg_x, y);
        let diff_norm = dot(&diff, &diff).sqrt();
        let c_sqrt = self.c.sqrt();
        let two = T::from_f64(2.0).unwrap();
        two / c_sqrt * (c_sqrt * diff_norm).atanh()
    }

    /// Log map at origin.
    pub fn log_map_zero(&self, y: &[T]) -> Vec<T> {
        let y_norm = dot(y, y).sqrt();
        let eps = T::from_f64(1e-7).unwrap();
        if y_norm < eps {
            return y.to_vec();
        }
        let c_sqrt = self.c.sqrt();
        let scale = (c_sqrt * y_norm).atanh() / (c_sqrt * y_norm);
        let mut out = vec![T::zero(); y.len()];
        for i in 0..y.len() {
            out[i] = y[i] * scale;
        }
        out
    }

    /// Exp map at origin.
    pub fn exp_map_zero(&self, v: &[T]) -> Vec<T> {
        let v_norm = dot(v, v).sqrt();
        let eps = T::from_f64(1e-7).unwrap();
        if v_norm < eps {
            return v.to_vec();
        }
        let c_sqrt = self.c.sqrt();
        let scale = (c_sqrt * v_norm).tanh() / (c_sqrt * v_norm);
        let mut out = vec![T::zero(); v.len()];
        for i in 0..v.len() {
            out[i] = v[i] * scale;
        }
        out
    }

    /// Check if point lies inside ball (||x|| < 1/sqrt(c)).
    pub fn is_in_ball(&self, x: &[T]) -> bool {
        let norm_sq = dot(x, x);
        norm_sq < T::one() / self.c
    }

    /// Project onto ball boundary if outside.
    pub fn project(&self, x: &[T]) -> Vec<T> {
        let norm = dot(x, x).sqrt();
        let one = T::one();
        let eps = T::from_f64(1e-5).unwrap();
        let max_norm = (one / self.c).sqrt() - eps;

        if norm > max_norm {
            let scale = max_norm / norm;
            let mut out = vec![T::zero(); x.len()];
            for i in 0..x.len() {
                out[i] = x[i] * scale;
            }
            out
        } else {
            x.to_vec()
        }
    }
}

fn dot<T: Float>(a: &[T], b: &[T]) -> T {
    assert_eq!(a.len(), b.len());
    let mut acc = T::zero();
    for i in 0..a.len() {
        acc = acc + a[i] * b[i];
    }
    acc
}

/// Lorentz (hyperboloid) model operations on slice inputs.
///
/// Points are represented as (n+1)-dimensional vectors:
/// - index 0: time coordinate (t > 0)
/// - indices 1..: space coordinates
#[derive(Debug, Clone, Copy)]
pub struct LorentzModelCore<T> {
    /// Curvature parameter (c > 0). The hyperboloid satisfies <x,x>_L = -1/c.
    pub c: T,
}

impl<T> LorentzModelCore<T>
where
    T: Float + FromPrimitive,
{
    /// Create a Lorentz (hyperboloid) model with curvature `c` (must be positive).
    pub fn new(c: T) -> Self {
        assert!(c > T::zero(), "curvature must be positive");
        Self { c }
    }

    /// Minkowski inner product: <x,y>_L = -x_0*y_0 + sum_{i>=1} x_i*y_i
    pub fn minkowski_dot(&self, x: &[T], y: &[T]) -> T {
        assert_eq!(x.len(), y.len());
        assert!(x.len() >= 2);
        let mut acc = -x[0] * y[0];
        for i in 1..x.len() {
            acc = acc + x[i] * y[i];
        }
        acc
    }

    /// Lorentzian distance: d(x,y) = (1/sqrt(c)) * arcosh(-c * <x,y>_L)
    pub fn distance(&self, x: &[T], y: &[T]) -> T {
        let inner = self.minkowski_dot(x, y);
        let arg = -self.c * inner;
        let one = T::one();
        let two = T::from_f64(2.0).unwrap();
        let eps = T::from_f64(1e-7).unwrap();

        // Clamp for numerical stability (should be >= 1).
        // When arg is very close to 1, use Taylor: acosh(1+x) ≈ sqrt(2x).
        if arg < one + eps {
            if arg <= one {
                return T::zero();
            }
            let x = arg - one;
            return (two * x).sqrt() / self.c.sqrt();
        }
        arg.acosh() / self.c.sqrt()
    }

    /// Check if point is on the hyperboloid: <x,x>_L ≈ -1/c.
    pub fn is_on_manifold(&self, x: &[T], tol: T) -> bool {
        let inner = self.minkowski_dot(x, x);
        (inner + T::one() / self.c).abs() < tol
    }

    /// Project point onto hyperboloid by fixing the time component.
    pub fn project(&self, x: &[T]) -> Vec<T> {
        assert!(x.len() >= 2);
        let mut space_norm_sq = T::zero();
        for &xi in x.iter().skip(1) {
            space_norm_sq = space_norm_sq + xi * xi;
        }
        let t = (space_norm_sq + T::one() / self.c).sqrt();
        let mut out = x.to_vec();
        out[0] = t;
        out
    }

    /// Map Euclidean space vector (n-dim) to hyperboloid point (n+1-dim).
    pub fn from_euclidean(&self, v: &[T]) -> Vec<T> {
        let mut space_norm_sq = T::zero();
        for &val in v {
            space_norm_sq = space_norm_sq + val * val;
        }
        let t = (space_norm_sq + T::one() / self.c).sqrt();
        let mut out = vec![T::zero(); v.len() + 1];
        out[0] = t;
        out[1..].copy_from_slice(v);
        out
    }

    /// Extract Euclidean space coordinates (drop time component).
    pub fn to_euclidean(&self, x: &[T]) -> Vec<T> {
        assert!(x.len() >= 2);
        x[1..].to_vec()
    }

    /// Exponential map at point x: exp_x(v) = cosh(s)||x|| + sinh(s) v / s, with s = sqrt(c) * ||v||_L.
    pub fn exp_map(&self, x: &[T], v: &[T]) -> Vec<T> {
        assert_eq!(x.len(), v.len());
        let v_norm_sq = self.minkowski_dot(v, v);
        let eps = T::from_f64(1e-15).unwrap();
        if v_norm_sq < eps {
            return x.to_vec();
        }
        let v_norm = v_norm_sq.sqrt();
        let c_sqrt = self.c.sqrt();
        let s = c_sqrt * v_norm;

        let cosh_term = s.cosh();
        let sinh_term = s.sinh() / s;

        let mut out = vec![T::zero(); x.len()];
        for i in 0..x.len() {
            out[i] = x[i] * cosh_term + v[i] * sinh_term;
        }
        out
    }

    /// Log map at point x.
    pub fn log_map(&self, x: &[T], y: &[T]) -> Vec<T> {
        assert_eq!(x.len(), y.len());
        let inner = self.minkowski_dot(x, y);
        let d = self.distance(x, y);
        let eps = T::from_f64(1e-15).unwrap();
        if d < eps {
            return vec![T::zero(); x.len()];
        }

        // v = y + c * inner * x (inner is negative).
        let mut v = vec![T::zero(); x.len()];
        for i in 0..x.len() {
            v[i] = y[i] + x[i] * (self.c * inner);
        }

        let v_norm_sq = self.minkowski_dot(&v, &v);
        if v_norm_sq < eps {
            return vec![T::zero(); x.len()];
        }
        let v_norm = v_norm_sq.sqrt();
        let scale = d / v_norm;
        for vi in &mut v {
            *vi = *vi * scale;
        }
        v
    }

    /// Origin: (1/sqrt(c), 0, ..., 0).
    pub fn origin(&self, space_dim: usize) -> Vec<T> {
        let mut out = vec![T::zero(); space_dim + 1];
        out[0] = T::one() / self.c.sqrt();
        out
    }
}

/// Core conversions between Poincaré and Lorentz models.
pub mod conversions {
    use super::{Float, FromPrimitive, LorentzModelCore, PoincareBallCore};

    /// Map a point from the Poincare ball to the Lorentz (hyperboloid) model.
    pub fn poincare_to_lorentz<T>(ball: &PoincareBallCore<T>, x: &[T]) -> Vec<T>
    where
        T: Float + FromPrimitive,
    {
        let c = ball.c;
        let one = T::one();
        let two = T::from_f64(2.0).unwrap();

        let mut x_norm_sq = T::zero();
        for &v in x {
            x_norm_sq = x_norm_sq + v * v;
        }

        let denom = one - c * x_norm_sq;
        let c_sqrt = c.sqrt();

        let t = (one + c * x_norm_sq) / (denom * c_sqrt);
        let scale = two / (denom * c_sqrt);

        let mut out = vec![T::zero(); x.len() + 1];
        out[0] = t;
        for i in 0..x.len() {
            out[i + 1] = x[i] * scale;
        }
        out
    }

    /// Map a point from the Lorentz (hyperboloid) model to the Poincare ball.
    pub fn lorentz_to_poincare<T>(lorentz: &LorentzModelCore<T>, x: &[T]) -> Vec<T>
    where
        T: Float + FromPrimitive,
    {
        assert!(x.len() >= 2);
        let c_sqrt = lorentz.c.sqrt();
        let one = T::one();
        let denom = x[0] * c_sqrt + one;

        let mut out = vec![T::zero(); x.len() - 1];
        for i in 1..x.len() {
            out[i - 1] = x[i] / denom;
        }
        out
    }
}

/// Diagnostics for “tree-likeness”.
///
/// These utilities are intentionally backend-agnostic and operate on a distance
/// matrix (flattened row-major).
pub mod diagnostics {
    /// Max violation of the ultrametric inequality over all triples.
    ///
    /// Ultrametric inequality:
    /// \[
    /// d(i,k) \le \max(d(i,j), d(j,k)).
    /// \]
    ///
    /// Returns:
    /// \[
    /// \max_{i,j,k}\; \max(0,\; d(i,k) - \max(d(i,j), d(j,k))).
    /// \]
    ///
    /// - `dist` is a row-major \(n \times n\) matrix.
    /// - Assumes `dist[i*n + i] == 0` and `dist` is symmetric. (Not enforced.)
    pub fn ultrametric_max_violation_f64(dist: &[f64], n: usize) -> f64 {
        assert_eq!(dist.len(), n * n);
        let mut max_v = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let dij = dist[i * n + j];
                for k in 0..n {
                    if k == i || k == j {
                        continue;
                    }
                    let dik = dist[i * n + k];
                    let djk = dist[j * n + k];
                    let rhs = dij.max(djk);
                    let v = dik - rhs;
                    if v > max_v {
                        max_v = v;
                    }
                }
            }
        }
        if max_v.is_sign_negative() {
            0.0
        } else {
            max_v
        }
    }

    /// Gromov 4-point δ-hyperbolicity (exact over all quadruples).
    ///
    /// For points \(a,b,c,d\), compute:
    /// \[
    /// s_1 = d(a,b)+d(c,d),\;
    /// s_2 = d(a,c)+d(b,d),\;
    /// s_3 = d(a,d)+d(b,c).
    /// \]
    /// Let \(m_1 \ge m_2 \ge m_3\) be the sorted values. The 4-point δ is:
    /// \[
    /// \delta(a,b,c,d) = \frac{m_1 - m_2}{2}.
    /// \]
    ///
    /// Returns \(\max_{a,b,c,d} \delta(a,b,c,d)\).
    ///
    /// This is \(O(n^4)\) and only intended for small \(n\).
    pub fn delta_hyperbolicity_four_point_exact_f64(dist: &[f64], n: usize) -> f64 {
        assert_eq!(dist.len(), n * n);
        let mut max_delta = 0.0f64;

        for a in 0..n {
            for b in 0..n {
                if b == a {
                    continue;
                }
                for c in 0..n {
                    if c == a || c == b {
                        continue;
                    }
                    for d in 0..n {
                        if d == a || d == b || d == c {
                            continue;
                        }
                        let s1 = dist[a * n + b] + dist[c * n + d];
                        let s2 = dist[a * n + c] + dist[b * n + d];
                        let s3 = dist[a * n + d] + dist[b * n + c];
                        let mut m1 = s1;
                        let mut m2 = s2;
                        let mut m3 = s3;

                        // Sort 3 values descending (branchy but small).
                        if m1 < m2 {
                            std::mem::swap(&mut m1, &mut m2);
                        }
                        if m2 < m3 {
                            std::mem::swap(&mut m2, &mut m3);
                        }
                        if m1 < m2 {
                            std::mem::swap(&mut m1, &mut m2);
                        }

                        let delta = 0.5 * (m1 - m2);
                        if delta > max_delta {
                            max_delta = delta;
                        }
                    }
                }
            }
        }
        max_delta
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn poincare_core_mobius_identity() {
        let ball = PoincareBallCore::new(1.0f64);
        let x = [0.2, -0.1, 0.05];
        let zero = [0.0, 0.0, 0.0];
        let r = ball.mobius_add(&x, &zero);
        for i in 0..x.len() {
            assert!(approx_eq(r[i], x[i], 1e-12));
        }
    }

    #[test]
    fn poincare_core_distance_self_zero() {
        let ball = PoincareBallCore::new(1.0f64);
        let x = [0.3, 0.1];
        let d = ball.distance(&x, &x);
        assert!(d.abs() < 1e-12);
    }

    #[test]
    fn poincare_core_exp_log_round_trip_small() {
        let ball = PoincareBallCore::new(1.0f64);
        let v = [0.3, -0.2];
        let y = ball.exp_map_zero(&v);
        let v2 = ball.log_map_zero(&y);
        for i in 0..v.len() {
            assert!(approx_eq(v2[i], v[i], 1e-7));
        }
    }

    #[test]
    fn lorentz_core_origin_is_on_manifold() {
        let lorentz = LorentzModelCore::new(1.0f64);
        let o = lorentz.origin(3);
        assert!(lorentz.is_on_manifold(&o, 1e-12));
    }

    #[test]
    fn conversion_core_round_trip_poincare() {
        let ball = PoincareBallCore::new(1.0f64);
        let lorentz = LorentzModelCore::new(1.0f64);
        let x = [0.2, 0.1, -0.05];
        assert!(ball.is_in_ball(&x));

        let xl = conversions::poincare_to_lorentz(&ball, &x);
        assert!(lorentz.is_on_manifold(&xl, 1e-10));
        let x2 = conversions::lorentz_to_poincare(&lorentz, &xl);
        for i in 0..x.len() {
            assert!(approx_eq(x2[i], x[i], 1e-10));
        }
    }

    #[test]
    fn ultrametric_violation_zero_for_simple_ultrametric() {
        // Leaves: (0,1) close; (2,3) close; across clusters far.
        // Distances: within cluster = 1, across = 2. This is ultrametric.
        let n = 4usize;
        let mut d = vec![0.0f64; n * n];
        let set = |d: &mut [f64], i: usize, j: usize, v: f64| {
            d[i * n + j] = v;
            d[j * n + i] = v;
        };
        set(&mut d, 0, 1, 1.0);
        set(&mut d, 2, 3, 1.0);
        for (i, j) in [(0, 2), (0, 3), (1, 2), (1, 3)] {
            set(&mut d, i, j, 2.0);
        }
        let v = diagnostics::ultrametric_max_violation_f64(&d, n);
        assert!(v.abs() < 1e-12, "expected 0, got {v}");
    }

    #[test]
    fn ultrametric_violation_positive_for_non_ultrametric() {
        // 0-1=1, 1-2=1, 0-2=3 violates ultrametric inequality:
        // 3 <= max(1,1) is false; violation = 2.
        let n = 3usize;
        let d = vec![
            0.0, 1.0, 3.0, //
            1.0, 0.0, 1.0, //
            3.0, 1.0, 0.0, //
        ];
        let v = diagnostics::ultrametric_max_violation_f64(&d, n);
        assert!((v - 2.0).abs() < 1e-12, "expected 2, got {v}");
    }

    #[test]
    fn delta_hyperbolicity_c4_is_one() {
        // Cycle C4 shortest-path distances:
        // adjacent: 1, opposite: 2.
        let n = 4usize;
        let d = vec![
            0.0, 1.0, 2.0, 1.0, //
            1.0, 0.0, 1.0, 2.0, //
            2.0, 1.0, 0.0, 1.0, //
            1.0, 2.0, 1.0, 0.0, //
        ];
        let delta = diagnostics::delta_hyperbolicity_four_point_exact_f64(&d, n);
        assert!((delta - 1.0).abs() < 1e-12, "expected 1, got {delta}");
    }
}
