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
//!
//! ## Typed points vs. free coordinates
//!
//! Admissibility (`||x|| < 1/sqrt(c)`) is part of the Poincaré metric space, not an
//! optional precondition. [`BallPoint`] carries that invariant in the type: it can
//! only be built through [`BallPoint::new`] (rejects inadmissible input) or
//! [`BallPoint::project`] (clamps exactly like [`PoincareBallCore::project`]).
//!
//! The slice-based methods on [`PoincareBallCore`] accept any `&[T]` and remain the
//! unchecked layer. **`BallPoint` is the surface intended to survive**; the free
//! slice methods are kept for the current additive pass and will be deprecated in a
//! later breaking release, so new code should take `&BallPoint<T>` rather than
//! re-deciding boundary behaviour per call site.

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
    /// Create a Poincare ball with finite, positive curvature `c`.
    pub fn new(c: T) -> Self {
        assert!(
            c.is_finite() && c > T::zero(),
            "curvature must be finite and positive"
        );
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

        // (1/sqrt(c)) arcosh(1 + 2c||x - y||^2 / ((1 - c||x||^2)(1 - c||y||^2))),
        // equal to (2/sqrt(c)) artanh(sqrt(c) ||(-x) ⊕ y||). The artanh form
        // rounds its argument to 1 near the boundary and returns infinity.
        let one = T::one();
        let two = T::from_f64(2.0).unwrap();
        let mut diff_sq = T::zero();
        for i in 0..x.len() {
            let di = x[i] - y[i];
            diff_sq = diff_sq + di * di;
        }
        let delta =
            two * self.c * diff_sq / ((one - self.c * dot(x, x)) * (one - self.c * dot(y, y)));
        // arcosh(1 + delta) = ln(1 + delta + sqrt(delta * (delta + 2))), exact for small delta.
        (delta + (delta * (delta + two)).sqrt()).ln_1p() / self.c.sqrt()
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
        let radius = (one / self.c).sqrt();
        let max_norm = radius * (one - T::epsilon().sqrt());

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

/// Reason a [`BallPoint`] was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum BallPointError {
    /// Curvature was not finite and strictly positive.
    #[error("curvature must be finite and positive")]
    InvalidCurvature,
    /// At least one coordinate was `NaN` or infinite.
    #[error("coordinates must all be finite")]
    NonFiniteCoords,
    /// The point is not strictly inside the ball: `||x|| >= 1/sqrt(c)`.
    #[error("point is not strictly inside the ball: ||x|| >= 1/sqrt(c)")]
    OutsideBall,
}

/// A point strictly inside the Poincaré ball of curvature `c`.
///
/// The admissibility condition `||x|| < 1/sqrt(c)` holds for the whole lifetime of
/// the value: both constructors either reject or clamp, so `distance` and
/// `mobius_add` never see a boundary or out-of-ball argument. Results are
/// bit-identical to the slice API, because the arithmetic is delegated unchanged.
///
/// ```
/// use hyperball::core::BallPoint;
///
/// let x = BallPoint::new(1.0, vec![0.2, -0.1]).unwrap();
/// let y = BallPoint::project(1.0, &[2.0, -3.0]);
/// assert!(y.coords()[0] < 1.0);
/// assert!(x.distance(&x) == 0.0);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct BallPoint<T> {
    c: T,
    coords: Vec<T>,
}

impl<T> BallPoint<T>
where
    T: Float + FromPrimitive,
{
    /// Validate `coords` as a point of the ball with curvature `c`.
    ///
    /// Rejects non-finite curvature, non-finite coordinates, and any point with
    /// `||x|| >= 1/sqrt(c)` (the boundary is *not* part of the space).
    pub fn new(c: T, coords: Vec<T>) -> Result<Self, BallPointError> {
        if !(c.is_finite() && c > T::zero()) {
            return Err(BallPointError::InvalidCurvature);
        }
        if coords.iter().any(|v| !v.is_finite()) {
            return Err(BallPointError::NonFiniteCoords);
        }
        // `dot` sums non-negative products, so `norm_sq` is `>= 0` or `inf`, never NaN.
        if dot(&coords, &coords) >= T::one() / c {
            return Err(BallPointError::OutsideBall);
        }
        Ok(Self { c, coords })
    }

    /// Clamp `x` into the ball exactly as [`PoincareBallCore::project`] does, then
    /// return it as a validated point.
    ///
    /// The reused clamp is `max_norm = sqrt(1/c) * (1 - sqrt(epsilon))`, with
    /// `scale = max_norm / ||x||` applied to every coordinate when
    /// `||x|| > max_norm`; otherwise `x` is copied unchanged.
    ///
    /// # Panics
    ///
    /// Panics if `c` is not finite and positive, or if any coordinate of `x` is
    /// non-finite (`project` cannot clamp `NaN`/`inf` into a valid point).
    pub fn project(c: T, x: &[T]) -> Self {
        let ball = PoincareBallCore::new(c);
        let coords = ball.project(x);
        assert!(
            coords.iter().all(|v| v.is_finite()),
            "cannot project non-finite coordinates into the ball"
        );
        assert!(
            ball.is_in_ball(&coords),
            "projected point is not strictly inside the ball"
        );
        Self { c, coords }
    }

    /// Coordinates of the point.
    pub fn coords(&self) -> &[T] {
        &self.coords
    }

    /// Curvature `c` of the ball this point belongs to.
    pub fn curvature(&self) -> T {
        self.c
    }

    /// Hyperbolic distance, delegated to [`PoincareBallCore::distance`].
    ///
    /// # Panics
    ///
    /// Panics if the two points belong to balls of different curvature.
    pub fn distance(&self, other: &Self) -> T {
        assert!(
            self.c == other.c,
            "points must belong to balls of the same curvature"
        );
        self.ball().distance(&self.coords, &other.coords)
    }

    /// Möbius addition, delegated to [`PoincareBallCore::mobius_add`].
    ///
    /// # Panics
    ///
    /// Panics if the two points belong to balls of different curvature.
    pub fn mobius_add(&self, other: &Self) -> Vec<T> {
        assert!(
            self.c == other.c,
            "points must belong to balls of the same curvature"
        );
        self.ball().mobius_add(&self.coords, &other.coords)
    }

    fn ball(&self) -> PoincareBallCore<T> {
        PoincareBallCore::new(self.c)
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
        let scale = two / denom;

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
    fn poincare_core_rejects_invalid_curvature() {
        for c in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(std::panic::catch_unwind(|| PoincareBallCore::new(c)).is_err());
        }
    }

    #[test]
    fn poincare_core_projection_is_scale_relative() {
        for c in [1e-12, 1e-6, 1.0, 1e6, 1e12] {
            let ball = PoincareBallCore::new(c);
            let x = [2.0 / c.sqrt(), -3.0 / c.sqrt()];
            let projected = ball.project(&x);

            assert!(projected.iter().all(|value| value.is_finite()));
            assert!(ball.is_in_ball(&projected), "c={c}, p={projected:?}");

            let scale = projected[0] / x[0];
            assert!(scale >= 0.0, "c={c}, scale={scale}");
            assert!(approx_eq(projected[1], x[1] * scale, 1e-12 / c.sqrt()));
            assert_eq!(ball.project(&[0.0, 0.0]), [0.0, 0.0]);
        }

        for c in [1e-12_f32, 1e-6, 1.0, 1e6, 1e12] {
            let ball = PoincareBallCore::new(c);
            let x = [2.0 / c.sqrt(), -3.0 / c.sqrt()];
            let projected = ball.project(&x);

            assert!(projected.iter().all(|value| value.is_finite()));
            assert!(ball.is_in_ball(&projected), "c={c}, p={projected:?}");

            let scale = projected[0] / x[0];
            assert!(scale >= 0.0, "c={c}, scale={scale}");
            assert!((projected[1] - x[1] * scale).abs() <= 1e-5 / c.sqrt());
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
    fn conversion_core_lands_on_hyperboloid_for_any_curvature() {
        // -t^2 + |s|^2 = -1/c must hold for every c, not only c = 1.
        for c in [0.1f64, 0.5, 2.0, 4.0] {
            let ball = PoincareBallCore::new(c);
            let lorentz = LorentzModelCore::new(c);
            let x = [0.2 / c.sqrt(), 0.1 / c.sqrt(), -0.05 / c.sqrt()];
            assert!(ball.is_in_ball(&x));

            let xl = conversions::poincare_to_lorentz(&ball, &x);
            assert!(lorentz.is_on_manifold(&xl, 1e-10), "c={c}: {xl:?}");
            let x2 = conversions::lorentz_to_poincare(&lorentz, &xl);
            for i in 0..x.len() {
                assert!(approx_eq(x2[i], x[i], 1e-10), "c={c}: {x2:?} != {x:?}");
            }
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

    #[test]
    fn ball_point_project_lands_strictly_inside() {
        for c in [1e-6f64, 0.25, 1.0, 4.0, 1e6] {
            let ball = PoincareBallCore::new(c);
            let radius = (1.0 / c).sqrt();
            // Out of the ball, and exactly at the boundary.
            for x in [
                vec![2.0 / c.sqrt(), -3.0 / c.sqrt()],
                vec![radius, 0.0],
                vec![0.0, -radius, radius],
            ] {
                let p = BallPoint::project(c, &x);
                let norm = dot(p.coords(), p.coords()).sqrt();

                assert!(norm < radius, "c={c}: norm={norm} radius={radius}");
                assert!(norm > 0.0, "c={c}: projection collapsed to the origin");
                assert!(ball.is_in_ball(p.coords()), "c={c}");
                assert_eq!(p.curvature(), c);
                // Reuses the existing clamp exactly, so results are bit-identical.
                let clamped = ball.project(&x);
                assert_eq!(p.coords().len(), clamped.len());
                for (a, b) in p.coords().iter().zip(clamped.iter()) {
                    assert_eq!(a.to_bits(), b.to_bits(), "c={c}: {a} != {b}");
                }
            }

            // Already-admissible input is copied, not rescaled.
            let inside = vec![0.1 / c.sqrt(), -0.2 / c.sqrt()];
            let p = BallPoint::project(c, &inside);
            for (a, b) in p.coords().iter().zip(inside.iter()) {
                assert_eq!(a.to_bits(), b.to_bits(), "c={c}");
            }
        }
    }

    #[test]
    fn ball_point_rejects_boundary_point() {
        for c in [0.5f64, 1.0, 2.0] {
            let radius = (1.0 / c).sqrt();
            assert_eq!(
                BallPoint::new(c, vec![radius, 0.0]),
                Err(BallPointError::OutsideBall)
            );
            assert_eq!(
                BallPoint::new(c, vec![0.0, 0.0, radius * 1.000_000_1]),
                Err(BallPointError::OutsideBall)
            );
            // Just inside is accepted.
            let ok = BallPoint::new(c, vec![radius * 0.999, 0.0]);
            assert!(ok.is_ok(), "c={c}: {ok:?}");
        }
    }

    #[test]
    fn ball_point_rejects_non_finite_coords() {
        for coords in [
            vec![f64::NAN, 0.0],
            vec![0.0, f64::INFINITY],
            vec![f64::NEG_INFINITY, 0.0],
        ] {
            assert_eq!(
                BallPoint::new(1.0, coords),
                Err(BallPointError::NonFiniteCoords)
            );
        }
        assert_eq!(
            BallPoint::new(f64::NAN, vec![0.0, 0.0]),
            Err(BallPointError::InvalidCurvature)
        );
    }

    #[test]
    fn ball_point_distance_matches_slice_api_bit_for_bit() {
        for c in [0.25f64, 1.0, 4.0] {
            let ball = PoincareBallCore::new(c);
            let scale = 1.0 / c.sqrt();
            // Grid of admissible base vectors in the unit ball.
            let mut raw: Vec<Vec<f64>> = Vec::new();
            for a in [-0.8, -0.3, 0.0, 0.3, 0.8] {
                for b in [-0.4, 0.0, 0.4] {
                    raw.push(vec![a * scale, b * scale]);
                }
            }
            let points: Vec<BallPoint<f64>> = raw
                .iter()
                .map(|x| BallPoint::new(c, x.clone()).expect("grid point is admissible"))
                .collect();

            for (i, x) in raw.iter().enumerate() {
                for (j, y) in raw.iter().enumerate() {
                    let typed = points[i].distance(&points[j]);
                    let slice = ball.distance(x, y);
                    assert_eq!(typed.to_bits(), slice.to_bits(), "c={c} i={i} j={j}");

                    let typed_add = points[i].mobius_add(&points[j]);
                    let slice_add = ball.mobius_add(x, y);
                    for (k, (a, b)) in typed_add.iter().zip(slice_add.iter()).enumerate() {
                        assert_eq!(a.to_bits(), b.to_bits(), "c={c} i={i} j={j} k={k}");
                    }
                }
            }
        }
    }

    #[test]
    fn ball_point_distance_is_symmetric_and_zero_on_self() {
        let c = 2.0f64;
        let scale = 1.0 / c.sqrt();
        let points: Vec<BallPoint<f64>> = [
            vec![0.0, 0.0],
            vec![0.1 * scale, -0.2 * scale],
            vec![-0.7 * scale, 0.3 * scale],
            vec![0.999_999 * scale, 0.0],
        ]
        .iter()
        .map(|x| BallPoint::new(c, x.clone()).unwrap())
        .collect();

        for (i, p) in points.iter().enumerate() {
            assert_eq!(p.distance(p), 0.0, "d(x,x) must be exactly zero at i={i}");
            for (j, q) in points.iter().enumerate() {
                assert_eq!(p.distance(q).to_bits(), q.distance(p).to_bits(), "{i},{j}");
                assert!(p.distance(q) >= 0.0);
            }
        }
    }

    #[test]
    fn ball_point_mobius_add_matches_slice_api_bit_for_bit() {
        let c = 1.0f64;
        let ball = PoincareBallCore::new(c);
        let x = [0.2, -0.1, 0.05];
        let zero = [0.0, 0.0, 0.0];
        let px = BallPoint::new(c, x.to_vec()).unwrap();
        let pz = BallPoint::new(c, zero.to_vec()).unwrap();

        let typed = px.mobius_add(&pz);
        let slice = ball.mobius_add(&x, &zero);
        for (a, b) in typed.iter().zip(slice.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        assert_eq!(px.coords(), &x);
    }
}
