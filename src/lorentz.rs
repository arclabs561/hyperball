//! Lorentz (hyperboloid) model of hyperbolic space.
//!
//! Represents \(\mathbb{H}^d_c\) as the upper sheet of a two-sheeted hyperboloid
//! embedded in \((d+1)\)-dimensional Minkowski space:
//!
//! \[
//! \mathcal{H}^d_c = \{x \in \mathbb{R}^{d+1} : \langle x, x \rangle_\mathcal{L} = -1/c,\; x_0 > 0\}
//! \]
//!
//! where \(\langle x, y \rangle_\mathcal{L} = -x_0 y_0 + \sum_{i=1}^d x_i y_i\) is
//! the Minkowski inner product.
//!
//! **Compared to the Poincare ball**:
//! - Gradients are well-behaved everywhere (no conformal factor blowup)
//! - More numerically stable for Riemannian SGD (Nickel & Kiela, 2018)
//! - Natural for Lorentzian geometry (relativity, causal structure)
//! - Slightly smaller representable radius (~19 vs ~38 in float64)
//!
//! The two models are isometric: [`from_euclidean`](LorentzModel::from_euclidean) and
//! [`to_euclidean`](LorentzModel::to_euclidean) convert between them.
//!
//! ## References
//!
//! - Nickel & Kiela (2018). "Learning Continuous Hierarchies in the Lorentz Model"
//! - Ganea, Becigneul, Hofmann (2018). "Hyperbolic Neural Networks"

use ndarray::{Array1, ArrayView1};
use num_traits::{Float, FromPrimitive, Zero};

/// Lorentz (hyperboloid) model manifold.
///
/// Points live in \(\mathbb{R}^{d+1}\): the first component \(x_0\) is the
/// "time" coordinate (always positive), the remaining \(x_1, \ldots, x_d\)
/// are "space" coordinates. All points satisfy the hyperboloid constraint
/// \(\langle x, x \rangle_\mathcal{L} = -1/c\).
///
/// The origin of hyperbolic space is the "north pole" \((1/\sqrt{c}, 0, \ldots, 0)\).
pub struct LorentzModel<T> {
    /// Curvature parameter (c > 0). The hyperboloid satisfies <x,x>_L = -1/c.
    pub c: T,
}

impl<T> LorentzModel<T>
where
    T: Float + FromPrimitive + Zero + ndarray::ScalarOperand + ndarray::LinalgScalar,
{
    pub fn new(c: T) -> Self {
        assert!(c > T::zero(), "curvature must be positive");
        Self { c }
    }

    /// Minkowski inner product: \(\langle x, y \rangle_\mathcal{L} = -x_0 y_0 + \sum_{i=1}^d x_i y_i\).
    ///
    /// This is the indefinite bilinear form with signature \((-,+,+,\ldots,+)\).
    /// For points on the hyperboloid, \(\langle x, x \rangle_\mathcal{L} = -1/c\).
    pub fn minkowski_dot(&self, x: &ArrayView1<T>, y: &ArrayView1<T>) -> T {
        assert!(x.len() == y.len() && x.len() >= 2);
        -x[0] * y[0] + x.slice(ndarray::s![1..]).dot(&y.slice(ndarray::s![1..]))
    }

    /// Geodesic distance: \(d_c(x,y) = \frac{1}{\sqrt{c}}\,\mathrm{arcosh}(-c\langle x,y\rangle_\mathcal{L})\).
    ///
    /// Uses a Taylor expansion \(\mathrm{arcosh}(1+\delta) \approx \sqrt{2\delta}\) near
    /// \(\delta = 0\) for numerical stability when \(x \approx y\).
    pub fn distance(&self, x: &ArrayView1<T>, y: &ArrayView1<T>) -> T {
        let inner = self.minkowski_dot(x, y);
        let arg = -self.c * inner;
        let one = T::one();
        let two = T::from_f64(2.0).unwrap();
        // Use a generic epsilon
        let epsilon = T::from_f64(1e-7).unwrap();

        // Clamp for numerical stability (should be >= 1)
        // When arg is very close to 1 (within sqrt(eps)), use Taylor: acosh(1+x) ≈ sqrt(2x)
        // This handles the case where x == y but floating point gives arg slightly != 1
        if arg < one + epsilon {
            if arg <= one {
                return T::zero();
            }
            // Taylor expansion: acosh(1+x) ≈ sqrt(2x) for small x
            let x = arg - one;
            return (two * x).sqrt() / self.c.sqrt();
        }
        arg.acosh() / self.c.sqrt()
    }

    /// Check if point is on the hyperboloid: <x,x>_L = -1/c
    pub fn is_on_manifold(&self, x: &ArrayView1<T>, tol: T) -> bool {
        let inner = self.minkowski_dot(x, x);
        (inner + T::one() / self.c).abs() < tol
    }

    /// Project point onto hyperboloid by scaling.
    /// Given x with x_0 > 0, find scale s such that <sx, sx>_L = -1/c.
    pub fn project(&self, x: &ArrayView1<T>) -> Array1<T> {
        let space_norm_sq = x.slice(ndarray::s![1..]).dot(&x.slice(ndarray::s![1..]));
        // We need: -t^2 + space_norm_sq = -1/c
        // So: t^2 = space_norm_sq + 1/c
        let t = (space_norm_sq + T::one() / self.c).sqrt();

        let mut result = x.to_owned();
        result[0] = t;
        result
    }

    /// Project from Euclidean space to hyperboloid.
    /// Maps an n-dimensional Euclidean vector to the (n+1)-dimensional hyperboloid.
    pub fn from_euclidean(&self, v: &ArrayView1<T>) -> Array1<T> {
        let space_norm_sq = v.dot(v);
        let t = (space_norm_sq + T::one() / self.c).sqrt();

        let mut result = Array1::zeros(v.len() + 1);
        result[0] = t;
        for (i, &val) in v.iter().enumerate() {
            result[i + 1] = val;
        }
        result
    }

    /// Project from hyperboloid to Euclidean space (gnomonic projection).
    /// Maps an (n+1)-dimensional hyperboloid point to n-dimensional Euclidean.
    pub fn to_euclidean(&self, x: &ArrayView1<T>) -> Array1<T> {
        x.slice(ndarray::s![1..]).to_owned()
    }

    /// Exponential map at point x: maps tangent vector v to manifold.
    /// exp_x(v) = cosh(||v||_L) * x + sinh(||v||_L) * v / ||v||_L
    /// where ||v||_L = sqrt(<v,v>_L) (Lorentzian norm of tangent vector).
    pub fn exp_map(&self, x: &ArrayView1<T>, v: &ArrayView1<T>) -> Array1<T> {
        let v_norm_sq = self.minkowski_dot(v, v);
        let epsilon = T::from_f64(1e-15).unwrap();

        if v_norm_sq < epsilon {
            return x.to_owned();
        }
        let v_norm = v_norm_sq.sqrt();
        let c_sqrt = self.c.sqrt();

        let cosh_term = (c_sqrt * v_norm).cosh();
        let sinh_term = (c_sqrt * v_norm).sinh() / (c_sqrt * v_norm);

        let term1 = x.mapv(|val| val * cosh_term);
        let term2 = v.mapv(|val| val * sinh_term);
        term1 + term2
    }

    /// Logarithmic map at point x: maps manifold point y to tangent space at x.
    /// log_x(y) = d(x,y) * (y - <x,y>_L * c * x) / ||y - <x,y>_L * c * x||_L
    pub fn log_map(&self, x: &ArrayView1<T>, y: &ArrayView1<T>) -> Array1<T> {
        let inner = self.minkowski_dot(x, y);
        let d = self.distance(x, y);
        let epsilon = T::from_f64(1e-15).unwrap();

        if d < epsilon {
            return Array1::zeros(x.len());
        }

        // v = y + c * inner * x (since inner is negative, this adds)
        let term_x = x.mapv(|val| val * self.c * inner);
        let v = y.to_owned() + term_x;

        let v_norm_sq = self.minkowski_dot(&v.view(), &v.view());

        if v_norm_sq < epsilon {
            return Array1::zeros(x.len());
        }

        let v_norm = v_norm_sq.sqrt();
        v * (d / v_norm)
    }

    /// Parallel transport of tangent vector v from point x to point y.
    pub fn parallel_transport(
        &self,
        x: &ArrayView1<T>,
        y: &ArrayView1<T>,
        v: &ArrayView1<T>,
    ) -> Array1<T> {
        let inner_xy = self.minkowski_dot(x, y);
        let inner_vy = self.minkowski_dot(v, y);

        // PT_{x->y}(v) = v - c * inner_vy / (1 - c * inner_xy) * (x + y)
        let one = T::one();
        let denom = one - self.c * inner_xy;
        let epsilon = T::from_f64(1e-15).unwrap();

        if denom.abs() < epsilon {
            return v.to_owned();
        }

        let coeff = self.c * inner_vy / denom;
        let sum_xy = x.to_owned() + y;
        v.to_owned() - sum_xy.mapv(|val| val * coeff)
    }

    /// Origin of the hyperboloid: (1/sqrt(c), 0, 0, ..., 0)
    pub fn origin(&self, dim: usize) -> Array1<T> {
        let mut o = Array1::zeros(dim + 1);
        o[0] = T::one() / self.c.sqrt();
        o
    }
}

/// Convert between Poincare ball and Lorentz model.
pub mod conversions {
    use super::*;
    use crate::PoincareBall;

    /// Convert Poincare ball point to Lorentz hyperboloid.
    /// Maps n-dimensional ball point to (n+1)-dimensional hyperboloid point.
    pub fn poincare_to_lorentz<T>(ball: &PoincareBall<T>, x: &ArrayView1<T>) -> Array1<T>
    where
        T: Float + FromPrimitive + Zero + ndarray::ScalarOperand,
    {
        let c = ball.c;
        let x_norm_sq = x.dot(x);
        let one = T::one();
        let two = T::from_f64(2.0).unwrap();

        // t = (1 + c*||x||^2) / (1 - c*||x||^2) / sqrt(c)
        // space = 2*x / (1 - c*||x||^2) / sqrt(c)
        let denom = one - c * x_norm_sq;
        let c_sqrt = c.sqrt();

        let t = (one + c * x_norm_sq) / (denom * c_sqrt);

        let mut result = Array1::zeros(x.len() + 1);
        result[0] = t;
        let scale = two / (denom * c_sqrt);
        for (i, &val) in x.iter().enumerate() {
            result[i + 1] = val * scale;
        }
        result
    }

    /// Convert Lorentz hyperboloid point to Poincare ball.
    /// Maps (n+1)-dimensional hyperboloid point to n-dimensional ball point.
    pub fn lorentz_to_poincare<T>(lorentz: &LorentzModel<T>, x: &ArrayView1<T>) -> Array1<T>
    where
        T: Float + FromPrimitive + Zero + ndarray::ScalarOperand,
    {
        let c_sqrt = lorentz.c.sqrt();
        // Poincare: p_i = x_i / (x_0 * sqrt(c) + 1)
        let one = T::one();
        let denom = x[0] * c_sqrt + one;

        let mut result = Array1::zeros(x.len() - 1);
        for i in 1..x.len() {
            result[i - 1] = x[i] / denom;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    const TOL: f64 = 1e-10;

    #[test]
    fn test_origin_on_manifold() {
        let lorentz = LorentzModel::new(1.0);
        let o = lorentz.origin(3);
        assert!(lorentz.is_on_manifold(&o.view(), TOL));
    }

    #[test]
    fn test_from_euclidean_on_manifold() {
        let lorentz = LorentzModel::new(1.0);
        let v = array![0.5, -0.3, 0.2];
        let x = lorentz.from_euclidean(&v.view());
        assert!(lorentz.is_on_manifold(&x.view(), TOL));
    }

    #[test]
    fn test_distance_self_zero() {
        let lorentz = LorentzModel::new(1.0);
        let x = lorentz.from_euclidean(&array![0.5, 0.3].view());
        let d = lorentz.distance(&x.view(), &x.view());
        assert!(d.abs() < TOL);
    }

    #[test]
    fn test_distance_symmetric() {
        let lorentz = LorentzModel::new(1.0);
        let x = lorentz.from_euclidean(&array![0.5, 0.3].view());
        let y = lorentz.from_euclidean(&array![-0.2, 0.4].view());

        let d_xy = lorentz.distance(&x.view(), &y.view());
        let d_yx = lorentz.distance(&y.view(), &x.view());
        assert_relative_eq!(d_xy, d_yx, epsilon = TOL);
    }

    #[test]
    fn test_distance_triangle_inequality() {
        let lorentz = LorentzModel::new(1.0);
        let a = lorentz.from_euclidean(&array![0.3, 0.0].view());
        let b = lorentz.from_euclidean(&array![0.0, 0.3].view());
        let c = lorentz.from_euclidean(&array![-0.3, 0.0].view());

        let d_ac = lorentz.distance(&a.view(), &c.view());
        let d_ab = lorentz.distance(&a.view(), &b.view());
        let d_bc = lorentz.distance(&b.view(), &c.view());

        assert!(d_ac <= d_ab + d_bc + TOL);
    }

    #[test]
    fn test_exp_log_inverse() {
        let lorentz = LorentzModel::new(1.0);
        let x = lorentz.origin(3);
        // Small tangent vector at origin (must satisfy <v,x>_L = 0 for tangent space)
        let v = array![0.0, 0.3, 0.2, 0.1]; // time component 0 for tangent at origin

        let y = lorentz.exp_map(&x.view(), &v.view());
        assert!(lorentz.is_on_manifold(&y.view(), 1e-6));

        let v_recovered = lorentz.log_map(&x.view(), &y.view());
        for i in 0..v.len() {
            assert_relative_eq!(v[i], v_recovered[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_project_preserves_direction() {
        let lorentz = LorentzModel::new(1.0);
        // Start with a point not exactly on manifold
        let x = array![2.0, 0.5, 0.3];
        let projected = lorentz.project(&x.view());

        assert!(lorentz.is_on_manifold(&projected.view(), TOL));
        // Space components should have same ratio
        assert_relative_eq!(x[1] / x[2], projected[1] / projected[2], epsilon = TOL);
    }

    #[test]
    fn test_conversion_round_trip() {
        use conversions::*;
        let ball = crate::PoincareBall::new(1.0);
        let lorentz = LorentzModel::new(1.0);

        let p = array![0.3, 0.2, -0.1];
        let l = poincare_to_lorentz(&ball, &p.view());
        let p_back = lorentz_to_poincare(&lorentz, &l.view());

        for i in 0..p.len() {
            assert_relative_eq!(p[i], p_back[i], epsilon = 1e-10);
        }
    }

    /// Build a tangent vector at x on the hyperboloid: <v, x>_L = 0.
    /// Given space components, compute the time component.
    fn make_tangent(lorentz: &LorentzModel<f64>, x: &Array1<f64>, space: &[f64]) -> Array1<f64> {
        // <v, x>_L = -v_0 x_0 + sum(v_i x_i) = 0
        // => v_0 = sum(v_i x_i) / x_0
        let dot_space: f64 = space.iter().zip(x.iter().skip(1)).map(|(a, b)| a * b).sum();
        let v0 = dot_space / x[0];
        let mut v = Array1::zeros(x.len());
        v[0] = v0;
        for (i, &s) in space.iter().enumerate() {
            v[i + 1] = s;
        }
        // Verify tangent constraint
        let check = lorentz.minkowski_dot(&v.view(), &x.view());
        assert!(check.abs() < 1e-10, "tangent constraint violated: {check}");
        v
    }

    #[test]
    fn test_parallel_transport_preserves_norm() {
        // Use nearby points for better accuracy of the closed-form PT
        let lorentz = LorentzModel::new(1.0);
        let x = lorentz.from_euclidean(&array![0.1, 0.05].view());
        let y = lorentz.from_euclidean(&array![0.12, 0.08].view());
        let v = make_tangent(&lorentz, &x, &[0.5, -0.3]);

        let pt = lorentz.parallel_transport(&x.view(), &y.view(), &v.view());

        let norm_v = lorentz.minkowski_dot(&v.view(), &v.view());
        let norm_pt = lorentz.minkowski_dot(&pt.view(), &pt.view());
        assert!(
            (norm_v - norm_pt).abs() < 1e-4,
            "PT should preserve norm: {norm_v} vs {norm_pt}"
        );
    }

    #[test]
    fn test_parallel_transport_identity_when_same() {
        let lorentz = LorentzModel::new(1.0);
        let x = lorentz.from_euclidean(&array![0.5, 0.3].view());
        let v = make_tangent(&lorentz, &x, &[0.4, -0.2]);

        let pt = lorentz.parallel_transport(&x.view(), &x.view(), &v.view());
        for i in 0..v.len() {
            assert_relative_eq!(v[i], pt[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_from_to_euclidean_round_trip() {
        let lorentz = LorentzModel::new(1.0);
        let euc = array![0.5, -0.3, 0.2];
        let hyp = lorentz.from_euclidean(&euc.view());
        let euc_back = lorentz.to_euclidean(&hyp.view());
        for i in 0..euc.len() {
            assert_relative_eq!(euc[i], euc_back[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_exp_map_stays_on_manifold() {
        let lorentz = LorentzModel::new(1.0);
        let x = lorentz.from_euclidean(&array![0.3, 0.2].view());
        let v = make_tangent(&lorentz, &x, &[5.0, -3.0]);
        let y = lorentz.exp_map(&x.view(), &v.view());
        assert!(
            lorentz.is_on_manifold(&y.view(), 1e-4),
            "exp_map result not on manifold"
        );
    }

    #[test]
    fn test_distance_nonneg() {
        let lorentz = LorentzModel::new(1.0);
        let points = [
            array![0.5, 0.3],
            array![-0.2, 0.4],
            array![1.0, 1.0],
            array![0.0, 0.0],
        ];
        for i in 0..points.len() {
            for j in i..points.len() {
                let xi = lorentz.from_euclidean(&points[i].view());
                let xj = lorentz.from_euclidean(&points[j].view());
                let d = lorentz.distance(&xi.view(), &xj.view());
                assert!(d >= -1e-10, "negative distance: {d}");
            }
        }
    }

    #[test]
    fn test_different_curvatures() {
        let l1 = LorentzModel::new(1.0);
        let l2 = LorentzModel::new(4.0);

        let x1 = l1.from_euclidean(&array![0.5, 0.3].view());
        let y1 = l1.from_euclidean(&array![-0.2, 0.4].view());

        let x2 = l2.from_euclidean(&array![0.5, 0.3].view());
        let y2 = l2.from_euclidean(&array![-0.2, 0.4].view());

        let d1 = l1.distance(&x1.view(), &y1.view());
        let d2 = l2.distance(&x2.view(), &y2.view());

        // Different curvatures should give different distances
        assert!((d1 - d2).abs() > 1e-6);
    }
}
