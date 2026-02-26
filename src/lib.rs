//! Hyperbolic geometry for embedding hierarchical structures.
//!
//! Embed trees and hierarchies in low dimensions where Euclidean would need thousands.
//!
//! # Which Model Should I Use?
//!
//! | Task | Model | Why |
//! |------|-------|-----|
//! | **Learning embeddings** | [`LorentzModel`] | Stable gradients everywhere |
//! | **Visualization** | [`PoincareBall`] | Bounded, intuitive |
//! | **Mixed hierarchy + similarity** | Consider mixed-curvature | Best of both |
//!
//! # Why Hyperbolic?
//!
//! | Year | Observation | Implication |
//! |------|-------------|-------------|
//! | 2017 | Trees embed poorly in Euclidean space | Need exponential dims |
//! | 2017 | Hyperbolic space has exponential volume | Trees embed naturally |
//! | 2018 | Lorentz model more stable | Better for optimization |
//! | 2021 | Mixed-curvature spaces | Best of both worlds |
//!
//! **Key insight**: Hyperbolic space has *exponentially growing* volume
//! with radius, just like trees have exponentially growing nodes with depth.
//! A 10-dimensional hyperbolic space can embed trees that would require
//! thousands of Euclidean dimensions.
//!
//! # The Two Models
//!
//! | Model | Representation | Pros | Cons |
//! |-------|---------------|------|------|
//! | Poincaré Ball | Unit ball | Intuitive, conformal | Gradients vanish at boundary |
//! | Lorentz (Hyperboloid) | Upper sheet of hyperboloid | Stable gradients | Less intuitive |
//!
//! For **learning**, prefer Lorentz: gradients are well-behaved everywhere.
//! For **visualization**, prefer Poincaré: it's a bounded disk.
//!
//! # Mathematical Background
//!
//! The Poincaré ball model represents hyperbolic space as the interior of
//! a unit ball with the metric:
//!
//! ```text
//! ds² = (2/(1-||x||²))² ||dx||²
//! ```
//!
//! As points approach the boundary (||x|| → 1), distances grow infinitely—
//! this is how infinite hierarchical depth fits in finite Euclidean volume.
//!
//! # When to Use
//!
//! - **Taxonomies**: WordNet, Wikipedia categories
//! - **Organizational hierarchies**: Company structures, file systems
//! - **Evolutionary trees**: Phylogenetics, language families
//! - **Social networks**: Often have hierarchical community structure
//!
//! # When NOT to Use
//!
//! - **Flat structures**: No hierarchy to exploit (use Euclidean)
//! - **Grid-like data**: Images, audio (use CNN/RNN)
//! - **Very shallow trees**: Depth < 5, Euclidean often suffices
//!
//! # Connection to Intrinsic Dimension
//!
//! Local Intrinsic Dimensionality (LID) can help decide between hyperbolic
//! and Euclidean embeddings:
//!
//! - **Low LID + hierarchical structure**: Use hyperbolic (Poincaré/Lorentz)
//! - **High LID + uniform structure**: Use Euclidean (HNSW, IVF-PQ)
//! - **Variable LID across regions**: Consider mixed-curvature spaces
//!
//! Research (D-Mercator, 2023) shows that networks with low intrinsic
//! dimension in hyperbolic space exhibit high navigability—meaning greedy
//! routing succeeds. This connects to HNSW's small-world navigation: graphs
//! with low effective dimension are easier to search.
//!
//! See `jin::lid` for LID estimation utilities.
//!
//! # References
//!
//! ## Foundational
//!
//! - Nickel & Kiela (2017), "Poincare Embeddings for Learning Hierarchical
//!   Representations" -- introduced hyperbolic embeddings for ML.
//! - Nickel & Kiela (2018), "Learning Continuous Hierarchies in the Lorentz
//!   Model" -- showed the Lorentz model is more stable for optimization.
//! - Ganea, Becigneul, Hofmann (2018), "Hyperbolic Neural Networks" --
//!   foundational hyperbolic NN layers using Mobius gyrovector operations.
//! - Chami et al. (2019), "Hyperbolic Graph Convolutional Neural Networks"
//!   -- GCNs in hyperbolic space.
//!
//! ## Capacity and training
//!
//! - Kratsios et al. (2023), "Capacity Bounds for Hyperbolic Neural Network
//!   Representations of Latent Tree Structures" -- proves that 2D hyperbolic
//!   space suffices for embedding any tree with bounded distortion; justifies
//!   the low-dimensional regime this library targets.
//! - Yang et al. (2023), "Hyperbolic Representation Learning: Revisiting and
//!   Advancing" (ICML) -- training tricks (gradient clipping, LR scheduling)
//!   matter as much as model choice; practical guidance for users of this
//!   library.
//!
//! ## Flow matching and geometry
//!
//! - Chen & Lipman (2023), "Riemannian Flow Matching on General Geometries"
//!   -- foundational for hyperbolic flow matching via the Manifold trait.
//! - Zaghen et al. (2025), "Towards Variational Flow Matching on General
//!   Geometries" (RG-VFM).
//!
//! ## Mixed curvature (future direction)
//!
//! - Di Giovanni et al. (2022), "Heterogeneous Manifolds for Curvature-Aware
//!   Graph Embedding" -- mixed-curvature product spaces (hyperbolic x
//!   spherical x Euclidean) as a natural extension when data has both
//!   hierarchical and cyclical structure.
//!
//! ## General geometric deep learning
//!
//! - Bronstein et al. (2021/2025), "Geometric Deep Learning" (book/course)
//!   -- see `geometricdeeplearning.com`.

pub mod core;

#[cfg(feature = "ndarray")]
use ndarray::{Array1, ArrayView1};
#[cfg(feature = "ndarray")]
use num_traits::{Float, FromPrimitive, Zero};

#[cfg(feature = "ndarray")]
pub mod lorentz;

#[cfg(feature = "ndarray")]
pub use lorentz::LorentzModel;

use skel::Manifold;

#[cfg(feature = "ndarray")]
impl Manifold for PoincareBall<f64> {
    fn exp_map(&self, x: &ArrayView1<f64>, v: &ArrayView1<f64>) -> Array1<f64> {
        let c_sqrt = self.c.sqrt();
        let v_norm = v.dot(v).sqrt();
        let lambda_x = 2.0 / (1.0 - self.c * x.dot(x));

        if v_norm < 1e-6 {
            return x.to_owned();
        }

        let direction = (c_sqrt * lambda_x * v_norm / 2.0).tanh();
        let scaled_v = (direction / (c_sqrt * v_norm)) * v;

        self.mobius_add(x, &scaled_v.view())
    }

    fn log_map(&self, x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> Array1<f64> {
        let neg_x = -x;
        let diff = self.mobius_add(&neg_x.view(), y);
        let diff_norm = diff.dot(&diff).sqrt();
        let c_sqrt = self.c.sqrt();
        let lambda_x = 2.0 / (1.0 - self.c * x.dot(x));

        if diff_norm < 1e-6 {
            return Array1::zeros(x.len());
        }

        let scale = (2.0 / (c_sqrt * lambda_x)) * (c_sqrt * diff_norm).atanh();
        (scale / diff_norm) * diff
    }

    fn parallel_transport(
        &self,
        x: &ArrayView1<f64>,
        y: &ArrayView1<f64>,
        v: &ArrayView1<f64>,
    ) -> Array1<f64> {
        // Parallel transport along the unique geodesic from x to y, implemented by integrating
        // the parallel-transport ODE induced by the Levi-Civita connection of the conformal metric
        // g_x = λ(x)^2 I (λ(x) = 2 / (1 - c ||x||^2)).
        //
        // This is more faithful than the older "λ-ratio scaling" approximation: it preserves both
        // length and direction (up to numeric error) by following the connection along the geodesic.

        let n_steps: usize = 128;
        let dt: f64 = 1.0 / (n_steps as f64);

        let x0 = x.to_owned();
        let x1 = y.to_owned();

        if (&x0 - &x1).dot(&(&x0 - &x1)).sqrt() < 1e-12 {
            return v.to_owned();
        }

        let geodesic_point = |t: f64| -> Array1<f64> {
            let neg_x0 = x0.mapv(|v| -v);
            let delta = self.mobius_add(&neg_x0.view(), &x1.view());
            let log0 = self.log_map_zero(&delta.view());
            let scaled = log0.mapv(|u| u * t);
            let delta_t = self.exp_map_zero(&scaled.view());
            let xt = self.mobius_add(&x0.view(), &delta_t.view());
            self.project(&xt.view())
        };

        let mut v_cur = v.to_owned();

        for i in 0..n_steps {
            let t = (i as f64) * dt;
            let xt = geodesic_point(t);
            let xt_next = geodesic_point(t + dt);
            let xdot = (&xt_next - &xt) / dt;

            let c = self.c;
            let lambda = 2.0 / (1.0 - c * xt.dot(&xt));

            // (Γ(xt)(xdot, v))^k = c*λ * ( xdot^k (xt·v) + (xt·xdot) v^k - xt^k (xdot·v) )
            let s1 = xt.dot(&v_cur);
            let s2 = xt.dot(&xdot);
            let s3 = xdot.dot(&v_cur);

            let dv = -c * lambda * (&xdot * s1 + &v_cur * s2 - &xt * s3);
            v_cur = v_cur + dv * dt;
        }

        v_cur
    }
}

/// Poincare ball model of hyperbolic geometry.
///
/// The open ball \(\mathbb{B}^d_c = \{x \in \mathbb{R}^d : c\|x\|^2 < 1\}\)
/// equipped with the Riemannian metric
///
/// \[
/// g_x = \lambda_x^2 I, \quad \lambda_x = \frac{2}{1 - c\|x\|^2}
/// \]
///
/// where \(\lambda_x\) is the conformal factor. As \(\|x\| \to 1/\sqrt{c}\),
/// \(\lambda_x \to \infty\), so distances near the boundary grow without bound --
/// this is how infinite hierarchical depth fits in a finite Euclidean volume.
///
/// **Curvature**: the sectional curvature is \(-c\) everywhere. Larger \(c\)
/// means stronger negative curvature (more "room" for tree branching).
/// The standard choice is \(c = 1\).
///
/// **Key operations**: all defined via Mobius gyrovector space arithmetic
/// (Ungar, 2008). See individual method docs for formulas.
#[cfg(feature = "ndarray")]
pub struct PoincareBall<T> {
    /// Curvature parameter (c > 0)
    pub c: T,
}

#[cfg(feature = "ndarray")]
impl<T> PoincareBall<T>
where
    T: Float + FromPrimitive + Zero + ndarray::ScalarOperand + ndarray::LinalgScalar,
{
    /// Create a Poincare ball model with curvature `c`.
    pub fn new(c: T) -> Self {
        Self { c }
    }

    /// Mobius addition in the Poincare ball (gyrovector space operation).
    ///
    /// \[
    /// x \oplus_c y = \frac{(1 + 2c\langle x,y\rangle + c\|y\|^2)\,x + (1 - c\|x\|^2)\,y}
    ///                      {1 + 2c\langle x,y\rangle + c^2\|x\|^2\|y\|^2}
    /// \]
    ///
    /// This is the non-commutative, non-associative "addition" that replaces
    /// Euclidean vector addition in hyperbolic space. It satisfies:
    /// - \(x \oplus_c 0 = x\) (identity)
    /// - \(x \oplus_c (-x) = 0\) (inverse)
    /// - Left cancellation: \((-x) \oplus_c (x \oplus_c y) = y\)
    pub fn mobius_add(&self, x: &ArrayView1<T>, y: &ArrayView1<T>) -> Array1<T> {
        let x_norm_sq = x.dot(x);
        let y_norm_sq = y.dot(y);
        let xy = x.dot(y);

        let c = self.c;
        let one = T::one();
        let two = T::from_f64(2.0).unwrap();

        let denom = one + two * c * xy + c * c * x_norm_sq * y_norm_sq;

        // ndarray supports `&Array * scalar`, but not `scalar * &Array`.
        let s1 = one + two * c * xy + c * y_norm_sq;
        let s2 = one - c * x_norm_sq;
        let term1 = x.to_owned() * s1;
        let term2 = y.to_owned() * s2;

        (term1 + term2) / denom
    }

    /// Hyperbolic distance in the Poincare ball.
    ///
    /// \[
    /// d_c(x, y) = \frac{2}{\sqrt{c}}\,\mathrm{arctanh}\bigl(\sqrt{c}\,\|(-x) \oplus_c y\|\bigr)
    /// \]
    ///
    /// This is the geodesic distance on the Riemannian manifold \((\mathbb{B}^d_c, g)\).
    /// Near the origin it approximates \(2\|x - y\|\); near the boundary it grows
    /// logarithmically in \(1/(1-c\|x\|^2)\).
    pub fn distance(&self, x: &ArrayView1<T>, y: &ArrayView1<T>) -> T {
        let neg_x = x.mapv(|v| -v);
        let neg_x_view = neg_x.view();
        let diff = self.mobius_add(&neg_x_view, y);
        let diff_norm = diff.dot(&diff).sqrt();
        let c_sqrt = self.c.sqrt();
        let two = T::from_f64(2.0).unwrap();

        two / c_sqrt * (c_sqrt * diff_norm).atanh()
    }

    /// Logarithmic map at the origin: \(\log_0(y) = \frac{\mathrm{arctanh}(\sqrt{c}\|y\|)}{\sqrt{c}\|y\|}\,y\).
    ///
    /// Maps a point on the manifold to a tangent vector at the origin.
    /// For small \(\|y\|\), this is approximately the identity.
    pub fn log_map_zero(&self, y: &ArrayView1<T>) -> Array1<T> {
        let y_norm = y.dot(y).sqrt();
        let epsilon = T::from_f64(1e-7).unwrap(); // f32 friendly epsilon

        if y_norm < epsilon {
            return y.to_owned();
        }
        let c_sqrt = self.c.sqrt();
        let scale = (c_sqrt * y_norm).atanh() / (c_sqrt * y_norm);
        y * scale
    }

    /// Exponential map at the origin: \(\exp_0(v) = \frac{\tanh(\sqrt{c}\|v\|)}{\sqrt{c}\|v\|}\,v\).
    ///
    /// Maps a tangent vector at the origin to a point on the manifold.
    /// The \(\tanh\) ensures the result stays inside the ball (\(\tanh(t) < 1\) for all \(t\)).
    pub fn exp_map_zero(&self, v: &ArrayView1<T>) -> Array1<T> {
        let v_norm = v.dot(v).sqrt();
        let epsilon = T::from_f64(1e-7).unwrap();

        if v_norm < epsilon {
            return v.to_owned();
        }
        let c_sqrt = self.c.sqrt();
        let scale = (c_sqrt * v_norm).tanh() / (c_sqrt * v_norm);
        v * scale
    }

    /// Check if point is inside the Poincare ball (||x|| < 1/sqrt(c)).
    pub fn is_in_ball(&self, x: &ArrayView1<T>) -> bool {
        let norm_sq = x.dot(x);
        norm_sq < T::one() / self.c
    }

    /// Project point onto ball boundary if outside.
    pub fn project(&self, x: &ArrayView1<T>) -> Array1<T> {
        let norm = x.dot(x).sqrt();
        let one = T::one();
        let epsilon = T::from_f64(1e-5).unwrap();
        let max_norm = (one / self.c).sqrt() - epsilon;

        if norm > max_norm {
            x * (max_norm / norm)
        } else {
            x.to_owned()
        }
    }
}

#[cfg(all(test, feature = "ndarray"))]
mod tests {
    use super::*;
    use ndarray::array;
    use ndarray::ArrayView1;
    use skel::Manifold;

    const EPS: f64 = 1e-10;

    #[test]
    fn test_distance_self_is_zero() {
        let ball = PoincareBall::new(1.0);
        let x = array![0.1, 0.2, 0.3];
        let d = ball.distance(&x.view(), &x.view());
        assert!(d.abs() < EPS, "distance to self should be 0, got {}", d);
    }

    #[test]
    fn test_distance_symmetric() {
        let ball = PoincareBall::new(1.0);
        let x = array![0.1, 0.2];
        let y = array![0.3, -0.1];
        let d_xy = ball.distance(&x.view(), &y.view());
        let d_yx = ball.distance(&y.view(), &x.view());
        assert!((d_xy - d_yx).abs() < EPS, "distance not symmetric");
    }

    #[test]
    fn test_distance_non_negative() {
        let ball = PoincareBall::new(1.0);
        let x = array![0.1, 0.2];
        let y = array![0.3, -0.1];
        let d = ball.distance(&x.view(), &y.view());
        assert!(d >= 0.0, "distance should be non-negative");
    }

    #[test]
    fn test_distance_triangle_inequality() {
        let ball = PoincareBall::new(1.0);
        let a = array![0.1, 0.0];
        let b = array![0.0, 0.1];
        let c = array![-0.1, 0.0];

        let d_ac = ball.distance(&a.view(), &c.view());
        let d_ab = ball.distance(&a.view(), &b.view());
        let d_bc = ball.distance(&b.view(), &c.view());

        assert!(
            d_ac <= d_ab + d_bc + EPS,
            "triangle inequality violated: {} > {} + {}",
            d_ac,
            d_ab,
            d_bc
        );
    }

    #[test]
    fn test_mobius_add_identity() {
        let ball = PoincareBall::new(1.0);
        let x = array![0.1, 0.2, 0.3];
        let zero = array![0.0, 0.0, 0.0];

        // x + 0 = x
        let result = ball.mobius_add(&x.view(), &zero.view());
        for i in 0..3 {
            assert!(
                (result[i] - x[i]).abs() < EPS,
                "mobius_add with zero failed at index {}",
                i
            );
        }
    }

    #[test]
    fn test_exp_log_round_trip() {
        let ball = PoincareBall::new(1.0);
        let v = array![0.3, 0.2, 0.1]; // tangent vector

        // exp then log should recover original (approximately)
        let on_manifold = ball.exp_map_zero(&v.view());
        let recovered = ball.log_map_zero(&on_manifold.view());

        for i in 0..3 {
            assert!(
                (recovered[i] - v[i]).abs() < 1e-6,
                "exp/log round trip failed at index {}: {} vs {}",
                i,
                recovered[i],
                v[i]
            );
        }
    }

    #[test]
    fn test_exp_map_stays_in_ball() {
        let ball = PoincareBall::new(1.0);

        // Even large tangent vectors should map to inside the ball
        let large_v = array![10.0, 10.0, 10.0];
        let result = ball.exp_map_zero(&large_v.view());

        assert!(
            ball.is_in_ball(&result.view()),
            "exp_map result escaped the ball"
        );
    }

    #[test]
    fn test_project_inside_unchanged() {
        let ball = PoincareBall::new(1.0);
        let inside = array![0.1, 0.2]; // clearly inside unit ball
        let projected = ball.project(&inside.view());

        for i in 0..2 {
            assert!(
                (projected[i] - inside[i]).abs() < EPS,
                "projection changed point already inside ball"
            );
        }
    }

    #[test]
    fn test_project_outside_onto_boundary() {
        let ball = PoincareBall::new(1.0);
        let outside = array![2.0, 0.0]; // clearly outside unit ball
        let projected = ball.project(&outside.view());

        assert!(
            ball.is_in_ball(&projected.view()),
            "projection did not bring point inside ball"
        );
    }

    #[test]
    fn test_curvature_affects_distance() {
        // Different curvatures should give different distances
        // Note: the relationship is subtle - higher c means smaller ball radius
        let ball_c1 = PoincareBall::new(1.0);
        let ball_c2 = PoincareBall::new(4.0);

        let x = array![0.1, 0.0];
        let y = array![0.0, 0.1];

        let d1 = ball_c1.distance(&x.view(), &y.view());
        let d2 = ball_c2.distance(&x.view(), &y.view());

        // Just verify they're different (curvature has an effect)
        // Difference is small for points near origin but non-zero
        assert!(
            (d1 - d2).abs() > 1e-6,
            "curvature should affect distance: c=1 gives {}, c=4 gives {}",
            d1,
            d2
        );
    }

    #[test]
    fn test_mobius_add_inverse() {
        // x + (-x) = 0
        let ball = PoincareBall::new(1.0);
        let x = array![0.3, -0.2, 0.1];
        let neg_x = x.mapv(|v| -v);
        let result = ball.mobius_add(&x.view(), &neg_x.view());
        let norm = result.dot(&result).sqrt();
        assert!(norm < 1e-8, "x + (-x) should be ~0, got norm={norm}");
    }

    #[test]
    fn test_mobius_left_cancellation() {
        // (-x) + (x + y) = y
        let ball = PoincareBall::new(1.0);
        let x = array![0.1, 0.2];
        let y = array![0.15, -0.1];
        let neg_x = x.mapv(|v| -v);
        let x_plus_y = ball.mobius_add(&x.view(), &y.view());
        let result = ball.mobius_add(&neg_x.view(), &x_plus_y.view());
        for i in 0..2 {
            assert!(
                (result[i] - y[i]).abs() < 1e-8,
                "left cancellation failed at dim {i}: {:.8} vs {:.8}",
                result[i],
                y[i]
            );
        }
    }

    #[test]
    fn test_distance_grows_near_boundary() {
        // Points near the boundary should have larger distances
        let ball = PoincareBall::new(1.0);
        let near = array![0.1, 0.0];
        let far = array![0.9, 0.0];
        let offset = array![0.0, 0.05];

        let d_near = ball.distance(&near.view(), &(&near + &offset).view());
        let d_far = ball.distance(&far.view(), &(&far + &offset).view());

        assert!(
            d_far > d_near,
            "distance near boundary should be larger: d_near={d_near}, d_far={d_far}"
        );
    }

    #[test]
    fn test_exp_log_round_trip_non_origin() {
        // exp/log round-trip at a non-origin base point
        let ball = PoincareBall::new(1.0);
        let base = array![0.2, -0.1, 0.15];
        let target = array![0.3, 0.1, -0.05];

        let v = ball.log_map(&base.view(), &target.view());
        let recovered = ball.exp_map(&base.view(), &v.view());

        for i in 0..3 {
            assert!(
                (recovered[i] - target[i]).abs() < 1e-5,
                "non-origin exp/log round trip failed at {i}: {:.6} vs {:.6}",
                recovered[i],
                target[i]
            );
        }
    }

    #[test]
    fn test_distance_matches_between_models() {
        // Poincare and Lorentz models should give the same distance
        use crate::lorentz::{conversions::poincare_to_lorentz, LorentzModel};
        let ball = PoincareBall::new(1.0);
        let lorentz = LorentzModel::new(1.0);

        let p = array![0.3, 0.2];
        let q = array![-0.1, 0.4];

        let d_poincare = ball.distance(&p.view(), &q.view());

        let lp = poincare_to_lorentz(&ball, &p.view());
        let lq = poincare_to_lorentz(&ball, &q.view());
        let d_lorentz = lorentz.distance(&lp.view(), &lq.view());

        assert!(
            (d_poincare - d_lorentz).abs() < 1e-8,
            "models should be isometric: poincare={d_poincare}, lorentz={d_lorentz}"
        );
    }

    fn lambda(c: f64, x: &ArrayView1<f64>) -> f64 {
        2.0 / (1.0 - c * x.dot(x))
    }

    #[test]
    fn parallel_transport_is_identity_when_x_equals_y() {
        let ball = PoincareBall::new(1.0);
        let x = array![0.07, -0.02, 0.03];
        let v = array![0.3, -0.1, 0.2];

        let pt = ball.parallel_transport(&x.view(), &x.view(), &v.view());
        let err = (&pt - &v).mapv(|t| t.abs()).sum();
        assert!(err < 1e-10, "err={}", err);
    }

    #[test]
    fn parallel_transport_preserves_metric_norm_smoke() {
        let ball = PoincareBall::new(1.0);
        let x = array![0.06, -0.03, 0.02];
        let y = array![0.01, 0.04, -0.02];
        let v = array![0.2, -0.15, 0.05];

        let pt = ball.parallel_transport(&x.view(), &y.view(), &v.view());

        let lx = lambda(ball.c, &x.view());
        let ly = lambda(ball.c, &y.view());

        let norm2_x = lx * lx * v.dot(&v);
        let norm2_y = ly * ly * pt.dot(&pt);

        let rel = (norm2_x - norm2_y).abs() / norm2_x.max(1e-12);
        assert!(rel < 5e-4, "rel={}", rel);
    }

    #[test]
    fn parallel_transport_from_origin_matches_lambda_ratio() {
        // Ganea et al. (2018), Thm 4: P_{0->x}(v) = (λ_0 / λ_x) v.
        let ball = PoincareBall::new(1.0);
        let origin = array![0.0, 0.0, 0.0];
        let x = array![0.08, -0.01, 0.03];
        let v = array![0.4, -0.2, 0.1];

        let pt = ball.parallel_transport(&origin.view(), &x.view(), &v.view());
        let l0 = lambda(ball.c, &origin.view());
        let lx = lambda(ball.c, &x.view());
        let expected = v.mapv(|t| t * (l0 / lx));

        let err = (&pt - &expected).mapv(|t| t.abs()).sum();
        assert!(err < 5e-4, "err={}", err);
    }
}
