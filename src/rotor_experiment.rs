//! Lorentz boosts realized as `geomalg` rotors, for the `rotor-experiment`
//! feature.
//!
//! This module is a measurement harness, not supported API: it is off by
//! default and `#[doc(hidden)]` at the crate root. It exists so
//! `examples/rotor_experiment.rs` and `tests/rotor_experiment.rs` can compare a
//! Cl(D,1) rotor boost against [`LorentzModel::exp_map`] on identical inputs.
//! See `docs/rotor-experiment.md` for the measured numbers.
//!
//! ### Algebra
//!
//! Points are embedded in `Cl(D, 1)`: basis vectors `0..D` are spacelike
//! (square `+1`) and basis vector `D` is timelike (square `-1`), matching the
//! `(-, +, ..., +)` Minkowski form. A vector `m = sum p_i e_i + p_0 e_D` then
//! satisfies `m^2 = <p, p>_L`.
//!
//! A boost of rapidity `eta` that carries a timelike unit vector `t` along a
//! spacelike unit vector `s` is the even element
//!
//! ```text
//! R = cosh(eta/2) - sinh(eta/2) * (s t)
//! ```
//!
//! with `R R~ = 1`, acting as `t -> R t R~ = cosh(eta) t + sinh(eta) s`. The
//! sandwich product is the whole exponential; there is no matrix involved.
//! `axis_boost_rotor` builds the same rotor through `geomalg`'s
//! `Cl::rotor(plane_a, plane_b, angle)` for the axis-aligned case.
//!
//! ### Precomputed action
//!
//! Applying a rotor to a grade-1 vector is linear, so `geomalg` 0.1.1's
//! [`geomalg::Rotor`] caches the images of the basis vectors and
//! [`geomalg::Rotor::apply_vector`] becomes a dense `(D+1) x (D+1)` product
//! instead of two geometric products. `axis_rotor` exposes that type for the
//! coordinate planes; it takes basis-vector indices, so an arbitrary boost
//! plane (`sqrt(c) x ^ v`) is not expressible through it.

use crate::lorentz::LorentzModel;
use geomalg::Cl;
use ndarray::{Array1, ArrayView1};

/// Minkowski Clifford algebra `Cl(D, 1)`: `D` spacelike basis vectors at
/// indices `0..D` and one timelike basis vector at index `D`.
pub type Minkowski<const D: usize> = Cl<D, 1>;

/// Project a hyperboloid point (`[time, space...]`) into `Cl(D, 1)`.
///
/// # Panics
///
/// Panics when `p` does not have `D + 1` components.
pub fn mv_from_point<const D: usize>(p: &ArrayView1<f64>) -> Minkowski<D> {
    assert!(
        p.len() == D + 1,
        "point needs D + 1 = {} components, got {}",
        D + 1,
        p.len()
    );
    let mut values: Vec<f64> = p.iter().skip(1).copied().collect();
    values.push(p[0]);
    Minkowski::<D>::vector(&values)
}

/// Project a `Cl(D, 1)` vector back to `[time, space...]` coordinates.
pub fn point_from_mv<const D: usize>(m: &Minkowski<D>) -> Array1<f64> {
    let mut out = Array1::zeros(D + 1);
    out[0] = m.coefficient(1 << D);
    for (axis, value) in out.iter_mut().skip(1).enumerate() {
        *value = m.coefficient(1 << axis);
    }
    out
}

/// Boost rotor of rapidity `rapidity` carrying timelike unit `time_axis` toward
/// spacelike unit `space_dir`.
///
/// `R = cosh(eta/2) - sinh(eta/2) * (space_dir * time_axis)`.
pub fn boost_rotor<const D: usize>(
    time_axis: &Minkowski<D>,
    space_dir: &Minkowski<D>,
    rapidity: f64,
) -> Minkowski<D> {
    let bivector = space_dir.geometric(time_axis);
    let half = 0.5 * rapidity;
    Minkowski::<D>::scalar(half.cosh()).sub(&bivector.scale(half.sinh()))
}

/// Axis-aligned boost rotor built with `geomalg`'s `Cl::rotor` constructor.
///
/// Boosts along a basis axis by `rapidity`, so this is the same rotor as
/// [`boost_rotor`] with `space_dir = e(space_axis)` and `time_axis = e(D)`.
pub fn axis_boost_rotor<const D: usize>(space_axis: usize, rapidity: f64) -> Minkowski<D> {
    Minkowski::<D>::rotor(space_axis, D, rapidity)
}

/// Rotor analogue of [`LorentzModel::exp_map`]: the boost along the geodesic
/// leaving `x` in tangent direction `v`.
///
/// Returns the identity rotor when `v` is not spacelike, mirroring `exp_map`'s
/// small-vector guard.
pub fn boost_from_tangent<const D: usize>(
    lorentz: &LorentzModel<f64>,
    x: &ArrayView1<f64>,
    v: &ArrayView1<f64>,
) -> Minkowski<D> {
    let v_norm_sq = lorentz.minkowski_dot(v, v);
    if v_norm_sq.is_nan() || v_norm_sq <= 0.0 {
        return Minkowski::<D>::scalar(1.0);
    }
    // exp_x(v) = cosh(sigma) x + sinh(sigma) v / |v|_L with sigma = sqrt(c)|v|_L,
    // which is exactly a boost of rapidity sigma in the (x, v) plane.
    let v_norm = v_norm_sq.sqrt();
    let sigma = lorentz.c.sqrt() * v_norm;
    let time_axis = mv_from_point(x).scale(lorentz.c.sqrt());
    let space_dir = mv_from_point(v).scale(1.0 / v_norm);
    boost_rotor(&time_axis, &space_dir, sigma)
}

/// Rotor analogue of `exp_map(x, log_map(x, y))`: the geodesic translation
/// taking `x` to `y`.
///
/// The direction and rapidity come from the same log map the exp path uses, so
/// the two paths differ only in how the boost is represented and applied.
pub fn translation_rotor<const D: usize>(
    lorentz: &LorentzModel<f64>,
    x: &ArrayView1<f64>,
    y: &ArrayView1<f64>,
) -> Minkowski<D> {
    let v = lorentz.log_map(x, y);
    boost_from_tangent(lorentz, x, &v.view())
}

/// Sandwich action `rotate * m * reverse(rotate)` on a point or tangent vector.
pub fn apply_rotor<const D: usize>(rotate: &Minkowski<D>, m: &ArrayView1<f64>) -> Array1<f64> {
    point_from_mv(&rotate.sandwich(&mv_from_point(m)))
}

/// `geomalg::Rotor` for the axis-aligned boost plane `e(space_axis) ^ e(D)`.
///
/// The angle convention matches [`boost_rotor`] with `time_axis = e(D)`, so the
/// argument is the rapidity. This is the only shape of boost `Rotor::new` can
/// build: it indexes two basis vectors, and a boost at a general point lives in
/// the plane `sqrt(c) x ^ v`, which is not a coordinate plane.
pub fn axis_rotor<const D: usize>(space_axis: usize, rapidity: f64) -> geomalg::Rotor<D, 1> {
    geomalg::Rotor::new(space_axis, D, rapidity)
}

/// Flat `[space..., time]` form of a hyperboloid point.
///
/// This is the memory order `mv_from_point` writes and the slice order
/// [`geomalg::Rotor::apply_vector`] and [`geomalg::Rotor::apply_vectors`]
/// expect, so one buffer can be moved between the two paths unchanged.
pub fn point_to_vector<const D: usize>(p: &ArrayView1<f64>) -> Vec<f64> {
    assert!(
        p.len() == D + 1,
        "point needs D + 1 = {} components, got {}",
        D + 1,
        p.len()
    );
    let mut out: Vec<f64> = p.iter().skip(1).copied().collect();
    out.push(p[0]);
    out
}

/// Inverse of [`point_to_vector`].
///
/// # Panics
///
/// Panics when `v` does not have `D + 1` components.
pub fn vector_to_point<const D: usize>(v: &[f64]) -> Array1<f64> {
    assert!(
        v.len() == D + 1,
        "vector needs D + 1 = {} components, got {}",
        D + 1,
        v.len()
    );
    let mut out = Array1::zeros(D + 1);
    out[0] = v[D];
    for (axis, value) in v[..D].iter().enumerate() {
        out[axis + 1] = *value;
    }
    out
}

/// Deviation of `R R~` from the identity scalar, relative to the square of the
/// rotor's own coefficient norm.
///
/// The raw difference is dominated by the `cosh^2 - sinh^2` cancellation for a
/// large rapidity, so it is normalized by `sum(R_i^2)` to stay scale free.
pub fn rotor_unit_drift<const D: usize>(rotate: &Minkowski<D>) -> f64 {
    let product = rotate.geometric(&rotate.reverse());
    let scale: f64 = rotate.coefficients().iter().map(|c| c * c).sum();
    (product.coefficient(0) - 1.0).abs() / scale.max(1.0)
}
