# hyperball

[![crates.io](https://img.shields.io/crates/v/hyperball.svg)](https://crates.io/crates/hyperball)
[![Documentation](https://docs.rs/hyperball/badge.svg)](https://docs.rs/hyperball)
[![CI](https://github.com/arclabs561/hyperball/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/hyperball/actions/workflows/ci.yml)

Hyperbolic geometry primitives: Poincare ball, Lorentz model,
and Mobius operations.

## Problem

Tree-like structures (taxonomies, file systems, parse trees) have exponentially many leaves but shallow depth. Euclidean space cannot embed them faithfully in low dimensions -- a binary tree of depth 10 has 1024 leaves and needs roughly that many Euclidean dimensions. Hyperbolic space has exponential volume growth, so the same tree fits in 2D with bounded distortion.

This library provides the core operations (distance, exp/log maps, parallel transport) for both the Poincare ball and Lorentz hyperboloid models, plus diagnostics for measuring how tree-like a dataset is.

## Examples

**Embed a tree in 2D hyperbolic space**. Place a small binary tree in the Poincare disk and compare hyperbolic vs. Euclidean distances:

```bash
cargo run --example tree_embedding
```

```text
Tree structure:
       root
      /    \
     A      B
    / \    / \
   A1 A2  B1 B2

   Distance | Euclidean | Hyperbolic | Ratio
   ---------|-----------|------------|-------
   d(root,A)  |     0.500 |      1.099 |   2.20x
   d(root,A1) |     0.806 |      2.232 |   2.77x
   d(A1,B1)   |     0.600 |      2.366 |   3.94x

   Notice: hyperbolic distances grow faster for points near boundary.
   This naturally encodes the tree hierarchy!
```

See `examples/` for more: `taxonomy_embedding`, `distortion_vs_dimension`, `graph_diagnostics`, `lorentz_basics`, `hierarchy_recovery`, `poincare_basics`.

## What it provides

`PoincareBall<T>` (Mobius addition, distance, exp/log maps, parallel transport, projection) and `LorentzModel<T>` (Minkowski metric, exp/log maps, parallel transport, Euclidean conversions) with round-trip conversions between models. Diagnostics for delta-hyperbolicity and ultrametric violation. `PoincareBall<f64>` implements the `skel::Manifold` trait. Generic over float type via `num_traits::Float`.

## Usage

```toml
[dependencies]
hyperball = "0.1"
```

```rust
use hyperball::PoincareBall;
use ndarray::array;

let ball = PoincareBall::new(1.0); // curvature c = 1

let x = array![0.3, 0.1];
let y = array![-0.2, 0.4];

let d = ball.distance(&x.view(), &y.view());
let v = ball.log_map_zero(&x.view());     // tangent at origin pointing to x
let x_back = ball.exp_map_zero(&v.view()); // round-trip
```

## Tests

```bash
cargo test -p hyperball
```

73 tests: unit tests, property-based tests (proptest), and numerical stability tests covering Mobius axioms (inverse, left cancellation), metric axioms (symmetry, triangle inequality, boundary growth), exp/log round-trips, parallel transport norm preservation, cross-model isometry, and Lorentz tangent space operations.

## References

- Nickel & Kiela (2017), "Poincare Embeddings for Learning Hierarchical Representations"
- Nickel & Kiela (2018), "Learning Continuous Hierarchies in the Lorentz Model"
- Mishne et al. (2023), "The Numerical Stability of Hyperbolic Representation Learning" (ICML)
- Kratsios et al. (2023), "Capacity Bounds for Hyperbolic Neural Network Representations of Latent Tree Structures"
- Yang et al. (2023), "Hyperbolic Representation Learning: Revisiting and Advancing" (ICML)
- Chen & Lipman (2023), "Riemannian Flow Matching on General Geometries"

## License

MIT OR Apache-2.0
