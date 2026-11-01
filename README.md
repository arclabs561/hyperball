# hyperball

[![crates.io](https://img.shields.io/crates/v/hyperball.svg)](https://crates.io/crates/hyperball)
[![Documentation](https://docs.rs/hyperball/badge.svg)](https://docs.rs/hyperball)

Hyperbolic geometry primitives for embeddings.

## Problem

Tree-like structures (taxonomies, file systems, parse trees) have exponentially many leaves but shallow depth. Euclidean space cannot embed them faithfully: no Euclidean embedding of a complete binary tree preserves all of its distances, and the best achievable distortion grows with depth however many dimensions are used (Bourgain 1986). Hyperbolic space has exponential volume growth, so a tree embeds in the 2D hyperbolic plane with distortion arbitrarily close to 1 (Sarkar 2011).

`hyperball` provides distance, exp/log maps, parallel transport, and model
conversion for the Poincare ball and Lorentz hyperboloid models, plus
diagnostics for measuring how tree-like a dataset is.

## Examples

**Embed a tree in 2D hyperbolic space**. Place a small binary tree in the Poincare disk and compare hyperbolic vs. Euclidean distances:

```bash
cargo run --example tree_embedding
```

Excerpt of the output:

```text
Tree structure:
       root
      /    \
     A      B
    / \    / \
   A1 A2  B1 B2

5. Euclidean vs Hyperbolic Comparison
   (Same coordinates, different metrics. See note in source.)
   Distance | Euclidean | Hyperbolic | Ratio
   ---------|-----------|------------|-------
   d(root,A)  |     0.500 |      1.099 |   2.20x
   d(root,A1) |     0.806 |      2.232 |   2.77x
   d(A,A1)    |     0.316 |      1.167 |   3.69x
   d(A1,B1)   |     0.600 |      2.366 |   3.94x
```

More in [`examples/`](examples/):

- `taxonomy_embedding` evaluates hand-crafted Poincare coordinates for a small, WordNet-inspired taxonomy.
- `hierarchy_recovery` examines how a synthetic hierarchy appears in hyperbolic and Euclidean distance summaries.
- `distortion_vs_dimension` compares one deterministic tree construction across dimensions, inspired by work on Poincare embeddings.
- `graph_diagnostics` reports delta-hyperbolicity and ultrametric violation for small graph distance matrices. These are descriptive signals, not a prediction of embedding quality. The exact four-point delta diagnostic is `O(n^4)`, intended for small `n`.
- `poincare_basics` and `lorentz_basics` walk through the two equivalent models of hyperbolic space (ball and hyperboloid) side by side.
- `poincare_sgd` learns Poincare embeddings by Riemannian SGD on a synthetic 25-node tree; `lorentz_sgd` does the same in the Lorentz model.

## What it provides

`PoincareBall<T>` (Mobius addition, distance, exp/log maps, parallel transport, projection) and `LorentzModel<T>` (Minkowski metric, exp/log maps, parallel transport, Euclidean conversions) with round-trip conversions between models. Diagnostics for delta-hyperbolicity and ultrametric violation. `PoincareBall<f64>` implements the `skel::Manifold` trait. Generic over float type via `num_traits::Float`.

## Usage

```toml
[dependencies]
hyperball = "0.1.7"
ndarray = "0.16"
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

Inputs are caller-validated: every method assumes each point satisfies
`c‖x‖² < 1`. Outside that domain the result is unspecified: `distance` and
`log_map_zero` can return `NaN`, `∞`, or a finite but meaningless value. Call
`is_in_ball` or `project` first.

### Validated points

`BallPoint<T>` (`hyperball::core`) carries the admissibility invariant
`‖x‖ < 1/√c`, so boundary handling is decided at construction rather than
per call site:

- `BallPoint::new(c, coords)` rejects non-finite coordinates and any point with
  `‖x‖ >= 1/√c` (`BallPointError`), including the boundary itself.
- `BallPoint::project(c, x)` clamps exactly like `PoincareBall::project`.
- `BallPoint::distance` and `BallPoint::mobius_add` match the slice API
  bit-for-bit, because they delegate to it unchanged.

The free slice methods (`PoincareBall::distance` and the rest) remain for now;
they are the unchecked layer and will be deprecated in a later breaking release.

## Tests

```bash
cargo test -p hyperball
```

The suite includes unit tests, property-based tests (proptest) for the Mobius
and metric axioms, exp/log round-trips and cross-model isometry, and
numerical stability tests near the boundary.

## References

- Nickel & Kiela (2017), "Poincare Embeddings for Learning Hierarchical Representations"
- Nickel & Kiela (2018), "Learning Continuous Hierarchies in the Lorentz Model"
- Mishne et al. (2023), "The Numerical Stability of Hyperbolic Representation Learning" (ICML)

## License

MIT OR Apache-2.0
