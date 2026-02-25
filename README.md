# hyperball

Hyperbolic geometry in Rust: Poincare ball and Lorentz hyperboloid models.

Hyperbolic spaces embed tree-like structures in low dimensions where Euclidean spaces cannot. A binary tree of depth 10 needs ~1000 dimensions in Euclidean space but fits in 2D hyperbolic space with bounded distortion.

## What it provides

- **`PoincareBall<T>`**: Mobius addition, distance, exp/log maps, parallel transport, projection.
- **`LorentzModel<T>`**: Minkowski metric, exp/log maps, parallel transport, Euclidean conversions.
- **Conversions**: Poincare <-> Lorentz round-trip.
- **Diagnostics**: delta-hyperbolicity and ultrametric violation for tree-likeness detection.
- **`skel::Manifold` impl**: `PoincareBall<f64>` implements the `Manifold` trait for use with [`flowmatch`](https://github.com/arclabs561/flowmatch) Riemannian ODE integrators.

Generic over float type via `num_traits::Float`.

## Usage

```toml
[dependencies]
hyperball = "0.1.0"
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

## Examples

```bash
cargo run --example poincare_basics      # distance growth near boundary
cargo run --example tree_embedding       # embed a tree in 2D hyperbolic space
cargo run --example taxonomy_embedding   # WordNet-style hierarchy
cargo run --example hierarchy_recovery   # recover tree structure from distances
cargo run --example graph_diagnostics        # delta-hyperbolicity measurement
cargo run --example lorentz_basics           # Lorentz hyperboloid model basics
cargo run --example distortion_vs_dimension  # Poincare@5D beats Euclidean@50D
```

## Tests

```bash
cargo test -p hyperball
```

73 tests: unit tests, property-based tests (proptest), and numerical stability tests covering Mobius axioms (inverse, left cancellation), metric axioms (symmetry, triangle inequality, boundary growth), exp/log round-trips, parallel transport norm preservation, cross-model isometry, and Lorentz tangent space operations.

## License

MIT OR Apache-2.0
