# hyperball

[![crates.io](https://img.shields.io/crates/v/hyperball.svg)](https://crates.io/crates/hyperball)
[![Documentation](https://docs.rs/hyperball/badge.svg)](https://docs.rs/hyperball)
[![CI](https://github.com/arclabs561/hyperball/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/hyperball/actions/workflows/ci.yml)

Hyperbolic geometry in Rust: Poincare ball and Lorentz hyperboloid models.

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

**WordNet-style taxonomy**. Embed a hierarchical vocabulary (animal -> mammal -> dog/cat, etc.) and verify that the hierarchy is recoverable from distances alone:

```bash
cargo run --example taxonomy_embedding
```

**Distortion vs. dimension**. Compare embedding quality across dimensions -- Poincare at 5D typically beats Euclidean at 50D for tree metrics:

```bash
cargo run --example distortion_vs_dimension
```

**Delta-hyperbolicity**. Measure how tree-like a metric space is using Gromov's four-point condition. Low delta means the space is close to a tree:

```bash
cargo run --example graph_diagnostics
```

### All examples

```bash
cargo run --example poincare_basics           # distance growth near boundary
cargo run --example tree_embedding            # embed a tree in 2D hyperbolic space
cargo run --example taxonomy_embedding        # WordNet-style hierarchy
cargo run --example hierarchy_recovery        # recover tree structure from distances
cargo run --example graph_diagnostics         # delta-hyperbolicity measurement
cargo run --example lorentz_basics            # Lorentz hyperboloid model
cargo run --example distortion_vs_dimension   # Poincare@5D beats Euclidean@50D
```

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
hyperball = "0.1.4"
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

### Foundational

- Nickel & Kiela (2017), "Poincare Embeddings for Learning Hierarchical Representations" -- introduced hyperbolic embeddings for ML.
- Nickel & Kiela (2018), "Learning Continuous Hierarchies in the Lorentz Model" -- Lorentz model is more stable for optimization.
- Ganea, Becigneul, Hofmann (2018), "Hyperbolic Neural Networks" -- foundational hyperbolic layers.
- Chami et al. (2019), "Hyperbolic Graph Convolutional Neural Networks" -- GCNs in hyperbolic space.

### Numerical stability and capacity

- Mishne et al. (2023), "The Numerical Stability of Hyperbolic Representation Learning" (ICML) -- validates Taylor expansion for acosh near 1, Lorentz over Poincare for stability.
- Kratsios et al. (2023), "Capacity Bounds for Hyperbolic Neural Network Representations of Latent Tree Structures" -- 2D hyperbolic space suffices for tree embedding.

### Training and practice

- Yang et al. (2023), "Hyperbolic Representation Learning: Revisiting and Advancing" (ICML) -- gradient clipping and LR scheduling matter as much as model choice.
- Sakai & Iiduka (2023), "Convergence of Riemannian SGD on Hadamard Manifolds" -- convergence guarantees for optimization on the hyperboloid.

### Recent Lorentz-space architectures

- van der Klis et al. (2026), "Fast and Geometrically Grounded Lorentz Neural Networks" -- fast Lorentz layers with geometric grounding.
- He et al. (2024), "Lorentzian Residual Neural Networks" -- residual connections via Lorentz exp/log.
- Fan et al. (2024), "Enhancing Hyperbolic KG Embeddings via Lorentz Transformations" (ACL Findings) -- Lorentz boosts for knowledge graphs.

### Embeddings and distortion

- van Spengler & Mettes (2025), "Low-distortion and GPU-compatible Tree Embeddings in Hyperbolic Space" -- improved embedding strategies.

### Flow matching and geometry

- Chen & Lipman (2023), "Riemannian Flow Matching on General Geometries" -- foundational for hyperbolic flow matching.
- Di Giovanni et al. (2022), "Heterogeneous Manifolds for Curvature-Aware Graph Embedding" -- mixed-curvature spaces as future direction.

## License

MIT OR Apache-2.0
