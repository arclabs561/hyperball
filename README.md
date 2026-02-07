# hyp

Hyperbolic geometry primitives for representation learning in non-Euclidean spaces.
Implements the Poincare ball and Lorentz (hyperboloid) models for embedding hierarchical data.

Dual-licensed under MIT or Apache-2.0.

[crates.io](https://crates.io/crates/hyp) | [docs.rs](https://docs.rs/hyp)

## Quickstart

```toml
[dependencies]
hyp = "0.1.0"
ndarray = "0.15"
```

```rust
use hyp::PoincareBall;
use ndarray::array;

let ball = PoincareBall::new(1.0);  // curvature c=1

let x = array![0.1, 0.2];
let y = array![0.3, -0.1];

// Hyperbolic distance
let dist = ball.distance(&x.view(), &y.view());

// Mobius addition (hyperbolic translation)
let sum = ball.mobius_add(&x.view(), &y.view());
```

## Backend choices

This crate is split into:

- `hyp::core`: backend-agnostic implementations that operate on slices (`&[T]`) and return `Vec<T>`.
  This is the “math substrate” layer.
- `hyp`’s `ndarray` API: kept behind the `ndarray` feature (enabled by default) for a convenient
  concrete backend.

## Operations

| Operation | Poincare | Lorentz |
|-----------|----------|---------|
| Distance | `distance()` | `distance()` |
| Addition | `mobius_add()` | - |
| Exp map | `exp_map_zero()` | `exp_map()` |
| Log map | `log_map_zero()` | `log_map()` |
| Project | `project()` | `project()` |

## Examples

- `cargo run --example graph_diagnostics`: build small graphs, compute all-pairs shortest-path
  distances, then measure:
  - δ-hyperbolicity (4-point, exact $O(n^4)$ for small $n$)
  - ultrametric max violation
  - defaults to `testdata/karate_club.edgelist`
  - set `HYP_DATASET=lesmis` or `HYP_DATASET=florentine` for other bundled graphs
  - set `HYP_EDGELIST=/path/to/edges.txt` to run on your own graph

## Why Hyperbolic?

Hyperbolic space has exponentially growing volume with radius, matching how trees have exponentially growing nodes with depth. A 10-dim hyperbolic space embeds trees that would need thousands of Euclidean dimensions.

## Curvature

- `c = 1.0` — Standard hyperbolic space
- `c > 1.0` — Stronger curvature (distances grow faster)
- `c → 0` — Approaches Euclidean

## References (high-signal starting points)

- Nickel & Kiela (2017): *Poincaré Embeddings for Learning Hierarchical Representations*.
- Ganea, Bécigneul, Hofmann (2018): *Hyperbolic Neural Networks* (Poincaré + Lorentz tools).
- Gromov (1987): *Hyperbolic groups* (δ-hyperbolicity; four-point characterizations).
