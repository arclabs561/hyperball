# Examples

Hyperbolic geometry for hierarchical data.

## Quick Start

| Example | Covers |
|---------|--------|
| `poincare_basics` | Core operations: distance, Mobius addition, exp/log maps |
| `lorentz_basics` | Lorentz (hyperboloid) model: same operations, different representation |
| `tree_embedding` | Tree distortion comparison |

```sh
cargo run --example poincare_basics --release
cargo run --example tree_embedding --release
```

## Synthetic Hierarchies

| Example | Covers |
|---------|--------|
| `taxonomy_embedding` | Evaluate hand-crafted coordinates for a WordNet-inspired taxonomy |
| `hierarchy_recovery` | Inspect hierarchy-related distance summaries |
| `poincare_sgd` | Learn Poincare embeddings via Riemannian SGD on a synthetic tree |
| `lorentz_sgd` | Learn Lorentz embeddings via Riemannian SGD on the same synthetic tree |

Lorentz avoids the numerical instability of Poincare near the boundary but requires Minkowski-metric gradient correction. Both examples embed the same 25-node tree; compare their distance ratios and depth-norm correlation to see the tradeoff.

```sh
cargo run --example taxonomy_embedding --release
cargo run --example hierarchy_recovery --release
cargo run --example poincare_sgd --release
cargo run --example lorentz_sgd --release
```

## Hyperbolic Geometry

Hyperbolic volume grows exponentially with radius.

```
Euclidean 2D:  area ~ r^2    (polynomial)
Hyperbolic 2D: area ~ e^r    (exponential)
```

Trees also grow exponentially: a binary tree at depth d has 2^d leaves.
That is why low-dimensional hyperbolic embeddings can represent tree-like
structure with lower distortion than low-dimensional Euclidean embeddings.
