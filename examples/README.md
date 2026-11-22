# Examples

Hyperbolic geometry for hierarchical data.

## Quick start

| Example | Covers |
|---------|--------|
| `poincare_basics` | Core operations: distance, Mobius addition, exp/log maps |
| `lorentz_basics` | Lorentz (hyperboloid) model: same operations, different representation |
| `tree_embedding` | Tree distortion comparison |

```sh
cargo run --example poincare_basics --release
cargo run --example tree_embedding --release
```

## Measurement

| Example | Covers |
|---------|--------|
| `rotor_experiment` | Cl(3,1) rotor boosts vs the exp/log path, generic sandwich and precomputed action: agreement, norm drift, ns/op |

```sh
cargo run --release --features rotor-experiment --example rotor_experiment
```

Opt-in feature, not part of the supported API. See `docs/rotor-experiment.md`.

## Synthetic hierarchies

| Example | Covers |
|---------|--------|
| `taxonomy_embedding` | Evaluate hand-crafted coordinates for a WordNet-inspired taxonomy |
| `hierarchy_recovery` | Inspect hierarchy-related distance summaries |
| `poincare_sgd` | Learn Poincare embeddings via Riemannian SGD on a synthetic tree |
| `lorentz_sgd` | Learn Lorentz embeddings via Riemannian SGD on the same synthetic tree |

The Lorentz example uses a Minkowski-metric gradient correction; the Poincare example rescales gradients using the ball metric. Both examples embed the same 26-node tree; compare their distance ratios and depth-norm correlation to see the tradeoff.

```sh
cargo run --example taxonomy_embedding --release
cargo run --example hierarchy_recovery --release
cargo run --example poincare_sgd --release
cargo run --example lorentz_sgd --release
```

## Hyperbolic geometry

Hyperbolic volume grows exponentially with radius.

```
Euclidean 2D:  area ~ r^2    (polynomial)
Hyperbolic 2D: area ~ e^r    (exponential)
```

Trees also grow exponentially: a binary tree at depth d has 2^d leaves.
That is why low-dimensional hyperbolic embeddings can represent tree-like
structure with lower distortion than low-dimensional Euclidean embeddings.
