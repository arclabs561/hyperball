# Examples

Hyperbolic geometry for hierarchical data.

## Quick Start

| Example | What It Teaches |
|---------|-----------------|
| `poincare_basics` | Core operations: distance, Mobius addition, exp/log maps |
| `lorentz_basics` | Lorentz (hyperboloid) model: same operations, different representation |
| `tree_embedding` | Why 2D hyperbolic can embed trees needing O(depth) Euclidean dims |

```sh
cargo run --example poincare_basics --release
cargo run --example tree_embedding --release
```

## Real Hierarchies

| Example | What It Teaches |
|---------|-----------------|
| `taxonomy_embedding` | Embed WordNet-style taxonomy, evaluate with MAP |
| `hierarchy_recovery` | Recover parent-child structure from distances |
| `poincare_sgd` | Learn Poincare embeddings via Riemannian SGD on a toy tree |
| `lorentz_sgd` | Learn Lorentz embeddings via Riemannian SGD on the same tree |

Lorentz avoids the numerical instability of Poincare near the boundary but requires Minkowski-metric gradient correction. Both examples embed the same 25-node tree; compare their distance ratios and depth-norm correlation to see the tradeoff.

```sh
cargo run --example taxonomy_embedding --release
cargo run --example hierarchy_recovery --release
cargo run --example poincare_sgd --release
cargo run --example lorentz_sgd --release
```

## Why Hyperbolic?

**The core insight**: Hyperbolic volume grows exponentially with radius.

```
Euclidean 2D:  area ~ r^2    (polynomial)
Hyperbolic 2D: area ~ e^r    (exponential)
```

Trees also grow exponentially: a binary tree at depth d has 2^d leaves.

**Result**: 2D hyperbolic space can embed trees with low distortion that would require O(log n) Euclidean dimensions.

## Geometric Hierarchy Stack

| Data Type | Geometry | Crate | Why |
|-----------|----------|-------|-----|
| Trees (strict) | Hyperbolic | `hyp` | Exponential volume matches tree growth |
| DAGs/Lattices | Boxes | `subsume` | Containment = entailment |
| Knowledge graphs | Euclidean | `tranz` | TransE, RotatE, point embeddings |
| Dense vectors | Euclidean | `vicinity` | HNSW, IVF-PQ, standard ANN |

## When to Use Hyperbolic

```
Is your data a strict tree?
  └─> hyp (Poincare + Lorentz)

Is your data a DAG with multiple parents?
  └─> subsume (box embeddings)

Is your data a knowledge graph for link prediction?
  └─> tranz (Euclidean KGE)

Just need nearest neighbor search?
  └─> vicinity (HNSW)
```
