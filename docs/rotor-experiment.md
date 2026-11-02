# Rotor boosts vs the exp/log path (experiment)

Status: measurement. Nothing here changed the public API or the default code
path. The rotor path is `src/rotor_experiment.rs`, compiled only with the
non-default `rotor-experiment` feature and re-exported as `#[doc(hidden)]`.

## Verdict

**Accuracy: tie. Speed: the precomputed rotor action wins whenever one rotor is
reused; the generic sandwich never does.**

An earlier version of this page concluded that rotors lose 2x to 11x and
rejected the idea. That measurement was correct about the code it ran and wrong
as a verdict: it used `geomalg` 0.1.0's generic `sandwich` on multivectors, which
is two geometric products plus two heap-allocated conversions per call.
`geomalg` 0.1.1 adds `Rotor`, which precomputes the grade-1 action once, after
which a point costs one dense `4x4` product. Measured here:

| Workload | exp/log | generic sandwich | precomputed action |
|---|---|---|---|
| one boost, rotor prebuilt | 88 ns | 216 ns (2x slower) | **22 ns (4x faster)** |
| one boost, rotor built per call | 88 ns | 443 ns | 665 ns (build + apply) |
| 1,000 points, one rotor shared | 95 ns/pt | 190 ns/pt | **3.7 ns/pt (26x faster)** |

Building a rotor costs ~640 ns, so the precomputed path pays for itself from
about **10 points** that share the rotor. Below that, `exp_map` wins outright,
and building the rotor per call is the worst of both. Per-run medians are 4.3x
(3.2-7.7x) for a single prebuilt call and 25.6x (22.5-33.1x) per point over
1,000 points; the ranges are the spread over eight runs on a loaded host.

Two limits keep this from being a migration:

- `geomalg::Rotor::new` indexes two basis vectors, so it builds only the six
  coordinate planes of Cl(3,1). A boost at a general point lives in the plane
  `sqrt(c) x ^ v`, which the type cannot express: the fast batch path is
  reachable today for axis-aligned boosts only. The per-point cost is a dense
  `(D+1)^2` product with no sparsity test, so it does not depend on the plane;
  what is missing is a constructor, not throughput.
- The win is per-point cost, not a new capability. Accuracy is unchanged, at the
  floating-point floor.

Accuracy and norm preservation, unchanged from the first measurement:

- Agreement between the generic rotor and the closed form is at the
  floating-point floor. The worst scaled disagreement over 1216 configurations
  is `1.3e-13`; on the origin boost, where both paths are pure closed forms, it
  is `6.0e-16`. The precomputed action agrees with `exp_map` to `1.9e-14` and
  with its own sandwich to `2.9e-16`.
- The Minkowski norm is preserved no worse by the rotor, and slightly better in
  the general case: rotor `1.8e-14` vs exp `1.0e-13` relative residual.

So a rotor boost is a *representation* that composes cheaply **and**, through the
precomputed action, applies cheaply to many points at once. The one real
functional difference found is in the small-angle regime, and it favours the
rotor: `exp_map` clamps boosts below `|v| = 3.2e-8` (details below).

## What was compared

`hyperball` has no explicit matrix path. Its Lorentz operations are closed forms
built on `cosh`, `sinh` and `acosh`. The nearest thing to "the matrix exponential" is
`exp_map`, and the nearest thing to a boost is `exp_map(x, log_map(x, y))`.
Those are the baselines below. A Lorentz boost along a unit direction is a
Cl(3,1) rotor, `R = cosh(eta/2) - sinh(eta/2) (s t)`, applied as `R m R~`, so
composing boosts is one geometric product instead of another exponential.

| Operation | Current path | Generic rotor | Precomputed rotor |
|---|---|---|---|
| Boost of the origin | `exp_map(o, v)` | `R` from `(e_t, v/|v|, sqrt(c)|v|)`, then sandwich | `Rotor::new`, then `apply_vector` |
| Boost at a point | `exp_map(x, v)` | `R` from `(sqrt(c) x, v/|v|, sqrt(c)|v|)`, then sandwich | not expressible (plane is not a coordinate plane) |
| Geodesic translation `x → y` | `exp_map(x, log_map(x, y))` | `R = exp` rotor from `log_map(x, y)`, then sandwich | not expressible |
| Parallel transport of `w` | `parallel_transport(x, y, w)` | `R w R~` with the translation rotor | not expressible |
| Compose two translations | two `log`+`PT`+`exp` steps | `R2 R1`, one sandwich | not expressible |

The precomputed column is `geomalg::Rotor`, whose cost is independent of the
plane but whose constructor takes basis-vector indices. Its batch case — one
rotor, many points — is measured on the axis-aligned boost and applies to any
plane.

Not compared: **distance**. `distance` is a scalar invariant,
`acosh(-c<x,y>_L)`, with no rotor form that does not first require the boost
angle, so a rotor version would be circular. Both paths share it.

## Setup

- `hyperball` 0.1.6, Apple M3 Max (`aarch64-apple-darwin`), rustc 1.99.1,
  `--release`.
- `geomalg` 0.1.1, `Cl(3, 1)`: 3 spacelike basis vectors (square `+1`) and one
  timelike (square `-1`), 16 blades. `D = 3` because `LorentzModel` is used at
  3+1 dimensions in the examples; larger `D` makes the rotor path worse, never
  better, because the multivector grows as `2^D`.
- The first measurement ran on `geomalg` 0.1.0. 0.1.1 changed both sides of the
  comparison — it moved `geometric` onto a precomputed blade table (~240 to
  ~177 ns per Cl(3,1) product) and added `Rotor` — so every number below was
  re-measured; none are carried over.
- 1216 configurations: 600 random `(point, direction, target, transport vector,
  rapidity)` draws per curvature for `c ∈ {1.0, 2.5}`, plus the edge rapidities
  `1e-8, 1e-6, 1e-4, pi-1e-9, pi, pi+1e-9, 6, 12` per curvature.
- Random rapidity in `[0, 4)`; directions are unit in the Minkowski form and
  tangent at the base point.

## Agreement

`absolute` is `max_i |exp_i - rotor_i|`. `scaled` divides each component by
`max(1, |exp_i|)`. Both paths are the same algebra, so the absolute number is
pure rounding. It grows with coordinate magnitude, `cosh(sqrt(c) * rapidity)`:
at rapidity 12 and `c = 2.5` the coordinates are `5.5e7`, which is why the boost-at-a-point row has a `1.2e-7` absolute difference and a `1.3e-13`
scaled one.

| Operation | absolute | scaled |
|---|---|---|
| Origin boost | 7.4e-9 | **6.0e-16** |
| Boost at a point | 1.2e-7 | **1.3e-13** |
| Translate `x → y` | 1.1e-13 | 6.5e-14 |
| Translate lands on `y` | 2.1e-14 | 2.1e-14 |
| Parallel transport | 1.0e-13 | 1.0e-13 |
| `Cl::rotor` vs explicit bivector | 0.0 | 0.0 |

The last row checks `geomalg`'s `rotor(plane_a, plane_b, angle)` constructor
against the explicit `cosh + sinh * bivector` form over all three axes and the
edge rapidities: bit-identical, so the constructor is a safe building block for
axis-aligned boosts, but it cannot express an arbitrary direction: it takes
basis-vector indices only.

### Minkowski norm preservation

`|<p,p>_L + 1/c|` after the operation, relative to the coordinate scale
`sum(p_i^2)`. Dividing by `1/c` instead would score the cancellation inside the
check at large rapidity, not the operation.

| Operation | exp | rotor |
|---|---|---|
| Origin boost | 4.6e-16 | 6.7e-16 |
| Boost at a point | 1.0e-13 | **1.8e-14** |
| Translate `x → y` | 1.2e-13 | **1.9e-14** |

Rotor unit drift `|R R~ - 1|`, relative to `sum(R_i^2)`: **7.3e-15**. The rotor
stays on the unit sphere to rounding, so the sandwich product is safe.

### Precomputed action

`geomalg::Rotor` copies the sandwich of each basis vector into a dense
`(D+1) x (D+1)` matrix at construction, so `apply_vector` is one vector product
instead of two geometric products. It is still the same transform, and the
check is not circular: the point is placed in the rotor's own plane, so a boost
along `e(axis)` of a point on the `(e(D), e(axis))` hyperbola is exactly a
coordinate-plane boost of rapidity `offset + rapidity`, and `exp_map` is an
independent closed form for it.

| Quantity | Measured |
|---|---|
| `apply_vector` vs `exp_map`, 132 configurations, 3 axes x 4 offsets x 11 rapidities | **1.9e-14** scaled |
| `apply_vector` vs `Rotor::apply` (the generic sandwich) | 2.9e-16 |
| `apply_vector` Minkowski norm residual | 6.7e-16 |
| batch of 1,000 vs single call | 0.0, bitwise equal |
| batch Minkowski norm residual | 7.9e-16 |

The batch row is exact rather than close because `apply_vectors` calls the same
kernel as `apply_vector`; the example and a test both assert `to_bits` equality.
At rapidity `1e-8` the precomputed path shows the same `1.3e-8` clamp asymmetry
as every other rotor path, for the same reason.

### Small-angle asymmetry (the one real behavioural difference)

`exp_map` returns `x` unchanged when `v_norm_sq < 1e-15`, i.e. `|v| < 3.2e-8`.
The rotor path applies the boost. At rapidity `1e-8` the two paths therefore
differ by `1.3e-8` scaled while `exp_map` returns the input exactly, and the
rotor displacement is `1.3e-8`. This is a property of the existing guard, not a
rotor error, and it is pinned by a test so it cannot change silently.

## Cost

`ns/op`, the minimum of eight independent runs of best-of-7 rounds of at least
80 ms. The host was loaded (load average 22 on 16 cores) and the spread is large:
the same `exp_map` call read 88 ns at its best and 196 ns at its worst across
those runs. Read the ratios, which held, not the absolute values, which are
above an idle machine by an unknown factor. Each row measures one expression
once, so no two rows disagree about the cost of the same code; the ratios are
per-run medians, not ratios of minima.

| Operation | exp/log ns | generic sandwich ns | precomputed ns |
|---|---|---|---|
| Boost of the origin (rotor prebuilt) | 88 | 216 | **22** |
| Boost of the origin (rotor built) | 88 | 443 | 665\* |
| Boost at a point (rotor prebuilt) | 96 | 343 | — |
| Boost at a point (rotor built) | 96 | 757 | — |
| Translate `x → y` (log shared) | 98 | 413 | — |
| Translate `x → y` (rotor built) | 346 | 999 | — |
| Parallel transport (rotor prebuilt) | 118 | 373 | — |
| Compose two translations | 703 | 1373 | — |
| 1,000 points, rotor prebuilt (per point) | 95 | 190 | **3.7** |
| 1,000 points, rotor built once (per point) | 95 | 217 | 4.1 |
| `Rotor::new` (one rotor) | 643 | — | — |

`—` is a path that cannot be constructed for that operation: the plane is not a
coordinate plane. `\*` is the build row plus the apply row: measuring them as one
closure makes the build-inclusive figure a separate sample, which on some runs
mins below the bare build (534-930 ns across the eight runs). Both are the same
~640 ns build plus a ~22 ns apply, so the sum is what a caller pays.

Median per-run ratios, with the observed range over the eight runs:

| Comparison | Median | Range |
|---|---|---|
| precomputed / exp, single (prebuilt) | 4.3x faster | 3.2-7.7x |
| generic sandwich / exp, single | 2.0x slower | 1.5-2.5x |
| precomputed / exp, 1,000 points | 25.6x faster | 22.5-33.1x |
| generic sandwich / exp, 1,000 points | 2.6x slower | 1.5-3.4x |
| generic sandwich / exp, boost at a point | 4.1x slower | 1.7-5.4x |

Breakeven for building one rotor, `643 / (88 - 22)`, is about 10 points; across
the eight runs it fell between 4 and 13. That is the number that decides the
question: below ~10 points sharing a rotor, `exp_map` is the cheaper path; above
it, the precomputed action is, by a margin that grows to 26x per point and stops
there.

Where each path's time goes (`Cl(3,1)`, ns):

| Step | ns |
|---|---|
| `mv_from_point` (allocate + fill) | 57 |
| `point_from_mv` (read out) | 30 |
| one geometric product | 177 |
| sandwich + conversions (what a generic call pays) | 343 |
| `Rotor::new` (build one action matrix) | 643 |
| `Rotor::apply` (sandwich, prebuilt) | 165 |
| `Rotor::apply_vector` (single) | 22 |
| `Rotor::apply_vectors` (per point) | 3.7 |

The generic path is not a representation problem any more, it is an
amortization one. Two geometric products at ~177 ns each against an 88 ns closed
form is why it still loses 2-4x; the precomputed action replaces both with one
16-term product, which is why it wins. The 643 ns build is 7.3x a single
`exp_map`, so the action has to be reused to be worth building — which is
exactly the shape of a batch.

## Reproduce

```sh
cd ~/Developer/hyperball

# Numbers above (rebuild first; the example asserts nothing).
cargo run --release --features rotor-experiment --example rotor_experiment

# Pinned agreement, norm drift and the small-angle asymmetry.
cargo test --features rotor-experiment --test rotor_experiment

# Repo gate, both configurations.
just check
cargo clippy --all-targets --features rotor-experiment -- -D warnings
cargo test --features rotor-experiment
```

The example prints agreement, norm residual and timings. The numbers here come
from one session on 2026-11-02 on `geomalg` 0.1.1; `git log --
docs/rotor-experiment.md` names the commits behind them.

## Pinned bounds

`tests/rotor_experiment.rs` asserts, with roughly 100x headroom over the values
measured here:

| Quantity | Measured | Pinned |
|---|---|---|
| Scaled disagreement, boost | 1.3e-13 | 1e-11 |
| Scaled disagreement, translate/transport | 1.0e-13 | 1e-11 |
| Minkowski norm residual | 1.9e-14 | 1e-11 |
| Rotor unit drift | 7.3e-15 | 1e-13 |
| Precomputed action vs `exp_map` | 1.9e-14 | 1e-11 |
| Precomputed action vs sandwich | 2.9e-16 | 1e-11 |
| Batch vs single call | bitwise equal | exact |

A sign error, a dropped `sqrt(c)`, a reversed bivector order or a missing
normalization all fail these bounds by many orders of magnitude, so a silent
drift in the experiment shows up as a failure.

## Notes and limits

- Single dimension (`D = 3`) and two curvatures. `D = 2` and `D = 4` were not
  measured; `geomalg`'s cost is `O(4^D)` per product, so `D = 4` is worse.
- The generic rotor path here converts to and from `ndarray` per call, as any
  call into this API would. The conversions are small next to the sandwich
  (tens of ns against ~350) and below this host's noise, so they are not quoted
  separately.
- The precomputed path's 22 ns single-call figure is `apply_vector` alone: it
  takes and returns a flat `[f64]` buffer, so it needs no `Cl` and no `ndarray`
  round trip. Its build, 643 ns, is the part that does the conversions.
- `geomalg::Rotor::new` cannot express a general boost plane, so the batch
  numbers are for axis-aligned boosts. The per-point kernel is a dense
  `(D+1)^2` product with no sparsity test, so throughput does not depend on the
  plane; a general-plane action has the same cost and only needs a constructor
  (or four `Cl::sandwich` calls from the public API).
- `geomalg` 0.1.1 declares `rust-version = 1.85`. The feature is opt-in and
  default builds keep the crate's 1.75 MSRV.
- Timings are wall clock on a shared machine, not instruction counts. Do not
  compare these absolute values with numbers from another host.
