# Rotor boosts vs the exp/log path (experiment)

Status: measurement. Nothing here changed the public API or the default code
path. The rotor path is `src/rotor_experiment.rs`, compiled only with the
non-default `rotor-experiment` feature and re-exported as `#[doc(hidden)]`.

## Verdict

**Accuracy: tie. Speed: rotors lose, 2x to 11x, and lose everywhere.**

- Agreement between the two paths is at the floating-point floor. The worst
  scaled disagreement over 1216 configurations is `1.3e-13`; on the origin
  boost, where both paths are pure closed forms, it is `6.0e-16`.
- The Minkowski norm is preserved no worse by the rotor, and slightly better in
  the general case: rotor `1.8e-14` vs exp `1.0e-13` relative residual.
- Every rotor operation is slower. The gap is widest for a single boost
  (`8e-11 s` vs `9e-8 s`, 8x) and narrowest for composing two translations
  (`774 ns` vs `1626 ns`, 2x). `geomalg` 0.1.0 is a dense, heap-allocated
  multivector: one Cl(3,1) geometric product costs ~240 ns on its own, while the
  whole closed-form `exp_map` costs ~90 ns.

So a rotor boost is a *representation* that composes cheaply, not a faster way
to apply one boost. Migrating the current path would pay 2-11x per operation for
no accuracy gain. The one real functional difference found is in the small-angle
regime, and it favours the rotor: `exp_map` clamps boosts below `|v| = 3.2e-8`
(details below).

## What was compared

`hyperball` has no explicit matrix path. Its Lorentz operations are closed forms
built on `cosh`, `sinh` and `acosh`. The nearest thing to "the matrix exponential" is
`exp_map`, and the nearest thing to a boost is `exp_map(x, log_map(x, y))`.
Those are the baselines below. A Lorentz boost along a unit direction is a
Cl(3,1) rotor, `R = cosh(eta/2) - sinh(eta/2) (s t)`, applied as `R m R~`, so
composing boosts is one geometric product instead of another exponential.

| Operation | Current path | Rotor path |
|---|---|---|
| Boost of the origin | `exp_map(o, v)` | `R` from `(e_t, v/|v|, sqrt(c)|v|)`, then sandwich |
| Boost at a point | `exp_map(x, v)` | `R` from `(sqrt(c) x, v/|v|, sqrt(c)|v|)`, then sandwich |
| Geodesic translation `x → y` | `exp_map(x, log_map(x, y))` | `R = exp` rotor from `log_map(x, y)`, then sandwich |
| Parallel transport of `w` | `parallel_transport(x, y, w)` | `R w R~` with the translation rotor |
| Compose two translations | two `log`+`PT`+`exp` steps | `R2 R1`, one sandwich |

Not compared: **distance**. `distance` is a scalar invariant,
`acosh(-c<x,y>_L)`, with no rotor form that does not first require the boost
angle, so a rotor version would be circular. Both paths share it.

## Setup

- `hyperball` 0.1.6, Apple M3 Max (`aarch64-apple-darwin`), rustc 1.99.1,
  `--release`.
- `geomalg` 0.1.0, `Cl(3, 1)`: 3 spacelike basis vectors (square `+1`) and one
  timelike (square `-1`), 16 blades. `D = 3` because `LorentzModel` is used at
  3+1 dimensions in the examples; larger `D` makes the rotor path worse, never
  better, because the multivector grows as `2^D`.
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

### Small-angle asymmetry (the one real behavioural difference)

`exp_map` returns `x` unchanged when `v_norm_sq < 1e-15`, i.e. `|v| < 3.2e-8`.
The rotor path applies the boost. At rapidity `1e-8` the two paths therefore
differ by `1.3e-8` scaled while `exp_map` returns the input exactly, and the
rotor displacement is `1.3e-8`. This is a property of the existing guard, not a
rotor error, and it is pinned by a test so it cannot change silently.

## Cost

`ns/op`, best of four independent runs of best-of-7 rounds of at least 80 ms.
The host was heavily loaded (load average 37-75), so absolute values are roughly
2x an idle machine and run-to-run spread is up to 40%; the ratios held inside
the ranges quoted.

| Operation | exp/log ns | rotor ns | ratio |
|---|---|---|---|
| Boost of the origin (rotor prebuilt) | 89 | 306 | 3.4x |
| Boost at a point (rotor prebuilt) | 84 | 686 | 8.2x |
| Boost at a point (rotor built) | 89 | 928 | 10.4x |
| Translate `x → y` (log shared) | 94 | 560 | 6.0x |
| Translate `x → y` (rotor built) | 381 | 1239 | 3.3x |
| Parallel transport (rotor prebuilt) | 106 | 678 | 6.4x |
| Compose two translations | 774 | 1626 | 2.1x |

Observed ratio ranges across the four runs: 2.4-3.4x (origin boost), 5.9-8.2x
(point, prebuilt), 8.5-11x (point, built), 4.2-9.3x (translate, log shared),
3.1-4.1x (translate, built), 5.0-7.8x (transport), 1.6-2.7x (compose).

Where the rotor time goes (`Cl(3,1)`, ns):

| Step | ns |
|---|---|
| `mv_from_point` (allocate + fill) | 44 |
| `point_from_mv` (read out) | 25 |
| one geometric product | 238 |
| sandwich, no conversions | 606 |
| sandwich + conversions (what a call pays) | 679 |

Two geometric products at ~240 ns each, against a full closed-form `exp_map` at
~90 ns, is the whole story. Replacing the heap allocation in `geomalg` with
fixed-size arrays, or precomputing a blade table, would narrow this. The closed
form would still win for a single boost. The composition row is the only
place the gap narrows, and it narrows to 2x, not to parity.

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
from one session on 2026-11-01; `git log -- docs/rotor-experiment.md` names the
commit that added them.

## Pinned bounds

`tests/rotor_experiment.rs` asserts, with roughly 100x headroom over the values
measured here:

| Quantity | Measured | Pinned |
|---|---|---|
| Scaled disagreement, boost | 1.3e-13 | 1e-11 |
| Scaled disagreement, translate/transport | 1.0e-13 | 1e-11 |
| Minkowski norm residual | 1.9e-14 | 1e-11 |
| Rotor unit drift | 7.3e-15 | 1e-13 |

A sign error, a dropped `sqrt(c)`, a reversed bivector order or a missing
normalization all fail these bounds by many orders of magnitude, so a silent
drift in the experiment shows up as a failure.

## Notes and limits

- Single dimension (`D = 3`) and two curvatures. `D = 2` and `D = 4` were not
  measured; `geomalg`'s cost is `O(4^D)` per product, so `D = 4` is worse.
- The rotor path here converts to and from `ndarray` per call, as any call into
  this API would. A pipeline holding multivectors would save the ~70 ns
  conversion, not the ~240 ns product.
- `geomalg` 0.1.0 declares `rust-version = 1.85`. The feature is opt-in and
  default builds keep the crate's 1.75 MSRV.
- Timings are wall clock on a shared machine, not instruction counts. Do not
  compare these absolute values with numbers from another host.
