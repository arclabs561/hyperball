//! Lorentz hyperboloid model basics.
//!
//! See also: `poincare_basics.rs` for the Poincare ball model equivalent.
//!
//! The Lorentz model represents hyperbolic space as the upper sheet of a
//! two-sheeted hyperboloid in Minkowski space: -x_0^2 + x_1^2 + ... + x_n^2 = -1/c
//!
//! Compared to the Poincare ball:
//! - Better gradient properties (no vanishing near boundary)
//! - More numerically stable for optimization
//! - Natural for Lorentzian geometry (relativity, causal structure)
//!
//! Reference: Nickel & Kiela, "Learning Continuous Hierarchies in the Lorentz Model" (2018)
//!
//! Run: cargo run --example lorentz_basics

use hyperball::lorentz::LorentzModel;
use ndarray::array;

fn main() {
    let model = LorentzModel::<f64>::new(1.0);

    // The origin in Lorentz coordinates is (1, 0, 0, ...) -- the "north pole"
    let origin = array![1.0, 0.0, 0.0];

    println!("=== Lorentz Hyperboloid Model ===\n");

    // Verify origin is on the manifold: <o,o>_L = -1/c
    let inner = model.minkowski_dot(&origin.view(), &origin.view());
    println!(
        "Minkowski norm of origin: {:.4} (should be -1/c = -1.0)",
        inner
    );
    println!(
        "Origin on manifold? {}\n",
        model.is_on_manifold(&origin.view(), 1e-6)
    );

    // Create points via from_euclidean (lifts 2D Euclidean to hyperboloid)
    let points_euclidean = [
        array![0.0, 0.0],
        array![0.5, 0.0],
        array![0.0, 0.5],
        array![1.0, 1.0],
        array![2.0, 0.0],
    ];

    let points_lorentz: Vec<_> = points_euclidean
        .iter()
        .map(|p| model.from_euclidean(&p.view()))
        .collect();

    println!("Euclidean -> Lorentz lifting:");
    for (euc, lor) in points_euclidean.iter().zip(&points_lorentz) {
        println!(
            "  {:>10} -> [{:.3}, {:.3}, {:.3}]  on manifold? {}",
            format!("[{:.1}, {:.1}]", euc[0], euc[1]),
            lor[0],
            lor[1],
            lor[2],
            model.is_on_manifold(&lor.view(), 1e-6)
        );
    }

    // Distance matrix
    println!("\nPairwise hyperbolic distances:");
    println!("{:>10}", "");
    for (i, pe) in points_euclidean.iter().enumerate() {
        print!("{:>10}", format!("[{:.1},{:.1}]", pe[0], pe[1]));
        for j in 0..=i {
            let d = model.distance(&points_lorentz[i].view(), &points_lorentz[j].view());
            print!(" {:>6.3}", d);
        }
        println!();
    }

    // Exp/log round-trip from origin
    println!("\nExp/log round-trip test:");
    let p = &points_lorentz[3]; // [1.0, 1.0] in Euclidean
    let v = model.log_map(&origin.view(), &p.view());
    let p_back = model.exp_map(&origin.view(), &v.view());
    let err: f64 = p
        .iter()
        .zip(p_back.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    println!("  point:     [{:.6}, {:.6}, {:.6}]", p[0], p[1], p[2]);
    println!("  log(o, p): [{:.6}, {:.6}, {:.6}]", v[0], v[1], v[2]);
    println!(
        "  exp(o, v): [{:.6}, {:.6}, {:.6}]",
        p_back[0], p_back[1], p_back[2]
    );
    println!("  round-trip error: {:.2e}", err);

    // Poincare <-> Lorentz conversion round-trip
    println!("\nPoincare <-> Lorentz round-trip:");
    let poincare_pt = model.to_euclidean(&p.view());
    let back_to_lorentz = model.from_euclidean(&poincare_pt.view());
    let conv_err: f64 = p
        .iter()
        .zip(back_to_lorentz.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    println!("  Lorentz:  [{:.6}, {:.6}, {:.6}]", p[0], p[1], p[2]);
    println!("  Poincare: [{:.6}, {:.6}]", poincare_pt[0], poincare_pt[1]);
    println!(
        "  Back:     [{:.6}, {:.6}, {:.6}]",
        back_to_lorentz[0], back_to_lorentz[1], back_to_lorentz[2]
    );
    println!("  conversion error: {:.2e}", conv_err);

    println!("\nKey properties:");
    println!("  - Distances grow exponentially near boundary (hyperbolic metric)");
    println!("  - Lorentz model is numerically stabler than Poincare for optimization");
    println!("  - Both models are isometric (same distances after conversion)");
}
