//! Tree Embedding in Hyperbolic Space
//!
//! Demonstrates that hyperbolic space is natural for embedding trees.
//! In Poincare ball:
//! - Root near origin
//! - Children pushed toward boundary
//! - Siblings at similar radius but different angles
//!
//! # Geometric Hierarchy Stack
//!
//! The Mathematical Foundation provides different geometries for different structures:
//!
//! | Data Type       | Geometry    | Crate       | Why                           |
//! |-----------------|-------------|-------------|-------------------------------|
//! | Trees           | Hyperbolic  | hyp         | Exponential volume = trees    |
//! | DAGs/Lattices   | Boxes       | subsume     | Containment = entailment      |
//! | Knowledge graphs | Euclidean  | tranz       | Point embeddings, TransE/RotatE |
//! | Dense vectors   | Euclidean   | vicinity    | HNSW, IVF-PQ, standard ANN    |
//!
//! This example shows why 2D hyperbolic space can embed trees that would
//! require O(depth) dimensions in Euclidean space.
//!
//! See also: `subsume/knowledge_graph.rs` for box embedding approach.
//!
//! ```bash
//! cargo run --example tree_embedding --release
//! ```

use hyperball::PoincareBall;
use ndarray::{array, Array1};

fn main() {
    println!("Tree Embedding in Hyperbolic Space");
    println!("===================================\n");

    let ball = PoincareBall::<f64>::new(1.0);

    // Embed a simple tree:
    //          root
    //         /    \
    //       A        B
    //      / \      / \
    //     A1  A2   B1  B2

    println!("Tree structure:");
    println!("       root");
    println!("      /    \\");
    println!("     A      B");
    println!("    / \\    / \\");
    println!("   A1 A2  B1 B2\n");

    // Position embeddings (2D for visualization)
    // Root near origin
    let root = array![0.0, 0.0];

    // Level 1: pushed out from origin
    let a = array![0.4, 0.3]; // Left branch
    let b = array![0.4, -0.3]; // Right branch

    // Level 2: even further out
    let a1 = array![0.7, 0.4];
    let a2 = array![0.7, 0.2];
    let b1 = array![0.7, -0.2];
    let b2 = array![0.7, -0.4];

    let nodes = vec![
        ("root", &root),
        ("A", &a),
        ("B", &b),
        ("A1", &a1),
        ("A2", &a2),
        ("B1", &b1),
        ("B2", &b2),
    ];

    // 1. Show positions
    println!("1. Node Positions (Euclidean coordinates)");
    for (name, pos) in &nodes {
        let dot_product: f64 = pos.dot(*pos);
        let norm = dot_product.sqrt();
        let d_from_root = ball.distance(&root.view(), &pos.view());
        println!(
            "   {}: {:?}, ||x||={:.2}, d(root)={:.2}",
            name,
            pos.to_vec(),
            norm,
            d_from_root
        );
    }

    // 2. Parent-child distances vs sibling distances
    println!("\n2. Structural Distances");
    println!("   Parent-child relationships:");
    println!(
        "     d(root, A) = {:.3}",
        ball.distance(&root.view(), &a.view())
    );
    println!(
        "     d(root, B) = {:.3}",
        ball.distance(&root.view(), &b.view())
    );
    println!(
        "     d(A, A1)   = {:.3}",
        ball.distance(&a.view(), &a1.view())
    );
    println!(
        "     d(A, A2)   = {:.3}",
        ball.distance(&a.view(), &a2.view())
    );
    println!(
        "     d(B, B1)   = {:.3}",
        ball.distance(&b.view(), &b1.view())
    );
    println!(
        "     d(B, B2)   = {:.3}",
        ball.distance(&b.view(), &b2.view())
    );

    println!("\n   Sibling distances:");
    println!(
        "     d(A, B)   = {:.3}  (level 1 siblings)",
        ball.distance(&a.view(), &b.view())
    );
    println!(
        "     d(A1, A2) = {:.3}  (level 2 siblings, same parent)",
        ball.distance(&a1.view(), &a2.view())
    );
    println!(
        "     d(B1, B2) = {:.3}  (level 2 siblings, same parent)",
        ball.distance(&b1.view(), &b2.view())
    );

    println!("\n   Cross-branch distances:");
    println!(
        "     d(A1, B1) = {:.3}  (different branches)",
        ball.distance(&a1.view(), &b1.view())
    );
    println!(
        "     d(A2, B2) = {:.3}  (different branches)",
        ball.distance(&a2.view(), &b2.view())
    );

    // 3. Key property: paths through root
    println!("\n3. Key Property: Paths Through Ancestor");
    println!("   In hyperbolic space, shortest paths between nodes in different");
    println!("   branches tend to go 'up' through their common ancestor.\n");

    let d_a1_b1_direct = ball.distance(&a1.view(), &b1.view());
    let d_a1_root = ball.distance(&a1.view(), &root.view());
    let d_root_b1 = ball.distance(&root.view(), &b1.view());

    println!("   Direct d(A1, B1) = {:.3}", d_a1_b1_direct);
    println!(
        "   Via root: d(A1, root) + d(root, B1) = {:.3} + {:.3} = {:.3}",
        d_a1_root,
        d_root_b1,
        d_a1_root + d_root_b1
    );

    // 4. Why 2D is enough for trees
    println!("\n4. Why Low Dimensions Work for Trees");
    println!("   A binary tree of depth d has 2^d leaves.");
    println!("   In Euclidean space, you'd need ~d dimensions.");
    println!("   In hyperbolic space, 2D is often sufficient!\n");
    println!("   This is because hyperbolic space has exponential volume growth:");
    println!("   Area at radius r grows like sinh(r), not r^2.");

    // 5. Compare to Euclidean
    //
    // Note: Both columns use the same hand-placed (x,y) coordinates.
    // A fair comparison would optimize each embedding for its own geometry.
    // The point here is that the hyperbolic metric *amplifies* distances
    // near the boundary, so the same coordinates encode more hierarchy.
    println!("\n5. Euclidean vs Hyperbolic Comparison");
    println!("   (Same coordinates, different metrics. See note in source.)");
    let euclidean_dist = |a: &Array1<f64>, b: &Array1<f64>| {
        let diff = a - b;
        diff.dot(&diff).sqrt()
    };

    println!("   Distance | Euclidean | Hyperbolic | Ratio");
    println!("   ---------|-----------|------------|-------");

    let pairs = [
        ("d(root,A)", &root, &a),
        ("d(root,A1)", &root, &a1),
        ("d(A,A1)", &a, &a1),
        ("d(A1,B1)", &a1, &b1),
    ];

    for (name, p1, p2) in pairs {
        let d_euc = euclidean_dist(p1, p2);
        let d_hyp = ball.distance(&p1.view(), &p2.view());
        println!(
            "   {:10} | {:9.3} | {:10.3} | {:6.2}x",
            name,
            d_euc,
            d_hyp,
            d_hyp / d_euc
        );
    }

    println!("\n   Notice: hyperbolic distances grow faster for points near boundary.");
    println!("   This naturally encodes the tree hierarchy!");

    println!("\nDone!");
}
