//! Taxonomy Embedding in Hyperbolic Space
//!
//! Evaluates hand-crafted Poincare coordinates on a small synthetic taxonomy.
//! Uses metrics described by Nickel & Kiela (2017), "Poincare Embeddings."
//!
//! # Why Hyperbolic Space for Hierarchies?
//!
//! Euclidean space has polynomial volume growth: area ~ r^d
//! Hyperbolic space has exponential growth: area ~ e^r
//!
//! This matches tree structure:
//! - Binary tree depth d has 2^d leaves
//! - Hyperbolic disk at radius r has ~e^r area
//!
//! Result: 2D hyperbolic ≈ log(n)-dimensional Euclidean for trees!
//!
//! # Evaluation Metrics (from Poincare Embeddings paper)
//!
//! 1. **Mean Rank (MR)**: For edge (u,v), rank of v among all nodes by distance to u
//!    Lower is better. Perfect = 1.0 (child is always closest).
//!
//! 2. **Mean Average Precision (MAP)**: Average precision at each recall level
//!    Higher is better. Perfect = 1.0.
//!
//! 3. **Reconstruction Accuracy**: How often do distances from these
//!    hand-crafted coordinates retrieve a parent-child edge?
//!
//! # Dataset
//!
//! A WordNet-inspired animal taxonomy (not loaded from WordNet):
//! ```text
//!                    animal
//!                   /      \
//!            mammal         bird
//!           /   |   \        |
//!       canine feline primate  songbird
//!        /  \    |      |       |
//!      dog  wolf cat  monkey  sparrow
//! ```
//!
//! ```bash
//! cargo run --example taxonomy_embedding --release
//! ```

use hyperball::PoincareBall;
use ndarray::array;
use std::collections::HashMap;

fn main() {
    println!("Taxonomy Embedding in Hyperbolic Space");
    println!("======================================\n");
    println!("Based on evaluation from Nickel & Kiela (2017) Poincare Embeddings.\n");

    let ball = PoincareBall::<f64>::new(1.0);

    // Define taxonomy with parent-child edges
    // Format: (parent, child, depth_of_child)
    let edges = [
        ("animal", "mammal", 1),
        ("animal", "bird", 1),
        ("mammal", "canine", 2),
        ("mammal", "feline", 2),
        ("mammal", "primate", 2),
        ("bird", "songbird", 2),
        ("canine", "dog", 3),
        ("canine", "wolf", 3),
        ("feline", "cat", 3),
        ("primate", "monkey", 3),
        ("songbird", "sparrow", 3),
    ];

    // Hand-crafted embeddings following Poincare structure:
    // - Root near origin
    // - Children at larger radius (toward boundary)
    // - Siblings spread angularly
    //
    // In practice, these would be learned via Riemannian SGD.
    let mut embeddings: HashMap<&str, [f64; 2]> = HashMap::new();

    // Root (depth 0)
    embeddings.insert("animal", [0.0, 0.0]);

    // Depth 1: spread around origin
    embeddings.insert("mammal", [0.35, 0.2]);
    embeddings.insert("bird", [0.35, -0.2]);

    // Depth 2: further out, grouped by parent
    embeddings.insert("canine", [0.6, 0.35]);
    embeddings.insert("feline", [0.6, 0.15]);
    embeddings.insert("primate", [0.55, 0.0]);
    embeddings.insert("songbird", [0.6, -0.25]);

    // Depth 3: near boundary, grouped by parent
    embeddings.insert("dog", [0.78, 0.42]);
    embeddings.insert("wolf", [0.78, 0.30]);
    embeddings.insert("cat", [0.78, 0.12]);
    embeddings.insert("monkey", [0.75, -0.05]);
    embeddings.insert("sparrow", [0.78, -0.28]);

    // Print embeddings with norms and depths
    println!("1. Embeddings (position, norm, depth-from-root):");
    println!(
        "   {:12} {:>18} {:>8} {:>6}",
        "Node", "Position", "||x||", "Depth"
    );
    println!("   {}", "-".repeat(50));

    for (name, &pos) in embeddings.iter() {
        let norm = (pos[0] * pos[0] + pos[1] * pos[1]).sqrt();
        let depth = get_depth(name, &edges);
        println!(
            "   {:12} ({:>7.3}, {:>7.3}) {:>8.3} {:>6}",
            name, pos[0], pos[1], norm, depth
        );
    }

    // Verify depth-distance correlation
    println!("\n2. Depth vs Distance from Origin (Correlation):");
    let mut depths = Vec::new();
    let mut distances = Vec::new();
    for (name, &pos) in embeddings.iter() {
        let depth = get_depth(name, &edges) as f64;
        let origin = array![0.0, 0.0];
        let point = array![pos[0], pos[1]];
        let hyp_dist = ball.distance(&origin.view(), &point.view());
        depths.push(depth);
        distances.push(hyp_dist);
    }
    let corr = pearson_correlation(&depths, &distances);
    println!("   Pearson correlation: {:.3}", corr);
    println!("   Expected: > 0.9 (deeper nodes further from origin)");

    // Compute Mean Rank (MR) for edge reconstruction
    println!("\n3. Mean Rank (MR) for Edge Reconstruction:");
    println!("   For each edge (parent, child), rank child by distance to parent.");
    println!("   Perfect MR = 1.0 (child is always nearest).\n");

    let all_nodes: Vec<&str> = embeddings.keys().copied().collect();
    let mut total_rank = 0.0;
    let mut edge_count = 0;

    for &(parent, child, _depth) in &edges {
        let parent_pos = embeddings[parent];
        let child_pos = embeddings[child];
        let parent_arr = array![parent_pos[0], parent_pos[1]];
        let child_arr = array![child_pos[0], child_pos[1]];
        let child_dist = ball.distance(&parent_arr.view(), &child_arr.view());

        // Compute distances to all other nodes
        let mut distances_to_others: Vec<(&str, f64)> = all_nodes
            .iter()
            .filter(|&&n| n != parent)
            .map(|&n| {
                let pos = embeddings[n];
                let arr = array![pos[0], pos[1]];
                let dist = ball.distance(&parent_arr.view(), &arr.view());
                (n, dist)
            })
            .collect();

        distances_to_others.sort_by(|a, b| a.1.total_cmp(&b.1));

        // Find rank of child
        let rank = distances_to_others
            .iter()
            .position(|(n, _)| *n == child)
            .map(|p| p + 1)
            .unwrap_or(all_nodes.len());

        println!(
            "   {} -> {}: rank {} (dist {:.3})",
            parent, child, rank, child_dist
        );
        total_rank += rank as f64;
        edge_count += 1;
    }

    let mean_rank = total_rank / edge_count as f64;
    println!("\n   Mean Rank (MR): {:.2}", mean_rank);
    println!("   Expected for good embedding: < 3.0");

    // Compute reconstruction accuracy
    println!("\n4. Edge Reconstruction Accuracy:");
    println!("   An edge is 'reconstructed' if child is in top-k nearest to parent.\n");

    for k in [1, 2, 3] {
        let mut correct = 0;
        for &(parent, child, _) in &edges {
            let parent_pos = embeddings[parent];
            let parent_arr = array![parent_pos[0], parent_pos[1]];

            let mut distances_to_others: Vec<(&str, f64)> = all_nodes
                .iter()
                .filter(|&&n| n != parent)
                .map(|&n| {
                    let pos = embeddings[n];
                    let arr = array![pos[0], pos[1]];
                    let dist = ball.distance(&parent_arr.view(), &arr.view());
                    (n, dist)
                })
                .collect();

            distances_to_others.sort_by(|a, b| a.1.total_cmp(&b.1));

            let top_k: Vec<&str> = distances_to_others
                .iter()
                .take(k)
                .map(|(n, _)| *n)
                .collect();
            if top_k.contains(&child) {
                correct += 1;
            }
        }
        let accuracy = correct as f64 / edges.len() as f64 * 100.0;
        println!(
            "   Accuracy@{}: {:.1}% ({}/{})",
            k,
            accuracy,
            correct,
            edges.len()
        );
    }

    // Compare to Euclidean
    println!("\n5. Hyperbolic vs Euclidean Distance Comparison:");
    println!(
        "   {:20} {:>12} {:>12} {:>8}",
        "Pair", "Euclidean", "Hyperbolic", "Ratio"
    );
    println!("   {}", "-".repeat(55));

    let pairs = [
        ("animal", "mammal"), // parent-child (close)
        ("animal", "dog"),    // ancestor-descendant (far)
        ("dog", "wolf"),      // siblings (moderate)
        ("dog", "sparrow"),   // cross-branch (should be far)
    ];

    for (a, b) in pairs {
        let pos_a = embeddings[a];
        let pos_b = embeddings[b];
        let arr_a = array![pos_a[0], pos_a[1]];
        let arr_b = array![pos_b[0], pos_b[1]];

        let euc_dist = ((pos_a[0] - pos_b[0]).powi(2) + (pos_a[1] - pos_b[1]).powi(2)).sqrt();
        let hyp_dist = ball.distance(&arr_a.view(), &arr_b.view());

        println!(
            "   {} -> {:10} {:>12.3} {:>12.3} {:>7.1}x",
            a,
            b,
            euc_dist,
            hyp_dist,
            hyp_dist / euc_dist
        );
    }

    println!("\n   Key observation: Hyperbolic distance grows faster near boundary.");
    println!("   Cross-branch paths (dog->sparrow) are especially expanded.");

    // Summary
    println!("\n--- Summary ---");
    println!("Depth-distance correlation: {:.3} (should be > 0.9)", corr);
    println!(
        "Mean Rank: {:.2} (should be < 3.0 for good embedding)",
        mean_rank
    );
    println!("\nThis demonstrates why hyperbolic space is natural for hierarchies:");
    println!("- Low-dimensional (2D) representation captures tree structure");
    println!("- Distance encodes both similarity AND hierarchical depth");
    println!("- Cross-branch paths are naturally longer (go through common ancestor)");
}

/// Get depth of a node in the taxonomy
fn get_depth(node: &str, edges: &[(&str, &str, usize)]) -> usize {
    if node == "animal" {
        return 0;
    }
    for &(_, child, depth) in edges {
        if child == node {
            return depth;
        }
    }
    0
}

/// Compute Pearson correlation coefficient
fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
    let sum_x2: f64 = x.iter().map(|a| a * a).sum();
    let sum_y2: f64 = y.iter().map(|a| a * a).sum();

    let numerator = n * sum_xy - sum_x * sum_y;
    let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();

    if denominator.abs() < 1e-10 {
        0.0
    } else {
        numerator / denominator
    }
}
