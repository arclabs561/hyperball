//! Hyperbolic Hierarchy Recovery
//!
//! Demonstrates that hyperbolic space naturally encodes tree-like hierarchies
//! with much lower distortion than Euclidean space.
//!
//! Key insight: In hyperbolic space, volume grows exponentially with radius,
//! matching the exponential growth of nodes in a tree.
//!
//! ```bash
//! cargo run --example hierarchy_recovery --release
//! ```

use hyperball::PoincareBall;
use ndarray::{array, Array1};
use std::collections::HashMap;

fn main() {
    println!("Hyperbolic Hierarchy Recovery");
    println!("==============================\n");

    // Build a small taxonomy (subset of WordNet-style hierarchy)
    let taxonomy = build_taxonomy();
    println!(
        "Taxonomy: {} entities, {} edges\n",
        taxonomy.entities.len(),
        taxonomy.edges.len()
    );

    // Display hierarchy
    println!("Hierarchy structure:");
    print_hierarchy(&taxonomy, "entity", 0);
    println!();

    // Embed in hyperbolic space (2D Poincare ball, curvature c=1.0)
    let poincare = PoincareBall::<f64>::new(1.0);
    let embeddings = embed_hierarchy(&taxonomy, &poincare);

    // Analyze embedding quality
    println!("--- Embedding Analysis ---\n");

    // 1. Check that depth correlates with distance from origin
    analyze_depth_distance(&taxonomy, &embeddings, &poincare);

    // 2. Check parent-child distances vs sibling distances
    analyze_structural_distances(&taxonomy, &embeddings, &poincare);

    // 3. Measure hierarchy preservation (MAP for ancestor retrieval)
    let map = measure_hierarchy_preservation(&taxonomy, &embeddings, &poincare);
    println!("\nAncestor retrieval MAP: {:.3}", map);

    // 4. Compare with Euclidean embedding (2D)
    println!("\n--- Hyperbolic vs Euclidean ---");
    compare_with_euclidean(&taxonomy, &embeddings);
}

struct Taxonomy {
    entities: Vec<String>,
    edges: Vec<(usize, usize)>, // (parent_idx, child_idx)
    depth: HashMap<String, usize>,
    parent: HashMap<String, String>,
}

fn build_taxonomy() -> Taxonomy {
    // A small hierarchy mimicking WordNet structure
    let edges_raw = vec![
        ("entity", "physical_entity"),
        ("entity", "abstract_entity"),
        ("physical_entity", "object"),
        ("physical_entity", "living_thing"),
        ("living_thing", "organism"),
        ("organism", "animal"),
        ("organism", "plant"),
        ("animal", "mammal"),
        ("animal", "bird"),
        ("animal", "fish"),
        ("mammal", "dog"),
        ("mammal", "cat"),
        ("mammal", "human"),
        ("bird", "eagle"),
        ("bird", "sparrow"),
        ("plant", "tree"),
        ("plant", "flower"),
        ("tree", "oak"),
        ("tree", "pine"),
        ("abstract_entity", "concept"),
        ("abstract_entity", "relation"),
        ("concept", "quantity"),
        ("concept", "attribute"),
    ];

    let mut entities: Vec<String> = Vec::new();
    let mut entity_idx: HashMap<String, usize> = HashMap::new();
    let mut depth: HashMap<String, usize> = HashMap::new();
    let mut parent: HashMap<String, String> = HashMap::new();

    // Collect all entities
    for (p, c) in &edges_raw {
        for e in [*p, *c] {
            if !entity_idx.contains_key(e) {
                entity_idx.insert(e.to_string(), entities.len());
                entities.push(e.to_string());
            }
        }
    }

    // Build edges and compute depths
    let mut edges = Vec::new();
    for (p, c) in &edges_raw {
        let pi = entity_idx[*p];
        let ci = entity_idx[*c];
        edges.push((pi, ci));
        parent.insert(c.to_string(), p.to_string());
    }

    // Compute depths via BFS
    depth.insert("entity".to_string(), 0);
    let mut queue = vec!["entity".to_string()];
    while let Some(node) = queue.pop() {
        let d = depth[&node];
        for (p, c) in &edges_raw {
            if *p == node.as_str() && !depth.contains_key(*c) {
                depth.insert(c.to_string(), d + 1);
                queue.push(c.to_string());
            }
        }
    }

    Taxonomy {
        entities,
        edges,
        depth,
        parent,
    }
}

fn print_hierarchy(tax: &Taxonomy, node: &str, indent: usize) {
    println!(
        "{}{} (depth {})",
        "  ".repeat(indent),
        node,
        tax.depth.get(node).unwrap_or(&0)
    );
    for (pi, ci) in &tax.edges {
        if tax.entities[*pi] == node {
            print_hierarchy(tax, &tax.entities[*ci], indent + 1);
        }
    }
}

/// Embed hierarchy using Poincare embeddings.
/// Places root at origin, children at angles around parent.
fn embed_hierarchy(tax: &Taxonomy, poincare: &PoincareBall<f64>) -> HashMap<String, Array1<f64>> {
    let mut embeddings: HashMap<String, Array1<f64>> = HashMap::new();

    // Root at origin
    embeddings.insert("entity".to_string(), array![0.0, 0.0]);

    // BFS to place children
    let mut queue = vec!["entity".to_string()];
    let mut child_counts: HashMap<String, usize> = HashMap::new();

    while let Some(node) = queue.pop() {
        let node_emb = embeddings[&node].clone();
        let node_depth = tax.depth[&node];

        // Find children
        let children: Vec<_> = tax
            .edges
            .iter()
            .filter(|(pi, _)| tax.entities[*pi] == node)
            .map(|(_, ci)| tax.entities[*ci].clone())
            .collect();

        let n_children = children.len();
        for (i, child) in children.into_iter().enumerate() {
            // Place child at radius proportional to depth
            // Angle distributed around parent
            let radius = 0.3 + 0.12 * (node_depth + 1) as f64;
            let radius = radius.min(0.95); // Stay in ball

            let base_angle = if node == "entity" {
                0.0
            } else {
                // Get parent's angle
                let px = node_emb[0];
                let py = node_emb[1];
                py.atan2(px)
            };

            let angle_spread = std::f64::consts::PI / (n_children.max(1) as f64);
            let angle = base_angle + (i as f64 - (n_children - 1) as f64 / 2.0) * angle_spread;

            let child_emb = array![radius * angle.cos(), radius * angle.sin()];

            // Project to ensure we stay in ball
            let projected = poincare.project(&child_emb.view());
            embeddings.insert(child.clone(), projected);

            child_counts.insert(node.clone(), *child_counts.get(&node).unwrap_or(&0) + 1);
            queue.push(child);
        }
    }

    embeddings
}

fn analyze_depth_distance(
    tax: &Taxonomy,
    embeddings: &HashMap<String, Array1<f64>>,
    _poincare: &PoincareBall<f64>,
) {
    println!("Depth vs Distance from Origin:");
    println!("{:<20} {:>6} {:>10}", "Entity", "Depth", "||x||");
    println!("{}", "-".repeat(38));

    let mut by_depth: Vec<_> = tax
        .entities
        .iter()
        .map(|e| {
            let d = tax.depth.get(e).unwrap_or(&0);
            let emb = &embeddings[e];
            let norm = (emb.dot(emb)).sqrt();
            (e.clone(), *d, norm)
        })
        .collect();
    by_depth.sort_by_key(|(_, d, _)| *d);

    for (e, d, norm) in by_depth.iter().take(12) {
        println!("{:<20} {:>6} {:>10.4}", e, d, norm);
    }
    println!("  ... ({} total entities)", tax.entities.len());

    // Compute correlation
    let depths: Vec<f64> = by_depth.iter().map(|(_, d, _)| *d as f64).collect();
    let norms: Vec<f64> = by_depth.iter().map(|(_, _, n)| *n).collect();
    let corr = pearson_correlation(&depths, &norms);
    println!("\nCorrelation(depth, ||x||): {:.3}", corr);
    println!("  (Positive correlation indicates hierarchy encoded in radius)");
}

fn analyze_structural_distances(
    tax: &Taxonomy,
    embeddings: &HashMap<String, Array1<f64>>,
    poincare: &PoincareBall<f64>,
) {
    println!("\nParent-Child vs Sibling Distances:");

    let mut parent_child_dists = Vec::new();
    let mut sibling_dists = Vec::new();

    // Parent-child distances
    for (pi, ci) in &tax.edges {
        let p_emb = &embeddings[&tax.entities[*pi]];
        let c_emb = &embeddings[&tax.entities[*ci]];
        let d = poincare.distance(&p_emb.view(), &c_emb.view());
        parent_child_dists.push(d);
    }

    // Sibling distances (nodes with same parent)
    for (p1, c1) in &tax.edges {
        for (p2, c2) in &tax.edges {
            if p1 == p2 && c1 < c2 {
                let e1 = &embeddings[&tax.entities[*c1]];
                let e2 = &embeddings[&tax.entities[*c2]];
                let d = poincare.distance(&e1.view(), &e2.view());
                sibling_dists.push(d);
            }
        }
    }

    let avg_pc = parent_child_dists.iter().sum::<f64>() / parent_child_dists.len() as f64;
    let avg_sib = sibling_dists.iter().sum::<f64>() / sibling_dists.len() as f64;

    println!("  Avg parent-child distance: {:.3}", avg_pc);
    println!("  Avg sibling distance:      {:.3}", avg_sib);
    println!("  Ratio (sib/pc):            {:.2}x", avg_sib / avg_pc);
    println!("  (Siblings should be farther apart than parent-child)");
}

fn measure_hierarchy_preservation(
    tax: &Taxonomy,
    embeddings: &HashMap<String, Array1<f64>>,
    poincare: &PoincareBall<f64>,
) -> f64 {
    // For each entity, retrieve nearest neighbors by hyperbolic distance
    // Measure if ancestors appear before non-ancestors

    let mut ap_sum = 0.0;
    let mut count = 0;

    for entity in &tax.entities {
        if entity == "entity" {
            continue;
        } // Root has no ancestors

        // Get ancestors
        let mut ancestors = std::collections::HashSet::new();
        let mut current = entity.clone();
        while let Some(p) = tax.parent.get(&current) {
            ancestors.insert(p.clone());
            current = p.clone();
        }

        if ancestors.is_empty() {
            continue;
        }

        // Rank all entities by distance
        let entity_emb = &embeddings[entity];
        let mut distances: Vec<_> = tax
            .entities
            .iter()
            .filter(|e| *e != entity)
            .map(|e| {
                let d = poincare.distance(&entity_emb.view(), &embeddings[e].view());
                (e.clone(), d)
            })
            .collect();
        distances.sort_by(|a, b| a.1.total_cmp(&b.1));

        // Compute AP for ancestor retrieval
        let mut relevant_found = 0;
        let mut precision_sum = 0.0;
        for (rank, (e, _)) in distances.iter().enumerate() {
            if ancestors.contains(e) {
                relevant_found += 1;
                precision_sum += relevant_found as f64 / (rank + 1) as f64;
            }
        }
        let ap = if !ancestors.is_empty() {
            precision_sum / ancestors.len() as f64
        } else {
            0.0
        };
        ap_sum += ap;
        count += 1;
    }

    ap_sum / count as f64
}

fn compare_with_euclidean(tax: &Taxonomy, hyp_embeddings: &HashMap<String, Array1<f64>>) {
    // Euclidean baseline: spread nodes by BFS depth (y) and sibling index (x).
    // This is a simple MDS-like layout -- not optimized, but uses the same
    // structural information as the hyperbolic embedding (depth + sibling order).
    let mut euc_embeddings: HashMap<String, (f64, f64)> = HashMap::new();

    // BFS to assign (x, y) positions: y = depth, x = centered sibling index.
    euc_embeddings.insert("entity".to_string(), (0.0, 0.0));
    let mut queue = vec!["entity".to_string()];

    while let Some(node) = queue.pop() {
        let d = *tax.depth.get(&node).unwrap_or(&0) as f64;

        let children: Vec<_> = tax
            .edges
            .iter()
            .filter(|(pi, _)| tax.entities[*pi] == node)
            .map(|(_, ci)| tax.entities[*ci].clone())
            .collect();

        let n_children = children.len();
        let parent_x = euc_embeddings.get(&node).map(|(x, _)| *x).unwrap_or(0.0);

        // Spread children symmetrically around parent's x, with width shrinking by depth.
        let spread = 2.0 / (d + 1.0).powi(2);
        for (i, child) in children.iter().enumerate() {
            let offset = if n_children <= 1 {
                0.0
            } else {
                (i as f64 - (n_children - 1) as f64 / 2.0) * spread
            };
            let child_x = parent_x + offset;
            let child_y = d + 1.0;
            euc_embeddings.insert(child.clone(), (child_x, child_y));
            queue.push(child.clone());
        }
    }

    // Measure distortion
    let mut hyp_distortion = 0.0;
    let mut euc_distortion = 0.0;
    let mut count = 0;

    let poincare = PoincareBall::new(1.0);

    for (pi, ci) in &tax.edges {
        let p = &tax.entities[*pi];
        let c = &tax.entities[*ci];

        // True graph distance = 1 (direct edge)
        let true_dist = 1.0;

        // Hyperbolic distance
        let hyp_dist = poincare.distance(&hyp_embeddings[p].view(), &hyp_embeddings[c].view());

        // Euclidean distance
        let (px, py) = euc_embeddings[p];
        let (cx, cy) = euc_embeddings[c];
        let euc_dist = ((px - cx).powi(2) + (py - cy).powi(2)).sqrt();

        // Distortion = |embedding_dist / true_dist - 1|
        hyp_distortion += (hyp_dist - true_dist).abs();
        euc_distortion += (euc_dist - true_dist).abs();
        count += 1;
    }

    println!("\nEmbedding Distortion (lower is better):");
    println!("  Hyperbolic (2D): {:.3}", hyp_distortion / count as f64);
    println!("  Euclidean (2D):  {:.3}", euc_distortion / count as f64);
    println!("\n  Note: the Euclidean baseline uses a BFS-depth + sibling-index layout.");
    println!("  Neither embedding is optimized; a fair comparison would optimize each");
    println!("  for its own geometry.");
}

fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;

    let mut num = 0.0;
    let mut dx2 = 0.0;
    let mut dy2 = 0.0;

    for (xi, yi) in x.iter().zip(y.iter()) {
        let dx = xi - mx;
        let dy = yi - my;
        num += dx * dy;
        dx2 += dx * dx;
        dy2 += dy * dy;
    }

    num / (dx2.sqrt() * dy2.sqrt())
}
