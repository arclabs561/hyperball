//! Distortion vs. Dimension: Euclidean vs. Poincare
//!
//! Demonstrates the key insight from Nickel & Kiela (2017): hyperbolic space
//! can represent hierarchies in far fewer dimensions than Euclidean space.
//!
//! We generate a complete binary tree, embed it in both Euclidean and Poincare
//! spaces at various dimensions, and compare how well pairwise graph distances
//! are preserved. The punchline: Poincare@5 can match or beat Euclidean@50.
//!
//! For improved embedding strategies (GPU-compatible, lower distortion
//! bounds), see van Spengler & Mettes (2025), "Low-distortion and
//! GPU-compatible Tree Embeddings in Hyperbolic Space."  The deterministic
//! path-based embedding used here is a simplified variant of the Sarkar
//! (2011) construction.
//!
//! Embedding strategy (same for both geometries):
//! - Each node's position is determined by its root-path (sequence of
//!   left/right choices). At each tree level, the left/right choice adds
//!   +/- offset along a dimension (cycling through available dims).
//! - For Euclidean: positions are raw coordinates; distance is L2.
//! - For Poincare: positions are tangent vectors at the origin, mapped onto
//!   the ball via exp_map_zero; distance is the hyperbolic metric.
//!
//! In low dimensions, the cycling causes axis collisions that confuse the
//! Euclidean metric. The Poincare metric, with its exponentially growing
//! volume, keeps branches separated even with collisions.
//!
//! ```bash
//! cargo run -p hyperball --example distortion_vs_dimension --release
//! ```

use hyperball::PoincareBall;
use ndarray::Array1;

// ---------------------------------------------------------------------------
// Deterministic LCG (avoids external rand dependency for examples)
// ---------------------------------------------------------------------------

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }

    /// Uniform f64 in [-1, 1).
    fn next_f64_signed(&mut self) -> f64 {
        let u = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        u * 2.0 - 1.0
    }
}

// ---------------------------------------------------------------------------
// Binary tree
// ---------------------------------------------------------------------------

struct BinaryTree {
    edges: Vec<Vec<usize>>,
    /// Depth of each node (root = 0). Used indirectly via root_path().len().
    #[allow(dead_code)]
    depth: Vec<usize>,
    num_nodes: usize,
    max_depth: usize,
}

impl BinaryTree {
    fn new(max_depth: usize) -> Self {
        let num_nodes = (1 << (max_depth + 1)) - 1;
        let mut edges = vec![vec![]; num_nodes];
        let mut depth = vec![0usize; num_nodes];

        for i in 0..num_nodes {
            let left = 2 * i + 1;
            let right = 2 * i + 2;
            if left < num_nodes {
                edges[i].push(left);
                edges[left].push(i);
                depth[left] = depth[i] + 1;
            }
            if right < num_nodes {
                edges[i].push(right);
                edges[right].push(i);
                depth[right] = depth[i] + 1;
            }
        }

        Self {
            edges,
            depth,
            num_nodes,
            max_depth,
        }
    }

    /// All-pairs shortest-path distances (BFS). Flat n*n matrix.
    fn all_pairs_distances(&self) -> Vec<usize> {
        let n = self.num_nodes;
        let mut dist = vec![usize::MAX; n * n];
        for src in 0..n {
            let mut queue = std::collections::VecDeque::new();
            dist[src * n + src] = 0;
            queue.push_back(src);
            while let Some(u) = queue.pop_front() {
                for &v in &self.edges[u] {
                    if dist[src * n + v] == usize::MAX {
                        dist[src * n + v] = dist[src * n + u] + 1;
                        queue.push_back(v);
                    }
                }
            }
        }
        dist
    }
}

// ---------------------------------------------------------------------------
// Embedding helpers
// ---------------------------------------------------------------------------

/// Root-path of node i: sequence of left(false)/right(true) choices.
fn root_path(i: usize) -> Vec<bool> {
    let mut path = Vec::new();
    let mut cur = i;
    while cur > 0 {
        path.push(cur % 2 == 0); // even = right child
        cur = (cur - 1) / 2;
    }
    path.reverse();
    path
}

/// Build a direction vector from a root-path. Each level contributes +/-1
/// along a cycling axis, with magnitude halving per level (like a binary
/// expansion). When dim >= tree_depth, every level gets its own axis and
/// branches never interfere. When dim < tree_depth, axes are reused and
/// different branches can collide.
fn direction_from_path(path: &[bool], dim: usize) -> Array1<f64> {
    let mut v: Array1<f64> = Array1::zeros(dim);
    for (level, &went_right) in path.iter().enumerate() {
        let axis = level % dim;
        let sign = if went_right { 1.0 } else { -1.0 };
        v[axis] += sign / (1 << (level + 1)) as f64;
    }
    v
}

// ---------------------------------------------------------------------------
// Euclidean embedding
// ---------------------------------------------------------------------------

/// Embed tree in Euclidean space. Positions come from direction_from_path,
/// globally rescaled so that mean embedding distance equals mean graph distance.
fn embed_euclidean(
    tree: &BinaryTree,
    dim: usize,
    graph_dist: &[usize],
    rng: &mut Lcg,
) -> Vec<Array1<f64>> {
    let n = tree.num_nodes;
    let mut coords: Vec<Array1<f64>> = vec![Array1::zeros(dim); n];

    for i in 1..n {
        let path = root_path(i);
        let mut pos = direction_from_path(&path, dim);
        // Tiny noise to break exact ties
        for d in 0..dim {
            pos[d] += rng.next_f64_signed() * 1e-6;
        }
        coords[i] = pos;
    }

    // Rescale so mean embedding distance matches mean graph distance
    let (mean_embed, mean_graph) = mean_distances_euclidean(&coords, graph_dist, n);
    if mean_embed > 1e-12 {
        let scale = mean_graph / mean_embed;
        for c in coords.iter_mut() {
            *c = c.mapv(|v| v * scale);
        }
    }

    coords
}

fn mean_distances_euclidean(coords: &[Array1<f64>], graph_dist: &[usize], n: usize) -> (f64, f64) {
    let mut sum_e = 0.0;
    let mut sum_g = 0.0;
    let mut count = 0u64;
    for i in 0..n {
        for j in (i + 1)..n {
            let diff = &coords[i] - &coords[j];
            sum_e += diff.dot(&diff).sqrt();
            sum_g += graph_dist[i * n + j] as f64;
            count += 1;
        }
    }
    (sum_e / count as f64, sum_g / count as f64)
}

// ---------------------------------------------------------------------------
// Poincare embedding
// ---------------------------------------------------------------------------

/// Embed tree in Poincare ball. Same direction logic as Euclidean, but:
/// 1. The direction is normalized and scaled by a depth-dependent radius.
/// 2. The tangent vector is mapped onto the ball via exp_map_zero.
/// 3. We grid-search over a scale parameter to minimize distortion.
///
/// The scale search is needed because the nonlinear exp_map means the
/// optimal tangent-space magnitude is not obvious a priori.
fn embed_poincare(
    tree: &BinaryTree,
    dim: usize,
    ball: &PoincareBall<f64>,
    graph_dist: &[usize],
    rng: &mut Lcg,
) -> Vec<Array1<f64>> {
    let n = tree.num_nodes;

    // Grid-search over scale parameter
    let mut best_distortion = f64::MAX;
    let mut best_scale = 1.0;

    for s_int in 1..=40 {
        let s = s_int as f64 * 0.25;
        let coords = poincare_embed_with_scale(tree, dim, ball, s, &mut Lcg::new(rng.0));
        let hyp_dist = pairwise_poincare(&coords, ball);
        let d = compute_distortion(graph_dist, &hyp_dist, n);
        if d < best_distortion {
            best_distortion = d;
            best_scale = s;
        }
    }

    // Final embedding at best scale
    poincare_embed_with_scale(tree, dim, ball, best_scale, rng)
}

fn poincare_embed_with_scale(
    tree: &BinaryTree,
    dim: usize,
    ball: &PoincareBall<f64>,
    scale: f64,
    rng: &mut Lcg,
) -> Vec<Array1<f64>> {
    let n = tree.num_nodes;
    let max_d = tree.max_depth as f64;
    let mut coords: Vec<Array1<f64>> = vec![Array1::zeros(dim); n];

    for i in 1..n {
        let path = root_path(i);
        let depth = path.len() as f64;

        let mut dir = direction_from_path(&path, dim);
        let norm = dir.dot(&dir).sqrt();
        if norm > 1e-12 {
            dir = dir / norm;
        }

        // Radius grows with depth; scale controls overall magnitude
        let r = scale * depth / max_d;
        let mut tangent = dir * r;

        for d in 0..dim {
            tangent[d] += rng.next_f64_signed() * 1e-6;
        }

        let point = ball.exp_map_zero(&tangent.view());
        coords[i] = ball.project(&point.view());
    }

    coords
}

// ---------------------------------------------------------------------------
// Distortion metric
// ---------------------------------------------------------------------------

/// Mean absolute relative error: (1/|P|) sum |d_embed/d_graph - 1|.
/// Lower is better; 0 = perfect isometry.
fn compute_distortion(graph_dist: &[usize], embed_dist: &[f64], n: usize) -> f64 {
    let mut total = 0.0;
    let mut count = 0u64;
    for i in 0..n {
        for j in (i + 1)..n {
            let gd = graph_dist[i * n + j] as f64;
            let ed = embed_dist[i * n + j];
            if gd > 0.0 {
                total += (ed / gd - 1.0).abs();
                count += 1;
            }
        }
    }
    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

fn pairwise_euclidean(points: &[Array1<f64>]) -> Vec<f64> {
    let n = points.len();
    let mut dist = vec![0.0f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let diff = &points[i] - &points[j];
            let d = diff.dot(&diff).sqrt();
            dist[i * n + j] = d;
            dist[j * n + i] = d;
        }
    }
    dist
}

fn pairwise_poincare(points: &[Array1<f64>], ball: &PoincareBall<f64>) -> Vec<f64> {
    let n = points.len();
    let mut dist = vec![0.0f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = ball.distance(&points[i].view(), &points[j].view());
            dist[i * n + j] = d;
            dist[j * n + i] = d;
        }
    }
    dist
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    println!("Distortion vs. Dimension: Euclidean vs. Poincare Ball");
    println!("======================================================");
    println!();
    println!("Insight (Nickel & Kiela, 2017): hyperbolic space can represent");
    println!("hierarchies in far fewer dimensions than Euclidean space.");
    println!();

    let tree_depth = 5;
    let tree = BinaryTree::new(tree_depth);
    let graph_dist = tree.all_pairs_distances();
    let ball = PoincareBall::<f64>::new(1.0);

    println!(
        "Binary tree: depth={}, nodes={}\n",
        tree_depth, tree.num_nodes
    );

    let dimensions = [2, 5, 10, 50];

    println!(
        "{:>5}  {:>18}  {:>18}  {:>10}",
        "dim", "Euclid. distort.", "Poincare distort.", "Euc/Poinc"
    );
    println!("{:-<5}  {:-<18}  {:-<18}  {:-<10}", "", "", "", "");

    for &dim in &dimensions {
        let mut rng_euc = Lcg::new(42 + dim as u64);
        let mut rng_hyp = Lcg::new(42 + dim as u64);

        let euc_coords = embed_euclidean(&tree, dim, &graph_dist, &mut rng_euc);
        let hyp_coords = embed_poincare(&tree, dim, &ball, &graph_dist, &mut rng_hyp);

        let euc_dist = pairwise_euclidean(&euc_coords);
        let hyp_dist = pairwise_poincare(&hyp_coords, &ball);

        let euc_d = compute_distortion(&graph_dist, &euc_dist, tree.num_nodes);
        let hyp_d = compute_distortion(&graph_dist, &hyp_dist, tree.num_nodes);

        let ratio = euc_d / hyp_d;

        println!(
            "{:>5}  {:>18.4}  {:>18.4}  {:>9.2}x",
            dim, euc_d, hyp_d, ratio
        );
    }

    println!();
    println!("Distortion = mean |d_embed/d_graph - 1| over all node pairs.");
    println!("Lower is better (0 = perfect isometry).");
    println!();
    println!("Key observations:");
    println!("  - Poincare consistently achieves lower distortion than Euclidean.");
    println!("  - Poincare@2-5 matches or beats Euclidean@50, because hyperbolic");
    println!("    volume grows exponentially with radius, just like tree branching.");
    println!("  - Euclidean improves slowly with dimension: it needs ~1 axis per");
    println!("    tree level to avoid collisions between branches.");
}
