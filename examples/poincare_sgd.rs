//! Learned Poincare Embedding via Riemannian SGD
//!
//! Embeds a small tree (25 nodes, 3 levels) into the 2D Poincare ball by
//! minimizing a hyperbolic distance loss.  Each epoch:
//!   1. For each edge (u,v), compute hyperbolic distance and loss.
//!   2. Compute the Euclidean gradient of the loss.
//!   3. Convert to Riemannian gradient: g_R = g_E / (conformal_factor)^2.
//!   4. Retract via gradient step and project back into the ball.
//!
//! After training, verifies that embedded distances respect tree structure:
//! parent-child distances should be smaller than cross-branch distances.
//!
//! Reference: Nickel & Kiela (2017), "Poincare Embeddings for Learning
//! Hierarchical Representations" (Sections 3--4).
//!
//! ```bash
//! cargo run --example poincare_sgd --release
//! ```

use hyperball::PoincareBall;
use ndarray::{array, Array1};

// ---- Deterministic LCG (avoids external rand for examples) ----

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Uniform in [-bound, bound).
    fn next_signed(&mut self, bound: f64) -> f64 {
        self.next_f64() * 2.0 * bound - bound
    }
}

// ---- Tree construction ----

/// Build a balanced ternary tree: root (0) has 3 children, each child has
/// 3 children, each grandchild has ~2 children.  Total: 1 + 3 + 9 + ~12 = 25.
fn build_tree() -> (usize, Vec<(usize, usize)>) {
    let mut edges = Vec::new();
    let mut next_id = 1usize;

    // Level 1: root -> 3 children
    let level1: Vec<usize> = (0..3)
        .map(|_| {
            let id = next_id;
            edges.push((0, id));
            next_id += 1;
            id
        })
        .collect();

    // Level 2: each level-1 node -> 3 children
    let mut level2 = Vec::new();
    for &parent in &level1 {
        for _ in 0..3 {
            let id = next_id;
            edges.push((parent, id));
            next_id += 1;
            level2.push(id);
        }
    }

    // Level 3: each level-2 node -> 1 or 2 children (to get ~25 total)
    let children_per = [2, 1, 2, 1, 2, 1, 2, 1, 1];
    for (i, &parent) in level2.iter().enumerate() {
        let nc = children_per.get(i).copied().unwrap_or(1);
        for _ in 0..nc {
            let id = next_id;
            edges.push((parent, id));
            next_id += 1;
        }
    }

    (next_id, edges)
}

/// Shortest-path distance in the tree between two nodes (BFS).
fn tree_distance(edges: &[(usize, usize)], n: usize, a: usize, b: usize) -> usize {
    if a == b {
        return 0;
    }
    // Build adjacency list
    let mut adj = vec![vec![]; n];
    for &(u, v) in edges {
        adj[u].push(v);
        adj[v].push(u);
    }
    let mut dist = vec![usize::MAX; n];
    dist[a] = 0;
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(a);
    while let Some(u) = queue.pop_front() {
        if u == b {
            return dist[u];
        }
        for &v in &adj[u] {
            if dist[v] == usize::MAX {
                dist[v] = dist[u] + 1;
                queue.push_back(v);
            }
        }
    }
    usize::MAX
}

// ---- Poincare SGD ----

/// Conformal factor: lambda(x) = 2 / (1 - ||x||^2).
fn conformal_factor(x: &Array1<f64>) -> f64 {
    2.0 / (1.0 - x.dot(x))
}

/// Euclidean gradient of d_hyp(u, v) w.r.t. u.
///
/// d = (2/sqrt(c)) * arctanh(sqrt(c) * || -u (+) v ||)
///
/// We use the chain rule through mobius_add and arctanh.  For c=1:
///   grad_u d = (4 / ((1 - ||u||^2) * sqrt(alpha) * (1 - alpha))) *
///              ((||v||^2 - 2<u,v> + 1)*u - (1 - ||u||^2)*v)
/// where alpha = || -u (+) v ||^2.
fn grad_distance_wrt_u(ball: &PoincareBall<f64>, u: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
    let neg_u = u.mapv(|x| -x);
    let diff = ball.mobius_add(&neg_u.view(), &v.view());
    let alpha = diff.dot(&diff); // || -u (+) v ||^2

    if alpha < 1e-12 {
        return Array1::zeros(u.len());
    }

    let u_sq = u.dot(u);
    let v_sq = v.dot(v);
    let uv = u.dot(v);

    let factor_u = v_sq - 2.0 * uv + 1.0;
    let factor_v = 1.0 - u_sq;

    let numerator = u.mapv(|ui| ui * factor_u) - v.mapv(|vi| vi * factor_v);

    let denom = (1.0 - u_sq) * alpha.sqrt() * (1.0 - alpha).max(1e-12);
    let scale = 4.0 / denom;

    numerator * scale
}

fn main() {
    let ball = PoincareBall::<f64>::new(1.0);
    let (n, edges) = build_tree();

    println!("Poincare SGD: learning embeddings for a tree");
    println!("=============================================\n");
    println!("Tree: {} nodes, {} edges, 3 levels deep\n", n, edges.len());

    // Initialize embeddings: small random points near the origin.
    let mut rng = Lcg::new(42);
    let mut emb: Vec<Array1<f64>> = (0..n)
        .map(|_| array![rng.next_signed(0.01), rng.next_signed(0.01)])
        .collect();

    let lr = 0.05;
    let epochs = 300;

    println!("{:>6}  {:>12}  {:>12}", "epoch", "mean_loss", "max_||x||");
    println!("{:-<6}  {:-<12}  {:-<12}", "", "", "");

    for epoch in 0..epochs {
        let mut total_loss = 0.0;

        for &(u, v) in &edges {
            let d = ball.distance(&emb[u].view(), &emb[v].view());
            // Loss = d^2 (pull connected nodes together)
            let loss = d * d;
            total_loss += loss;

            // Euclidean gradients: grad_u(d^2) = 2d * grad_u(d)
            let g_u_euc = grad_distance_wrt_u(&ball, &emb[u], &emb[v]) * (2.0 * d);
            let g_v_euc = grad_distance_wrt_u(&ball, &emb[v], &emb[u]) * (2.0 * d);

            // Riemannian gradient = Euclidean gradient / lambda(x)^2
            let lambda_u = conformal_factor(&emb[u]);
            let lambda_v = conformal_factor(&emb[v]);

            let g_u_riem = g_u_euc / (lambda_u * lambda_u);
            let g_v_riem = g_v_euc / (lambda_v * lambda_v);

            // Gradient step
            emb[u] = &emb[u] - &(g_u_riem * lr);
            emb[v] = &emb[v] - &(g_v_riem * lr);

            // Project back into the ball
            emb[u] = ball.project(&emb[u].view());
            emb[v] = ball.project(&emb[v].view());
        }

        // Also add a repulsive term for non-edges (sample a few)
        // to prevent collapse. Push non-neighbors apart.
        let mut rng_repulse = Lcg::new(epoch as u64);
        let n_neg = edges.len(); // same number of negative samples
        for _ in 0..n_neg {
            let u = (rng_repulse.next_f64() * n as f64) as usize % n;
            let v = (rng_repulse.next_f64() * n as f64) as usize % n;
            if u == v {
                continue;
            }
            // Skip if (u,v) is actually an edge
            if edges
                .iter()
                .any(|&(a, b)| (a == u && b == v) || (a == v && b == u))
            {
                continue;
            }

            let d = ball.distance(&emb[u].view(), &emb[v].view());
            if d < 1e-8 {
                continue;
            }
            // Loss = max(0, margin - d)^2 with margin = 2.0
            let margin = 2.0;
            let violation = margin - d;
            if violation <= 0.0 {
                continue;
            }
            let neg_loss = violation * violation;
            total_loss += neg_loss;

            // grad_u(margin - d)^2 = -2*(margin - d) * grad_u(d)
            let g_u_euc = grad_distance_wrt_u(&ball, &emb[u], &emb[v]) * (-2.0 * violation);
            let g_v_euc = grad_distance_wrt_u(&ball, &emb[v], &emb[u]) * (-2.0 * violation);

            let lambda_u = conformal_factor(&emb[u]);
            let lambda_v = conformal_factor(&emb[v]);

            let g_u_riem = g_u_euc / (lambda_u * lambda_u);
            let g_v_riem = g_v_euc / (lambda_v * lambda_v);

            emb[u] = &emb[u] - &(g_u_riem * lr);
            emb[v] = &emb[v] - &(g_v_riem * lr);

            emb[u] = ball.project(&emb[u].view());
            emb[v] = ball.project(&emb[v].view());
        }

        if epoch % 50 == 0 || epoch == epochs - 1 {
            let mean_loss = total_loss / (2 * edges.len()) as f64;
            let max_norm = emb.iter().map(|e| e.dot(e).sqrt()).fold(0.0f64, f64::max);
            println!("{:>6}  {:>12.6}  {:>12.4}", epoch, mean_loss, max_norm);
        }
    }

    // Evaluate: compare embedded distances vs tree distances.
    println!("\n--- Evaluation ---\n");

    // Parent-child distances
    let mut pc_dists = Vec::new();
    for &(u, v) in &edges {
        let d = ball.distance(&emb[u].view(), &emb[v].view());
        pc_dists.push(d);
    }
    let mean_pc = pc_dists.iter().sum::<f64>() / pc_dists.len() as f64;

    // Cross-branch distances (sample: leaf pairs from different subtrees)
    // Leaves are nodes with no children.
    let parents: std::collections::HashSet<usize> = edges.iter().map(|&(u, _)| u).collect();
    let leaves: Vec<usize> = (0..n).filter(|i| !parents.contains(i)).collect();

    let mut cross_dists = Vec::new();
    for i in 0..leaves.len() {
        for j in (i + 1)..leaves.len() {
            let td = tree_distance(&edges, n, leaves[i], leaves[j]);
            if td >= 4 {
                // Different subtrees
                let d = ball.distance(&emb[leaves[i]].view(), &emb[leaves[j]].view());
                cross_dists.push(d);
            }
        }
    }
    let mean_cross = if cross_dists.is_empty() {
        f64::NAN
    } else {
        cross_dists.iter().sum::<f64>() / cross_dists.len() as f64
    };

    println!("Mean parent-child distance:  {:.4}", mean_pc);
    println!("Mean cross-branch distance:  {:.4}", mean_cross);
    println!("Ratio (cross/parent-child):  {:.2}x", mean_cross / mean_pc);
    println!();

    // Depth vs norm (deeper nodes should be further from origin)
    println!("Depth vs norm from origin:");
    println!("{:>6}  {:>10}  {:>10}", "depth", "mean_norm", "count");
    println!("{:-<6}  {:-<10}  {:-<10}", "", "", "");
    for depth in 0..=3 {
        let norms: Vec<f64> = (0..n)
            .filter(|&i| tree_distance(&edges, n, 0, i) == depth)
            .map(|i| emb[i].dot(&emb[i]).sqrt())
            .collect();
        if !norms.is_empty() {
            let mean = norms.iter().sum::<f64>() / norms.len() as f64;
            println!("{:>6}  {:>10.4}  {:>10}", depth, mean, norms.len());
        }
    }
    println!();
    println!("Deeper nodes should have larger norms (pushed toward boundary).");
    println!("Cross-branch distances should exceed parent-child distances.");
}
