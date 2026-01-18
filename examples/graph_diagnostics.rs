//! Graph → distance matrix → “tree-likeness” diagnostics.
//!
//! This is a small-n diagnostic harness, meant to answer:
//! - is the *metric* induced by shortest-path distances plausibly tree-like?
//! - if it’s tree-like, hyperbolic embeddings are often a good geometric prior.
//!
//! Two checks:
//! - **δ-hyperbolicity** (4-point condition): trees have δ = 0.
//! - **ultrametric violation**: ultrametrics are “strongly tree-like” (hierarchical clustering).
//!
//! Distances here are shortest-path distances on an unweighted graph.

use hyp::core::diagnostics;

fn main() {
    // 1) A tree metric: path distances on a tree are 0-hyperbolic.
    let tree = make_path_graph(8);
    let d_tree = all_pairs_shortest_path(&tree);
    let n_tree = tree.len();
    let delta_tree = diagnostics::delta_hyperbolicity_four_point_exact_f64(&d_tree, n_tree);
    let um_tree = diagnostics::ultrametric_max_violation_f64(&d_tree, n_tree);

    println!("tree (path graph) n={n_tree}");
    println!("  δ (4-point exact) = {delta_tree}");
    println!("  ultrametric max violation = {um_tree}");
    println!();

    // 2) A cycle: C4 has δ=1 (see unit test), and larger cycles are “less tree-like”.
    let c = make_cycle_graph(8);
    let d_c = all_pairs_shortest_path(&c);
    let n_c = c.len();
    let delta_c = diagnostics::delta_hyperbolicity_four_point_exact_f64(&d_c, n_c);
    let um_c = diagnostics::ultrametric_max_violation_f64(&d_c, n_c);

    println!("cycle n={n_c}");
    println!("  δ (4-point exact) = {delta_c}");
    println!("  ultrametric max violation = {um_c}");
    println!();

    // 3) A true ultrametric (not from a graph): two tight clusters far apart.
    let n_u = 6usize;
    let mut d_u = vec![0.0f64; n_u * n_u];
    let set = |d: &mut [f64], n: usize, i: usize, j: usize, v: f64| {
        d[i * n + j] = v;
        d[j * n + i] = v;
    };
    // clusters: (0,1,2) and (3,4,5)
    for (i, j) in [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)] {
        set(&mut d_u, n_u, i, j, 1.0);
    }
    for i in 0..3 {
        for j in 3..6 {
            set(&mut d_u, n_u, i, j, 3.0);
        }
    }
    let delta_u = diagnostics::delta_hyperbolicity_four_point_exact_f64(&d_u, n_u);
    let um_u = diagnostics::ultrametric_max_violation_f64(&d_u, n_u);
    println!("synthetic ultrametric n={n_u}");
    println!("  δ (4-point exact) = {delta_u}");
    println!("  ultrametric max violation = {um_u}");
}

fn make_path_graph(n: usize) -> Vec<Vec<usize>> {
    let mut g = vec![Vec::new(); n];
    for i in 0..n {
        if i > 0 {
            g[i].push(i - 1);
        }
        if i + 1 < n {
            g[i].push(i + 1);
        }
    }
    g
}

fn make_cycle_graph(n: usize) -> Vec<Vec<usize>> {
    let mut g = vec![Vec::new(); n];
    for i in 0..n {
        let prev = if i == 0 { n - 1 } else { i - 1 };
        let next = (i + 1) % n;
        g[i].push(prev);
        g[i].push(next);
    }
    g
}

fn all_pairs_shortest_path(g: &[Vec<usize>]) -> Vec<f64> {
    let n = g.len();
    let mut dist = vec![0.0f64; n * n];
    for s in 0..n {
        let d = bfs(g, s);
        for t in 0..n {
            dist[s * n + t] = d[t] as f64;
        }
    }
    dist
}

fn bfs(g: &[Vec<usize>], start: usize) -> Vec<usize> {
    use std::collections::VecDeque;
    let n = g.len();
    let mut dist = vec![usize::MAX; n];
    let mut q = VecDeque::new();
    dist[start] = 0;
    q.push_back(start);
    while let Some(u) = q.pop_front() {
        let du = dist[u];
        for &v in &g[u] {
            if dist[v] == usize::MAX {
                dist[v] = du + 1;
                q.push_back(v);
            }
        }
    }
    // This example only uses connected graphs; keep it simple.
    for (i, &d) in dist.iter().enumerate() {
        assert!(d != usize::MAX, "graph disconnected (unreachable node {i})");
    }
    dist
}

