//! Learned Lorentz Embedding via Riemannian SGD
//!
//! Embeds a 25-node tree (3 levels) into the Lorentz (hyperboloid) model.
//! Each epoch: compute distance loss, Euclidean gradient, convert to
//! Riemannian gradient via Minkowski metric (not conformal factor), retract
//! with exp_map.
//!
//! Reference: Nickel & Kiela (2018), "Learning Continuous Hierarchies in
//! the Lorentz Model of Hyperbolic Space".
//!
//! ```bash
//! cargo run --example lorentz_sgd --release
//! ```

use hyperball::LorentzModel;
use ndarray::Array1;
use std::collections::{HashSet, VecDeque};

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
    fn next_signed(&mut self, b: f64) -> f64 {
        self.next_f64() * 2.0 * b - b
    }
}

fn build_tree() -> (usize, Vec<(usize, usize)>) {
    let mut edges = Vec::new();
    let mut id = 1usize;
    let l1: Vec<_> = (0..3)
        .map(|_| {
            edges.push((0, id));
            id += 1;
            id - 1
        })
        .collect();
    let mut l2 = Vec::new();
    for &p in &l1 {
        for _ in 0..3 {
            edges.push((p, id));
            l2.push(id);
            id += 1;
        }
    }
    for (i, &p) in l2.iter().enumerate() {
        for _ in 0..[2, 1, 2, 1, 2, 1, 2, 1, 1][i] {
            edges.push((p, id));
            id += 1;
        }
    }
    (id, edges)
}

fn tree_dist(edges: &[(usize, usize)], n: usize, a: usize, b: usize) -> usize {
    if a == b {
        return 0;
    }
    let mut adj = vec![vec![]; n];
    for &(u, v) in edges {
        adj[u].push(v);
        adj[v].push(u);
    }
    let mut d = vec![usize::MAX; n];
    d[a] = 0;
    let mut q = VecDeque::new();
    q.push_back(a);
    while let Some(u) = q.pop_front() {
        if u == b {
            return d[u];
        }
        for &v in &adj[u] {
            if d[v] == usize::MAX {
                d[v] = d[u] + 1;
                q.push_back(v);
            }
        }
    }
    usize::MAX
}

/// Euclidean gradient of d(u,v)^2 w.r.t. u.
fn grad_dsq(m: &LorentzModel<f64>, u: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
    let d = m.distance(&u.view(), &v.view());
    if d < 1e-12 {
        return Array1::zeros(u.len());
    }
    let arg = (-m.minkowski_dot(&u.view(), &v.view())).max(1.0);
    let dacosh = if arg > 1.0 + 1e-10 {
        1.0 / (arg * arg - 1.0).sqrt()
    } else {
        1.0 / (2.0 * (arg - 1.0).max(1e-15)).sqrt()
    };
    let mut ga = Array1::zeros(u.len());
    ga[0] = v[0];
    for i in 1..u.len() {
        ga[i] = -v[i];
    }
    ga * (2.0 * d * dacosh)
}

/// Euclidean -> Riemannian gradient: apply Minkowski inverse + tangent projection.
fn riem_grad(m: &LorentzModel<f64>, x: &Array1<f64>, ge: &Array1<f64>) -> Array1<f64> {
    let mut g = ge.clone();
    g[0] = -g[0]; // Minkowski inverse metric
    let inner = m.minkowski_dot(&g.view(), &x.view());
    &g + &x.mapv(|xi| xi * inner) // project onto tangent space
}

/// One SGD step: compute gradient, retract via exp_map, project.
fn step(
    m: &LorentzModel<f64>,
    emb: &mut [Array1<f64>],
    u: usize,
    v: usize,
    ge_u: Array1<f64>,
    ge_v: Array1<f64>,
    lr: f64,
) {
    let su = riem_grad(m, &emb[u], &ge_u).mapv(|x| -x * lr);
    let sv = riem_grad(m, &emb[v], &ge_v).mapv(|x| -x * lr);
    emb[u] = m.project(&m.exp_map(&emb[u].view(), &su.view()).view());
    emb[v] = m.project(&m.exp_map(&emb[v].view(), &sv.view()).view());
}

fn main() {
    let m = LorentzModel::<f64>::new(1.0);
    let (n, edges) = build_tree();
    let dim = 2;
    println!(
        "Lorentz SGD: {} nodes, {} edges, 3 levels\n",
        n,
        edges.len()
    );

    let mut rng = Lcg::new(42);
    let mut emb: Vec<Array1<f64>> = (0..n)
        .map(|_| {
            let mut x = Array1::zeros(dim + 1);
            x[0] = 1.0;
            for j in 1..=dim {
                x[j] = rng.next_signed(0.01);
            }
            m.project(&x.view())
        })
        .collect();

    let (lr, epochs) = (0.01, 300);
    println!("{:>6}  {:>12}  {:>12}", "epoch", "mean_loss", "max_norm");
    println!("{:-<6}  {:-<12}  {:-<12}", "", "", "");

    for epoch in 0..epochs {
        let mut loss = 0.0;
        for &(u, v) in &edges {
            let d = m.distance(&emb[u].view(), &emb[v].view());
            loss += d * d;
            let gu = grad_dsq(&m, &emb[u], &emb[v]);
            let gv = grad_dsq(&m, &emb[v], &emb[u]);
            step(&m, &mut emb, u, v, gu, gv, lr);
        }
        let mut rng_r = Lcg::new(epoch as u64);
        for _ in 0..edges.len() {
            let u = (rng_r.next_f64() * n as f64) as usize % n;
            let v = (rng_r.next_f64() * n as f64) as usize % n;
            if u == v {
                continue;
            }
            if edges
                .iter()
                .any(|&(a, b)| (a == u && b == v) || (a == v && b == u))
            {
                continue;
            }
            let d = m.distance(&emb[u].view(), &emb[v].view());
            let viol = 2.0 - d;
            if viol <= 0.0 || d < 1e-12 {
                continue;
            }
            loss += viol * viol;
            let scale = -2.0 * viol;
            let gu = grad_dsq(&m, &emb[u], &emb[v]).mapv(|x| x / (2.0 * d) * scale);
            let gv = grad_dsq(&m, &emb[v], &emb[u]).mapv(|x| x / (2.0 * d) * scale);
            step(&m, &mut emb, u, v, gu, gv, lr);
        }
        if epoch % 50 == 0 || epoch == epochs - 1 {
            let ml = loss / (2 * edges.len()) as f64;
            let origin = m.origin(dim);
            let mx = emb
                .iter()
                .map(|e| m.distance(&origin.view(), &e.view()))
                .fold(0.0f64, f64::max);
            println!("{:>6}  {:>12.6}  {:>12.4}", epoch, ml, mx);
        }
    }

    // ---- Evaluation ----
    println!("\n--- Evaluation ---\n");
    let pc: Vec<f64> = edges
        .iter()
        .map(|&(u, v)| m.distance(&emb[u].view(), &emb[v].view()))
        .collect();
    let mean_pc = pc.iter().sum::<f64>() / pc.len() as f64;

    let parents: HashSet<usize> = edges.iter().map(|&(u, _)| u).collect();
    let leaves: Vec<usize> = (0..n).filter(|i| !parents.contains(i)).collect();
    let mut cross = Vec::new();
    for i in 0..leaves.len() {
        for j in (i + 1)..leaves.len() {
            if tree_dist(&edges, n, leaves[i], leaves[j]) >= 4 {
                cross.push(m.distance(&emb[leaves[i]].view(), &emb[leaves[j]].view()));
            }
        }
    }
    let mean_cross = cross.iter().sum::<f64>() / cross.len().max(1) as f64;

    println!("Mean parent-child distance:  {:.4}", mean_pc);
    println!("Mean cross-branch distance:  {:.4}", mean_cross);
    println!(
        "Ratio (cross/parent-child):  {:.2}x\n",
        mean_cross / mean_pc
    );

    let origin = m.origin(dim);
    println!("{:>6}  {:>10}  {:>5}", "depth", "mean_dist", "count");
    println!("{:-<6}  {:-<10}  {:-<5}", "", "", "");
    for depth in 0..=3 {
        let norms: Vec<f64> = (0..n)
            .filter(|&i| tree_dist(&edges, n, 0, i) == depth)
            .map(|i| m.distance(&origin.view(), &emb[i].view()))
            .collect();
        if !norms.is_empty() {
            println!(
                "{:>6}  {:>10.4}  {:>5}",
                depth,
                norms.iter().sum::<f64>() / norms.len() as f64,
                norms.len()
            );
        }
    }
    println!("\nDeeper nodes should have larger distance from origin.");
    println!("Cross-branch distances should exceed parent-child distances.");
}
