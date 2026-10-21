"""Numerical checks for the rate-of-loss formulations (research ticket 01).

Three rates, all sharing the gambler's guaranteed payoff
    G(b) = min_j sum_i b_i (1_{E_i}(w_j) - p_i)
and differing only in the normalization of the stakes b:
    L   (Andrews 2026 / thesis): sum_i |b_i|                          <= 1
    rho (SSK bookie escrow)    : sum_i b_i^+ (1-p_i) + b_i^- p_i      <= 1
    psi (SSK gambler escrow)   : sum_i b_i^+ p_i     + b_i^- (1-p_i)  <= 1
(b_i > 0: gambler buys a $1 ticket on E_i at price p_i; b_i < 0: gambler sells.)
Each is solved as a primal LP (stakes) and a dual LP (nearest coherent prevision).
"""
import itertools
import numpy as np
from scipy.optimize import linprog


def cost_vectors(p, kind):
    """Per-unit cost of a buy (b+) and a sell (b-) stake on each event."""
    p = np.asarray(p, float)
    if kind == "L":
        return np.ones_like(p), np.ones_like(p)
    if kind == "rho":
        return 1 - p, p
    if kind == "psi":
        return p, 1 - p
    raise ValueError(kind)


def rate_primal(A, p, kind):
    """A: n x m 0/1 matrix, A[i, j] = 1_{E_i}(atom j). Returns (rate, b)."""
    A = np.asarray(A, float)
    p = np.asarray(p, float)
    n, m = A.shape
    cplus, cminus = cost_vectors(p, kind)
    D = A - p[:, None]  # D[i, j] = 1_{E_i}(w_j) - p_i
    # variables x = [b+ (n), b- (n), t]; maximize t  <=>  minimize -t
    c = np.r_[np.zeros(2 * n), -1.0]
    # t - sum_i (b+_i - b-_i) D[i, j] <= 0   for every atom j
    A_ub = np.hstack([-D.T, D.T, np.ones((m, 1))])
    b_ub = np.zeros(m)
    # normalization row
    A_ub = np.vstack([A_ub, np.r_[cplus, cminus, 0.0]])
    b_ub = np.r_[b_ub, 1.0]
    bounds = [(0, None)] * (2 * n) + [(None, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if res.status == 3:
        return np.inf, None
    assert res.status == 0, res.message
    b = res.x[:n] - res.x[n:2 * n]
    return max(0.0, -res.fun), b


def rate_dual(A, p, kind):
    """min eps s.t. q = A @ lam, lam in simplex, q inside the kind's eps-box around p."""
    A = np.asarray(A, float)
    p = np.asarray(p, float)
    n, m = A.shape
    # variables [lam (m), eps]
    c = np.r_[np.zeros(m), 1.0]
    if kind == "L":      # p - eps <= q <= p + eps
        lo_p, lo_e, hi_p, hi_e = p, -np.ones(n), p, np.ones(n)
    elif kind == "rho":  # (1-eps) p <= q <= (1-eps) p + eps
        lo_p, lo_e, hi_p, hi_e = p, -p, p, 1 - p
    elif kind == "psi":  # (1+eps) p - eps <= q <= (1+eps) p
        lo_p, lo_e, hi_p, hi_e = p, p - 1, p, p
    # q <= hi_p + eps*hi_e   ->  A lam - hi_e eps <= hi_p
    # q >= lo_p + eps*lo_e   -> -A lam + lo_e eps <= -lo_p
    A_ub = np.vstack([np.hstack([A, -hi_e[:, None]]), np.hstack([-A, lo_e[:, None]])])
    b_ub = np.r_[hi_p, -lo_p]
    A_eq = np.r_[np.ones(m), 0.0][None, :]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0],
                  bounds=[(0, None)] * (m + 1), method="highs")
    if res.status == 2:
        return np.inf, None
    assert res.status == 0, res.message
    return res.fun, A @ res.x[:m]


# ---------------------------------------------------------------- families
NEG_PAIR = np.array([[1, 0],   # A     atoms: A, notA
                     [0, 1]])  # notA


def conj_family(sx, sy):
    """Family {A, notA, B, notB, X and Y}, X = A if sx else notA, Y = B if sy else notB.
    Atoms (A,B) in TT, TF, FT, FF. Returns A matrix with rows A, notA, B, notB, C."""
    atoms = list(itertools.product([1, 0], [1, 0]))
    rows = []
    for a, b in atoms:
        x = a if sx else 1 - a
        y = b if sy else 1 - b
        rows.append([a, 1 - a, b, 1 - b, x * y])
    return np.array(rows).T


# ---------------------------------------------------------------- closed forms
def constraints_conj(sx, sy):
    """Facet list of the coherent set for the conj family, as (sigma, c):
    coherence <=> sigma . q <= c.  Index order A, notA, B, notB, C.
    Generated from X/Xbar, Y/Ybar polarity so it covers all four variants."""
    iX, iXb = (0, 1) if sx else (1, 0)
    iY, iYb = (2, 3) if sy else (3, 2)
    C = 4
    cons = []

    def add(coefs, c):
        s = np.zeros(5)
        for i, v in coefs:
            s[i] += v
        cons.append((s, c))
    # complements: q_A + q_notA = 1, q_B + q_notB = 1
    add([(0, 1), (1, 1)], 1); add([(0, -1), (1, -1)], -1)
    add([(2, 1), (3, 1)], 1); add([(2, -1), (3, -1)], -1)
    # C <= X, written with X or with 1 - Xbar; same for Y
    add([(C, 1), (iX, -1)], 0); add([(C, 1), (iXb, 1)], 1)
    add([(C, 1), (iY, -1)], 0); add([(C, 1), (iYb, 1)], 1)
    # C >= X + Y - 1, each of X, Y written directly or via complement
    for xs in [(iX, 1, 0), (iXb, -1, 1)]:      # q_X = xs[1]*q_idx + xs[2]
        for ys in [(iY, 1, 0), (iYb, -1, 1)]:
            add([(xs[0], xs[1]), (ys[0], ys[1]), (C, -1)], 1 - xs[2] - ys[2])
    # C >= 0 (never violated for p in [0,1]) omitted
    return cons


def closed_form(cons, p, kind):
    """Max over facets sigma.q <= c of the single-facet rate.
    v = sigma.p - c, k = #statements, S+/S- = positive/negative coefficients."""
    p = np.asarray(p, float)
    best = 0.0
    for s, c in cons:
        v = s @ p - c
        if v <= 1e-15:
            continue
        k = np.count_nonzero(s)
        splus = np.sum(s > 0)
        sminus = np.sum(s < 0)
        if kind == "L":
            r = v / k
        elif kind == "rho":
            r = v / (v + c + sminus)
        elif kind == "psi":
            den = splus - c - v
            r = np.inf if den <= 0 else v / den
        best = max(best, r)
    return best


NEG_CONS = [(np.array([1.0, 1.0]), 1.0), (np.array([-1.0, -1.0]), -1.0)]


def neg_closed(p, kind):
    d = abs(p[0] + p[1] - 1)
    return {"L": d / 2, "rho": d / (1 + d), "psi": (d / (1 - d) if d < 1 else np.inf)}[kind]


# ---------------------------------------------------------------- checks
def main():
    rng = np.random.default_rng(0)
    kinds = ["L", "rho", "psi"]

    print("== Negation pair p(A)=0.7, p(notA)=0.5 ==")
    p = [0.7, 0.5]
    for k in kinds:
        r, b = rate_primal(NEG_PAIR, p, k)
        rd, q = rate_dual(NEG_PAIR, p, k)
        print(f"{k:4s} primal={r:.6f} dual={rd:.6f} closed={neg_closed(p, k):.6f} "
              f"stakes b={np.round(b, 4)} nearest q={np.round(q, 4)}")
    print("SSK Thm 2: rho=(s-1)/s =", 0.2 / 1.2, "; SSK Ex. 6: psi=(s-1)/(2-s) =", 0.2 / 0.8)
    print("identity sum|b| = bookie escrow + gambler escrow; harmonic bound L <= rho*psi/(rho+psi):",
          (1 / 6 * 0.25) / (1 / 6 + 0.25))

    print("\n== SSK Examples 7/8 (3-cell partition), gambler rate psi ==")
    P3 = np.eye(3)
    for p3 in [0.7, 0.2, 0.3]:
        pp = [0.6, 0.7, p3]
        print(f"p={pp}: psi={rate_primal(P3, pp, 'psi')[0]:.4f} rho={rate_primal(P3, pp, 'rho')[0]:.4f} "
              f"L={rate_primal(P3, pp, 'L')[0]:.4f}")
    print("SSK: psi = 1.0 (p3=.7), 0.4286 (p3=.2), 0.4286 (p3=.3); rho=(s-1)/s")

    print("\n== Random negation pairs: primal = dual = closed form ==")
    err = 0
    for _ in range(2000):
        p = rng.uniform(0.01, 0.99, 2)
        for k in kinds:
            r = rate_primal(NEG_PAIR, p, k)[0]
            err = max(err, abs(r - rate_dual(NEG_PAIR, p, k)[0]), abs(r - neg_closed(p, k)))
    print("max abs err:", err)

    print("\n== Random conjunction families (all 4 polarities): primal = dual = closed form ==")
    for sx, sy in itertools.product([True, False], repeat=2):
        A = conj_family(sx, sy)
        cons = constraints_conj(sx, sy)
        err = 0
        for _ in range(1000):
            # mix of fully random and near-coherent previsions
            if rng.random() < 0.5:
                p = rng.uniform(0.01, 0.99, 5)
            else:
                lam = rng.dirichlet(np.ones(4))
                p = np.clip(A @ lam + rng.normal(0, 0.1, 5), 0.01, 0.99)
            for k in kinds:
                r = rate_primal(A, p, k)[0]
                rd = rate_dual(A, p, k)[0]
                rc = closed_form(cons, p, k)
                err = max(err, abs(r - rd), abs(r - rc))
        name = f"{'A' if sx else 'notA'} and {'B' if sy else 'notB'}"
        print(f"{name:14s} max abs err: {err:.2e}")

    print("\n== Coherent previsions give 0; ground-truth labels give 0 ==")
    A = conj_family(True, False)
    for lam in [np.array([.1, .2, .3, .4]), np.array([0, 1, 0, 0.])]:
        p = A @ lam
        print(p, [round(rate_primal(A, p, k)[0], 12) for k in kinds])

    print("\n== Counterexample: L and rho order two conj-family previsions differently ==")
    A = conj_family(True, False)  # C = A and notB
    # rows: A, notA, B, notB, A&notB
    p1 = [0.7, 0.5, 0.5, 0.5, 0.25]   # only violation: p(A)+p(notA) = 1.2
    p2 = [0.8, 0.2, 0.2, 0.8, 0.27]   # only violation: p(A)+p(notB)-p(A&notB) = 1.33 > 1
    for name, p in [("p1", p1), ("p2", p2)]:
        vals = {k: rate_primal(A, p, k)[0] for k in kinds}
        print(name, p, {k: round(v, 4) for k, v in vals.items()})
    print("L(p1) < L(p2) but rho(p1) > rho(p2); psi agrees with L here.")
    p3 = [0.65, 0.65, 0.5, 0.5, 0.25]  # only violation: p(A)+p(notA) = 1.3
    p4 = [0.8, 0.2, 0.2, 0.8, 0.20]   # only violation: p(A)+p(notB)-p(A&notB) = 1.4 > 1
    for name, p in [("p3", p3), ("p4", p4)]:
        vals = {k: rate_primal(A, p, k)[0] for k in kinds}
        print(name, p, {k: round(v, 4) for k, v in vals.items()})
    print("L and rho rank p3 worse; psi ranks p4 worse.")

    print("\n== Random search: how often do pairs disagree in ordering (conj family)? ==")
    A = conj_family(True, False)
    ps = [np.clip(A @ rng.dirichlet(np.ones(4)) + rng.normal(0, 0.15, 5), 0.01, 0.99) for _ in range(400)]
    R = np.array([[rate_primal(A, p, k)[0] for k in kinds] for p in ps])
    iu = np.triu_indices(len(ps), 1)

    def disc(x, y):
        dx = np.sign(x[:, None] - x[None, :])[iu]
        dy = np.sign(y[:, None] - y[None, :])[iu]
        return np.mean(dx * dy < 0)
    from scipy.stats import spearmanr
    for (i, a), (j, b) in itertools.combinations(enumerate(kinds), 2):
        print(f"{a} vs {b}: discordant pairs {disc(R[:, i], R[:, j]):.3f}, "
              f"spearman {spearmanr(R[:, i], R[:, j]).statistic:.3f}")
    print("bounds hold (L<=rho, L<=psi, L<=rho*psi/(rho+psi)):",
          bool(np.all(R[:, 0] <= R[:, 1] + 1e-9) and np.all(R[:, 0] <= R[:, 2] + 1e-9)
               and np.all(R[:, 0] <= R[:, 1] * R[:, 2] / np.maximum(R[:, 1] + R[:, 2], 1e-12) + 1e-9)))

    print("\n== psi blows up near 0: p(A)=p(notA)=0.01 ==")
    print({k: rate_primal(NEG_PAIR, [0.01, 0.01], k)[0] for k in kinds})
    print({k: rate_primal(NEG_PAIR, [0.0, 0.0], k)[0] for k in kinds})


if __name__ == "__main__":
    main()
