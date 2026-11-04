"""Sanity checks: does per-statement monotone calibration change / zero the rate of loss?

L(p) = max_b min_omega sum_i b_i (1_Ei(omega) - p_i)  s.t. sum |b_i| <= 1   (Andrews 2026, eq. 1)
"""
import numpy as np
from scipy.optimize import linprog
from scipy.special import expit, logit

rng = np.random.default_rng(0)

# Atom-indicator matrices M[i, j] = 1_{E_i}(omega_j)
NEG = np.array([[1, 0],   # A
                [0, 1]])  # not A
# conj family: atoms (A,B) in TT, TF, FT, FF; events A, notA, B, notB, A and notB
CONJ = np.array([[1, 1, 0, 0],
                 [0, 0, 1, 1],
                 [1, 0, 1, 0],
                 [0, 1, 0, 1],
                 [0, 1, 0, 0]])


def rate_of_loss(p, M):
    """Andrews' L(p). Variables: b+ (n), b- (n), t. Maximize t."""
    n, m = M.shape
    c = np.zeros(2 * n + 1); c[-1] = -1.0
    # t - sum_i (b+_i - b-_i)(M_ij - p_i) <= 0 for every atom j
    D = M - p[:, None]                      # n x m
    A_ub = np.hstack([-D.T, D.T, np.ones((m, 1))])
    b_ub = np.zeros(m)
    A_ub = np.vstack([A_ub, np.r_[np.ones(2 * n), 0.0]])
    b_ub = np.r_[b_ub, 1.0]
    bounds = [(0, None)] * (2 * n) + [(None, None)]
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    return -r.fun


def dist_to_coherent(p, M):
    """min_q ||M q - p||_inf over q in the simplex (the LP dual of L(p))."""
    n, m = M.shape
    c = np.zeros(m + 1); c[-1] = 1.0
    A_ub = np.vstack([np.hstack([M, -np.ones((n, 1))]),
                      np.hstack([-M, -np.ones((n, 1))])])
    b_ub = np.r_[p, -p]
    A_eq = np.r_[np.ones(m), 0.0][None, :]
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0],
                bounds=[(0, None)] * m + [(0, None)], method="highs")
    return r.fun


def temp(q, T, c=0.0):
    return expit(logit(np.clip(q, 1e-9, 1 - 1e-9)) / T + c)


print("== 1. Closed form and duality checks ==")
for _ in range(5):
    p = rng.uniform(size=2)
    print(f"neg p={p.round(3)}  L={rate_of_loss(p, NEG):.4f}  |1-sum|/2={abs(1 - p.sum()) / 2:.4f}"
          f"  dist={dist_to_coherent(p, NEG):.4f}")
for _ in range(5):
    p = rng.uniform(size=5)
    print(f"conj p={p.round(2)}  L={rate_of_loss(p, CONJ):.4f}  dist_inf={dist_to_coherent(p, CONJ):.4f}")
print("coherent conj (from q on atoms):", round(rate_of_loss(CONJ @ np.array([.1, .2, .3, .4]), CONJ), 6))
print("labels as previsions conj:", round(rate_of_loss(CONJ[:, 1].astype(float), CONJ), 6))
print("constant 1/2 conj:", round(rate_of_loss(np.full(5, 0.5), CONJ), 6),
      " constant 1/2 neg:", round(rate_of_loss(np.full(2, 0.5), NEG), 6))
for c in [0.3, 0.4, 0.6, 0.7]:
    print(f"constant {c} conj: {rate_of_loss(np.full(5, c), CONJ):.4f}")


# ---------------- synthetic probe -----------------
def sim_neg_pairs(n, sep, blind, noise=1.0):
    """Probe logits for (A, notA). y = truth of A.
    `blind` in [0,1]: fraction of the truth signal on notA that ignores the negation
    (Marks & Tegmark: probes trained on affirmatives transfer poorly to negations)."""
    y = rng.integers(0, 2, n)
    s = 2 * y - 1
    zA = sep * s + noise * rng.standard_normal(n)
    zN = sep * ((1 - blind) * (-s) + blind * s) + noise * rng.standard_normal(n) + 0.5  # +0.5 = bias
    return expit(zA), expit(zN), y


def sim_conj(n, sep, noise=1.0, bias=0.3):
    yA = rng.integers(0, 2, n); yB = rng.integers(0, 2, n)
    y = np.stack([yA, 1 - yA, yB, 1 - yB, yA * (1 - yB)], 1)
    z = sep * (2 * y - 1) + noise * rng.standard_normal(y.shape) + bias
    return expit(z), y


def mean_L(P, M):
    return float(np.mean([rate_of_loss(p, M) for p in P]))


def fit_temp(q, y, with_bias=False):
    """Fit T (and bias) by log-loss on held-out calibration data."""
    from scipy.optimize import minimize
    z = logit(np.clip(q, 1e-9, 1 - 1e-9))
    def nll(th):
        T = np.exp(th[0]); c = th[1] if with_bias else 0.0
        pp = np.clip(expit(z / T + c), 1e-12, 1 - 1e-12)
        return -np.mean(y * np.log(pp) + (1 - y) * np.log(1 - pp))
    r = minimize(nll, [0.0, 0.0] if with_bias else [0.0], method="Nelder-Mead")
    return np.exp(r.x[0]), (r.x[1] if with_bias else 0.0)


def fit_isotonic(q, y):
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(q, y)
    return ir.predict


print("\n== 2. Negation pairs: temperature sweep (shared map on A and notA) ==")
qA, qN, y = sim_neg_pairs(400, sep=2.0, blind=0.0)
P = np.stack([qA, qN], 1)
for T in [0.05, 0.25, 0.5, 1, 2, 4, 16, 1e3]:
    print(f"T={T:<6} mean L={mean_L(temp(P, T), NEG):.4f}")
hard = (P > 0.5).astype(float)
print("T->0 limit (hard labels) mean L:", round(mean_L(hard, NEG), 4),
      " frac pairs classified inconsistently / 2:", round(np.mean(hard.sum(1) != 1) / 2, 4))

print("\n== 3. Calibration fitted on held-out data (disjoint families) ==")
for blind in [0.0, 0.5, 1.0]:
    cA, cN, cy = sim_neg_pairs(2000, 2.0, blind)      # calibration families
    tA, tN, ty = sim_neg_pairs(400, 2.0, blind)       # booked families
    qc = np.r_[cA, cN]; yc = np.r_[cy, 1 - cy]
    Pt = np.stack([tA, tN], 1); yt = np.stack([ty, 1 - ty], 1)
    T, _ = fit_temp(qc, yc)
    T2, c2 = fit_temp(qc, yc, with_bias=True)
    iso = fit_isotonic(qc, yc)
    acc = lambda Q: np.mean((Q > 0.5) == yt)
    brier = lambda Q: np.mean((Q - yt) ** 2)
    rows = {"raw": Pt, f"temp T={T:.2f}": temp(Pt, T), f"platt T={T2:.2f},c={c2:.2f}": temp(Pt, T2, c2),
            "isotonic": iso(Pt.ravel()).reshape(Pt.shape), "const 0.5": np.full_like(Pt, 0.5)}
    print(f"-- negation-blindness={blind}")
    for k, Q in rows.items():
        print(f"   {k:<24} L={mean_L(Q, NEG):.4f}  acc={acc(Q):.3f}  brier={brier(Q):.4f}")

print("\n== 4. Conjunction families ==")
Pc, yc = sim_conj(300, 1.5)
for T in [0.05, 0.5, 1, 2, 8, 1e3]:
    print(f"T={T:<6} mean L={mean_L(temp(Pc, T), CONJ):.4f}")
Cc, cyc = sim_conj(2000, 1.5)
T, _ = fit_temp(Cc.ravel(), cyc.ravel()); T2, c2 = fit_temp(Cc.ravel(), cyc.ravel(), True)
iso = fit_isotonic(Cc.ravel(), cyc.ravel())
print(f"raw L={mean_L(Pc, CONJ):.4f}  temp(T={T:.2f}) L={mean_L(temp(Pc, T), CONJ):.4f}"
      f"  platt L={mean_L(temp(Pc, T2, c2), CONJ):.4f}"
      f"  isotonic L={mean_L(iso(Pc.ravel()).reshape(Pc.shape), CONJ):.4f}")


print("\n== 5. Best shared monotone map: LP over f values at the sorted raw previsions ==")
def best_monotone(Ps, M, slope):
    """min sum_k L_k(f(q_k)) over nondecreasing f with f(x')-f(x) >= slope*(x'-x).
    Uses L = min_q ||Mq - p||_inf (dual form), so the whole thing is one LP."""
    K, n = Ps.shape; m = M.shape[1]
    xs, inv = np.unique(Ps.ravel(), return_inverse=True); inv = inv.reshape(Ps.shape)
    nf = len(xs); nq = K * m; nv = nf + nq + K
    c = np.r_[np.zeros(nf + nq), np.ones(K)]
    A, b = [], []
    for k in range(K):
        for i in range(n):
            row = np.zeros(nv); row[nf + k * m: nf + (k + 1) * m] = M[i]; row[inv[k, i]] = -1; row[nf + nq + k] = -1
            A.append(row); b.append(0)
            A.append(-row - np.eye(nv)[nf + nq + k] * 2); b.append(0)   # -(Mq - f) - t <= 0
    for j in range(nf - 1):
        row = np.zeros(nv); row[j] = 1; row[j + 1] = -1
        A.append(row); b.append(-slope * (xs[j + 1] - xs[j]))
    Aeq = np.zeros((K, nv))
    for k in range(K):
        Aeq[k, nf + k * m: nf + (k + 1) * m] = 1
    r = linprog(c, A_ub=np.array(A), b_ub=np.array(b), A_eq=Aeq, b_eq=np.ones(K),
                bounds=[(0, 1)] * nf + [(0, None)] * nq + [(0, None)] * K, method="highs")
    return r.fun / K


qA, qN, _ = sim_neg_pairs(60, 2.0, 0.3)
Pn = np.stack([qA, qN], 1)
Pc, _ = sim_conj(40, 1.5)
print(f"raw: neg L={mean_L(Pn, NEG):.4f}  conj L={mean_L(Pc, CONJ):.4f}")
for s in [0.0, 0.05, 0.2, 0.5, 1.0]:
    print(f"min slope {s:<5} best neg L={best_monotone(Pn, NEG, s):.4f}  best conj L={best_monotone(Pc, CONJ, s):.4f}")


print("\n== 6. Can calibration flip a model ranking? ==")
# model X: soft (under-confident) but never negation-blind; model Y: sharp, noisy, 10% negation-blind.
rng = np.random.default_rng(1)


def run(sep, blind, noise):
    cA, cN, cy = sim_neg_pairs(3000, sep, blind, noise)
    A, N, y = sim_neg_pairs(400, sep, blind, noise)
    P = np.stack([A, N], 1); yt = np.stack([y, 1 - y], 1)
    T, _ = fit_temp(np.r_[cA, cN], np.r_[cy, 1 - cy])
    return mean_L(P, NEG), mean_L(temp(P, T), NEG), T, np.mean((P > 0.5) == yt)


for name, args in [("X", (0.6, 0.0, 0.45)), ("Y", (3.0, 0.1, 3.0))]:
    Lr, Lc, T, acc = run(*args)
    print(f"{name}: raw L={Lr:.4f}  calibrated L={Lc:.4f} (T={T:.2f})  acc={acc:.3f}")
