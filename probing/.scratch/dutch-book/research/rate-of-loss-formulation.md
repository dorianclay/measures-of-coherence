# Research: which rate-of-loss formulation(s) do we compute?

Resolves ticket `issues/01-rate-of-loss-formulation.md`. Terms follow `CONTEXT.md` (prevision, event family, atom, rate of loss).

Sources (all in `references/`, untracked):
- **Thesis**: *AI: Agency and Representation*, draft §4.1.2, eqs. 4.5–4.9 (p. 26).
- **Andrews**: Andrews (2026), *Revealed Rationality*, §2, eq. (1) and footnote 4 (pp. 4–5).
- **SSK**: Schervish, Seidenfeld & Kadane (1998), *Two Measures of Incoherence*: §2 Theorem 1, eqs. (2.2)–(2.3) and Theorem 2 (pp. 3, 6); §4 eq. (4.1), Examples 5–8 and Theorem 4 (pp. 9–11); §7 Theorem 7 (pp. 20–21). I checked the equations against the PDF page images, because the extracted text drops the minus signs.

Numerical check: [`rate_of_loss_check.py`](rate_of_loss_check.py), which uses `scipy.optimize.linprog` with HiGHS. Run it with `.venv/bin/python .scratch/dutch-book/research/rate_of_loss_check.py`. Every claim below that is marked "verified" is checked by that script.

---

## TL;DR

- Andrews' L, SSK's bookie rate ρ and SSK's gambler rate ψ all have the **same numerator**: the gambler's guaranteed payoff G(b) = min_j Σᵢ bᵢ(1_{Eᵢ}(ωⱼ) − pᵢ). They differ **only in how the stakes are normalized**. Andrews' Σ|bᵢ| is exactly SSK's bookie escrow **plus** SSK's gambler escrow.
- On **negation pairs** the three rates are strictly monotone functions of one number, d = |p(A)+p(¬A)−1|: L = d/2, ρ = d/(1+d), ψ = d/(1−d). So they produce **the same ordering**.
- On **conjunction families** they are **genuinely different orderings**. Counterexamples are in Q1. In a random sample about 1–5% of pairs of families are ranked differently.
- Each rate is a single LP. Each also has an equivalent dual LP that gives the "nearest coherent prevision", and L is simply the **L∞ (Chebyshev) distance from p to the coherent polytope**.
- There are closed forms for both family shapes, for all three rates and all four conjunction polarities. They match the LPs to 1e-13 on thousands of random previsions (verified).
- **Recommendation:** report **L** as the primary rate of loss, with **ρ** as a secondary robustness column. Do not report ψ, because it is unbounded near 0 and 1. The one-line thesis fix is in Q4.

---

## Q1. Equivalent, proportional, or different orderings?

### Aligning the notation

Andrews and the thesis write the gambler's stakes as b. Here bᵢ > 0 means the gambler buys a $1 ticket on Eᵢ at price pᵢ, and the gambler's payoff is bᵢ(1_{Eᵢ} − pᵢ) (Andrews §2 and fn. 4). SSK write the **bookie's** side: a gamble αᵢ(Xᵢ − xᵢ) pays the bookie, Xᵢ = 1_{Eᵢ} and xᵢ = pᵢ (SSK §2). So **αᵢ = −bᵢ**, and SSK's

- g(α) = sup_t Σ αᵢ(Xᵢ(t) − xᵢ) is the bookie's best-case payoff, which gives **−g(α) = min_j Σ bᵢ(1_{Eᵢ}(ωⱼ) − pᵢ) = G(b)**. This is Andrews' objective exactly.
- The bookie escrow is h(α) = −Σ min{0, inf_t αᵢ(Xᵢ(t) − xᵢ)}. For event indicators this is h = Σᵢ [bᵢ⁺(1−pᵢ) + bᵢ⁻pᵢ] (SSK Example 1).
- The gambler escrow is h′(α) = Σ max{0, sup_t αᵢ(Xᵢ(t) − xᵢ)}. For event indicators this is h′ = Σᵢ [bᵢ⁺pᵢ + bᵢ⁻(1−pᵢ)] (SSK §4, text before Example 5).

This gives the three rates. SSK's ρ = sup −g/h is eq. (2.2) and ψ = −inf g/h′ is eq. (4.1).

| rate | normalization (≤ 1) | source |
|---|---|---|
| L | Σᵢ \|bᵢ\| | Andrews eq. (1); thesis 4.5–4.6 |
| ρ (bookie escrow) | Σᵢ bᵢ⁺(1−pᵢ) + bᵢ⁻pᵢ | SSK Thm 1, (2.2)–(2.3) |
| ψ (gambler escrow) | Σᵢ bᵢ⁺pᵢ + bᵢ⁻(1−pᵢ) | SSK (4.1) |

The key identity is **h + h′ = Σᵢ |bᵢ|**. Andrews' normalization is the total escrow held by both sides. Two consequences follow for any previsions (verified on random samples):

- L ≤ ρ and L ≤ ψ;
- L ≤ ρψ/(ρ+ψ) (harmonic bound). It holds with equality when the same stakes are optimal for all three rates, as on negation pairs.

Ranges: 0 ≤ L ≤ 1, and L ≤ ½ on negation pairs. 0 ≤ ρ ≤ 1 (SSK p. 3: "ρ ≤ 1 in all cases"). 0 ≤ ψ ≤ ∞, and ψ = ∞ when, for example, p(A) = p(¬A) = 0 (SSK Theorem 4.1: ψ = (1−s)/s).

### Negation pairs: same ordering

For {A, ¬A}, let s = p(A)+p(¬A) and d = |s−1|. Then:

- SSK Theorem 2 (bookie rate on a partition): ρ = (s−1)/s if s > 1, and (1−s)/(n−s) = (1−s)/(2−s) if s < 1. Both equal **d/(1+d)**.
- SSK Example 6 and Theorem 4 (gambler rate, n = 2): ψ = (s−1)/(2−s) if s > 1, and (1−s)/s if s < 1. Both equal **d/(1−d)**.
- L = **d/2**. See Q3 for the derivation; verified.

All three are strictly increasing in d, so they give the same ordering. Converting between them: ρ = 2L/(1+2L) and ψ = 2L/(1−2L). For the ticket's example p(A) = 0.7, p(¬A) = 0.5: **L = 0.1, ρ = 1/6, ψ = 0.25**. The primal LP, the dual LP and the closed form all agree, and all three rates use equal short stakes on A and ¬A.

### Beyond negation pairs: genuinely different orderings

On a 3-cell partition, ρ depends only on s (SSK Thm 2). L and ψ do not: SSK Examples 7–8 show ψ changing strategy with p₃. So the rates can already disagree there. Within **our** conjunction family {A, ¬A, B, ¬B, A∧¬B} (rows in that order), the script verifies these counterexamples:

| previsions | only violated constraint | L | ρ | ψ |
|---|---|---|---|---|
| p1 = (0.70, 0.50, 0.50, 0.50, 0.25) | p(A)+p(¬A) = 1.2 > 1 | 0.1000 | **0.1667** | 0.2500 |
| p2 = (0.80, 0.20, 0.20, 0.80, 0.27) | p(A)+p(¬B)−p(A∧¬B) = 1.33 > 1 | **0.1100** | 0.1416 | **0.4925** |
| p3 = (0.65, 0.65, 0.50, 0.50, 0.25) | p(A)+p(¬A) = 1.3 > 1 | **0.1500** | **0.2308** | 0.4286 |
| p4 = (0.80, 0.20, 0.20, 0.80, 0.20) | p(A)+p(¬B)−p(A∧¬B) = 1.4 > 1 | 0.1333 | 0.1667 | **0.6667** |

- L and ψ both say p2 is worse than p1, but ρ says p1 is worse.
- L and ρ both say p3 is worse than p4, but ψ says p4 is worse.

So **none of the three is a monotone transform of another** on the conjunction family.

Why they disagree: a violation of size v in a constraint that involves k statements costs L = v/k. For example, the Fréchet lower bound on the conjunction involves 3 statements, while a complement constraint involves 2. The escrow rates weight the same violation by the prices involved (Q3). In a random sample of 400 near-coherent conjunction families (Gaussian noise, σ = 0.15, around coherent previsions), the share of discordant pairs was L vs ρ 1.2%, L vs ψ 4.3% and ρ vs ψ 5.4%. The Spearman correlations were 0.996, 0.971 and 0.951. The rates are close in practice, but they are not interchangeable.

---

## Q2. Exact LPs for `scipy.optimize.linprog`

Setup: there are n events and m atoms. Aᵢⱼ = 1_{Eᵢ}(ωⱼ) is the n×m 0/1 matrix, and Dᵢⱼ = Aᵢⱼ − pᵢ. Split the stakes as b = b⁺ − b⁻ with b⁺, b⁻ ≥ 0. Let (c⁺, c⁻) be the per-unit costs of a buy stake and a sell stake:

| rate | c⁺ᵢ | c⁻ᵢ |
|---|---|---|
| L | 1 | 1 |
| ρ | 1 − pᵢ | pᵢ |
| ψ | pᵢ | 1 − pᵢ |

### Primal (the optimal stakes, i.e. the Dutch book itself)

```
variables x = [b⁺ (n), b⁻ (n), t (free)]
minimize   −t
s.t.       t − Σᵢ (b⁺ᵢ − b⁻ᵢ) Dᵢⱼ ≤ 0         for each atom j     (m rows)
           Σᵢ c⁺ᵢ b⁺ᵢ + c⁻ᵢ b⁻ᵢ ≤ 1                                 (1 row)
           b⁺, b⁻ ≥ 0
rate = max(0, −res.fun)
```

In linprog terms: `A_ub = [[-D.T, D.T, 1], [c⁺, c⁻, 0]]`, `b_ub = [0,…,0, 1]`, `bounds = [(0,None)]*2n + [(None,None)]`, `method="highs"`.

This is SSK's normalized form (2.3). SSK Theorem 1 guarantees that the fractional program sup G/h equals this LP, because G and h are both positively homogeneous of degree 1. The b⁺/b⁻ split is exact because a solution never holds both a buy and a sell stake on the same event: that would only spend budget. If some pᵢ = 0 and we are computing ψ, the buy cost is 0. The LP can then be unbounded (status 3), which means ψ = ∞.

### Dual (the nearest coherent prevision; recommended for tests and diagnostics)

By LP duality (the minimax theorem over the simplex of atoms × the stake ball), each rate is the smallest ε such that some coherent prevision q = Aλ lies inside an ε-box around p. Here λ ranges over the atom simplex. The three boxes are:

| rate | box on qᵢ | reading |
|---|---|---|
| L | pᵢ − ε ≤ qᵢ ≤ pᵢ + ε | **L = min_{q coherent} ‖q − p‖∞**, the Chebyshev distance to the coherent polytope |
| ρ | (1−ε)pᵢ ≤ qᵢ ≤ (1−ε)pᵢ + ε | smallest ε such that q = (1−ε)p + εu is coherent for some u ∈ [0,1]ⁿ |
| ψ | (1+ε)pᵢ − ε ≤ qᵢ ≤ (1+ε)pᵢ | p = (1−λ)q + λu with λ = ψ/(1+ψ): p is coherent q "contaminated" with weight λ |

```
variables [λ (m), ε]
minimize   ε
s.t.       A λ − hi_e·ε ≤ p          (upper edges of the box)
          −A λ + lo_e·ε ≤ −p         (lower edges of the box)
           Σ λ = 1,  λ, ε ≥ 0
(hi_e, lo_e) = (1, −1) for L;  (1−p, −p) for ρ;  (p, p−1) for ψ
```

Primal and dual agree to about 1e-13 on every random test (verified). The dual's q = Aλ is a useful by-product: it is the coherent prevision that the bookie's numbers are "closest to" in the chosen gauge. Coherent previsions and ground-truth 0/1 labels give exactly 0 under all three rates (verified), which covers the ground-truth control on the map.

---

## Q3. Closed forms (test oracles)

### A general single-constraint formula

Write each facet of the coherent polytope as σ·q ≤ c, with σᵢ ∈ {−1, 0, +1}. For previsions p that violate it, define:

- v = σ·p − c > 0, the size of the violation;
- k = the number of statements in the constraint;
- |S⁺| and |S⁻| = the number of +1 and −1 coefficients.

Fixing that facet on its own, by moving each involved qᵢ as far as its box allows, costs:

- **L = v / k**
- **ρ = v / (v + c + |S⁻|)**, whose denominator is Σ_{S⁺} pᵢ + Σ_{S⁻} (1−pᵢ)
- **ψ = v / (|S⁺| − c − v)**, whose denominator is Σ_{S⁺} (1−pᵢ) + Σ_{S⁻} pᵢ; if it is ≤ 0, ψ = ∞.

For both of our family shapes the **rate equals the maximum of these single-facet values over the facet list**. I showed this by Fourier–Motzkin elimination (c, then a, then β): every feasibility condition that comes out is a pairwise interval condition, and each such condition is one facet. It is **verified numerically for all three rates and all four polarities** to below 1e-12. This max-over-facets property is not true for arbitrary event families. Merged algebras would need the LP.

### Negation pair {A, ¬A}

There are two facets: p(A)+p(¬A) ≤ 1 and ≥ 1. With d = |p(A)+p(¬A)−1|:

**L = d/2, ρ = d/(1+d), ψ = d/(1−d).**

The ρ and ψ forms are SSK Thm 2 and Thm 4 / Ex. 6 with n = 2. L is new here, and verified.

### Conjunction family {A, ¬A, B, ¬B, C = X∧Y}

Here X ∈ {A, ¬A} and Y ∈ {B, ¬B}, and X̄, Ȳ are their complements. Our datasets' A∧¬B is X = A, Y = ¬B. Write p_X for the prevision of whichever statement X is. The coherent set is the two complement equalities plus the Fréchet bounds max(0, q_X+q_Y−1) ≤ q_C ≤ min(q_X, q_Y). Because q_X can be written either as q_X or as 1 − q_X̄, the facets are:

| facet | k | c | \|S⁺\| | \|S⁻\| | violation v |
|---|---|---|---|---|---|
| p(A)+p(¬A) ≤ 1 / ≥ 1 | 2 | 1 / −1 | 2 / 0 | 0 / 2 | ±(p(A)+p(¬A)−1) |
| p(B)+p(¬B) ≤ 1 / ≥ 1 | 2 | 1 / −1 | 2 / 0 | 0 / 2 | ±(p(B)+p(¬B)−1) |
| C ≤ X | 2 | 0 | 1 | 1 | p_C − p_X |
| C ≤ 1 − X̄ | 2 | 1 | 2 | 0 | p_C + p_X̄ − 1 |
| C ≤ Y, C ≤ 1 − Ȳ | 2 | 0, 1 | (as for X) | | p_C − p_Y, p_C + p_Ȳ − 1 |
| C ≥ X + Y − 1 | 3 | 1 | 2 | 1 | p_X + p_Y − p_C − 1 |
| C ≥ (1−X̄) + Y − 1 | 3 | 0 | 1 | 2 | p_Y − p_X̄ − p_C |
| C ≥ X + (1−Ȳ) − 1 | 3 | 0 | 1 | 2 | p_X − p_Ȳ − p_C |
| C ≥ (1−X̄) + (1−Ȳ) − 1 | 3 | −1 | 0 | 3 | 1 − p_X̄ − p_Ȳ − p_C |

(C ≥ 0 can never be violated for p ∈ [0,1].) Each rate is the maximum over the rows of the single-facet formula, and 0 if no row is violated. For L this simplifies to

```
L = max(0,
        |p(A)+p(¬A)−1| / 2,
        |p(B)+p(¬B)−1| / 2,
        (p_C − min(p_X, 1−p_X̄)) / 2,
        (p_C − min(p_Y, 1−p_Ȳ)) / 2,
        (max(p_X, 1−p_X̄) + max(p_Y, 1−p_Ȳ) − 1 − p_C) / 3 )
```

and ρ has the same shape with v/(1+v) on the 2-statement terms and v/(2+v) on the Fréchet-lower term, where v is the bracketed violation. `constraints_conj()` / `closed_form()` in the script is the reference implementation for all three rates and all polarities, and should be ported into the solver's tests (ticket 07).

SSK do not give closed forms for this non-partition family. Their Theorems 2–6 cover partitions and simple random variables only, and §6 covers called-off (conditional) gambles. The table above is derived here and checked against the LP.

---

## Q4. Recommendation

**Report L (Andrews' Σ|b| ≤ 1 normalization) as the rate of loss. Report ρ (SSK bookie escrow) as a secondary column. Do not report ψ.**

Reasons:

1. **L is the thesis's definition and Andrews'.** It keeps the thesis and the experiments consistent, and it has a clean geometric reading: the Chebyshev distance from the bookie's previsions to the nearest coherent previsions. That reading also makes the "nearest coherent prevision" q available for figures.
2. **L is symmetric and bounded.** It does not depend on whether a statement's prevision is near 0 or 1, so it treats probe outputs (which are often extreme) and elicited previsions (which are often rounded) the same way.
3. **ψ is unsafe with probe previsions.** At p(A) = p(¬A) = 0.01, ψ = 49; at exactly 0 it is ∞ (verified; SSK Ex. 5 and Thm 4). Uncalibrated probes will produce values like these and dominate any mean.
4. **ρ is the literature-standard measure** (SSK's "extent of incoherence"). It is bounded in [0, 1] and costs one more LP with a different last row. Reporting it lets readers who know SSK compare results, and it shows whether conclusions depend on the normalization. On negation pairs it adds no information (ρ = 2L/(1+2L)), so compute it only for conjunction families, or report it as a derived column for negation pairs.
5. **Do not compare L across family shapes.** L divides a violation by the number of statements involved, so a given violation reads smaller in a 5-statement family than in a pair. This fits the map's decision to report negation and conjunction families as separate experiments.

Also record the normalization alongside every number, as `CONTEXT.md` requires ("Which normalization is used is stated alongside it").

**One-line fix to thesis §4.1.2.** Replace the "Need to check whether these are equivalent" note (and "KSS" → "SSK") with:

> SSK's g is the bookie's side of the same bets (αᵢ = −bᵢ), so −g(α) is exactly g(b) in (4.5); their ρ = max_b g(b) subject to Σᵢ [bᵢ⁺(1−p(Eᵢ)) + bᵢ⁻p(Eᵢ)] ≤ 1 differs from (4.6) only in normalizing by the bookie's escrow rather than Σ|bᵢ| ≤ 1 (which equals bookie plus gambler escrow), so the two agree in ordering on complementary pairs but not in general.

Smaller fixes in the same subsection:

- (4.6) should state the constraint inline: max over b ∈ ℝⁿ with Σ|bᵢ| ≤ 1.
- "L(p) ∈ [0, ∞)" should be **[0, 1]**.
- In (4.7) the gambles are the bookie's (α = −b).

---

## Script output (abridged)

```
== Negation pair p(A)=0.7, p(notA)=0.5 ==
L    primal=0.100000 dual=0.100000 closed=0.100000 stakes b=[-0.5 -0.5] nearest q=[0.6 0.4]
rho  primal=0.166667 dual=0.166667 closed=0.166667 stakes b=[-0.8333 -0.8333] nearest q=[0.5833 0.4167]
psi  primal=0.250000 dual=0.250000 closed=0.250000 stakes b=[-1.25 -1.25] nearest q=[0.625 0.375]
== SSK Examples 7/8 (3-cell partition), gambler rate psi ==
p=[0.6, 0.7, 0.7]: psi=1.0000 ...     (SSK: 1.0)
p=[0.6, 0.7, 0.2]: psi=0.4286 ...     (SSK: 0.4286)
p=[0.6, 0.7, 0.3]: psi=0.4286 ...     (SSK: 0.4286)
== Random negation pairs: primal = dual = closed form ==   max abs err: 6.8e-14
== Random conjunction families (all 4 polarities) ==       max abs err ≤ 2.0e-13
== Coherent previsions / ground-truth labels ==             [0.0, 0.0, 0.0]
== Discordant pairs (400 noisy conj families) ==            L/rho 0.012, L/psi 0.043, rho/psi 0.054
== psi near 0: p(A)=p(notA)=0.01 ==                         L 0.49, rho 0.495, psi 49.0
```
