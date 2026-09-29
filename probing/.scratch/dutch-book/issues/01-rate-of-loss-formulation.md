# Which rate-of-loss formulation(s) do we compute?

Type: research
Status: resolved
Blocked by:

## Question

The thesis (§4.1.2) writes Andrews' L(p): the maximum over stakes b with Σ|bᵢ| ≤ 1 of the minimum over atoms of Σ bᵢ(1_Eᵢ(ωⱼ) − p(Eᵢ)). It flags that SSK (1998) define the bookie's rate differently, as ρ = sup g/h with escrow h. Settle:

1. Are Andrews' L(p) and SSK's bookie-escrow rate ρ (and the gambler-escrow rate) equivalent, proportional, or genuinely different orderings over prevision assignments? Give a small counterexample if they differ.
2. For each formulation we might report, what is the exact LP (or LP family / fractional-program transform) over a finite set of atoms and events, in a form ready for `scipy.optimize.linprog`?
3. Closed forms for our two event-family shapes: the negation pair {A, ¬A} and the conjunction family {A, ¬A, B, ¬B, A∧¬B} (with its polarity variants). SSK give closed forms for some of these; these become test oracles for the solver.
4. A recommendation: which formulation(s) the experiments should report and why, and the one-line fix to the thesis's notation.

Sources: `references/` PDFs (thesis draft, Andrews 2026 §2, SSK 1998 §§2–5, 7).

## Answer

Findings: branch `research/rate-of-loss-formulation`, file `.scratch/dutch-book/research/rate-of-loss-formulation.md`. The verification script `rate_of_loss_check.py` is next to it and checks the stake LP against the dual LP and the closed forms to about 1e-13. It also reproduces SSK Theorem 2 and Examples 6–8.

1. **Not equivalent.** Andrews' L, SSK's bookie-escrow rate ρ and SSK's gambler-escrow rate ψ have the same numerator: SSK's bets are the bookie's side, so α = −b. Only the normalization differs, and Σ|bᵢ| = bookie escrow + gambler escrow. On negation pairs, with d = |p(A)+p(¬A)−1|: L = d/2, ρ = d/(1+d), ψ = d/(1−d). All three are monotone in d, so they give the same ordering. On conjunction families the orderings genuinely differ: p1 = (.7,.5,.5,.5,.25) has L = .100 and ρ = .167, while p2 = (.8,.2,.2,.8,.27) has L = .110 and ρ = .142.
2. **Exact LPs.** Each rate is one LP over split stakes b⁺, b⁻ ≥ 0 and t, maximizing t. There is one row per atom and one normalization row. The normalization costs per unit are (1,1) for L, (1−p, p) for ρ and (p, 1−p) for ψ. The dual gives the nearest coherent prevision: L is the L∞ distance to the coherent set.
3. **Closed forms.** Each rate is the maximum over the family's coherence constraints of one formula applied to that constraint's violation. For L, a violation involving k statements costs v/k. This covers both family shapes and all four conjunction polarities. `constraints_conj()` / `closed_form()` in the script are the test oracles for the solver ticket.
4. **Recommendation.** Report **L as the primary rate**. Report **ρ as a secondary column on conjunction families only**, since it adds nothing on negation pairs. **Don't report ψ**: it blows up as previsions approach 0, and uncalibrated probes produce previsions near 0. Don't compare L across family shapes. For the thesis: rename "KSS" to "SSK", put Σ|bᵢ| ≤ 1 into (4.6), change the range of L to [0, 1], and add the one-line reconciliation with SSK given in the findings file.
