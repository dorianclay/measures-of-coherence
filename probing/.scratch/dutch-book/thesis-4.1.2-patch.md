# Patch for §4.1.2, "Rate of loss to a Dutch Book"

Literal replacement text for the thesis draft (*AI: Agency and Representation*), to paste into the LaTeX source wherever it lives — this repo only has the compiled PDF, not the source. Everything below is already verified (LP vs. closed form vs. dual, to ~1e-13) on branch `research/rate-of-loss-formulation`, `.scratch/dutch-book/research/rate-of-loss-formulation.md`; see [Which rate-of-loss formulation(s) do we compute?](issues/01-rate-of-loss-formulation.md).

## Fix 1 — the "KSS" note after eq. (4.9)

The draft currently has this as an open note rather than body text:

> KSS's math here is a bit different (below). Need to check whether these are equivalent, and if not, how to get the above in line with KSS while remaining clearer

Replace it with (and rename "KSS" → "SSK" throughout the section — the authors are Schervish, Seidenfeld and Kadane):

> SSK's *g* is the bookie's side of the same bets (αᵢ = −bᵢ), so −g(α) is exactly g(b) in (4.5); their ρ = max_b g(b) subject to Σᵢ [bᵢ⁺(1−p(Eᵢ)) + bᵢ⁻p(Eᵢ)] ≤ 1 differs from (4.6) only in normalizing by the bookie's escrow rather than Σ|bᵢ| ≤ 1 (which equals bookie plus gambler escrow), so the two agree in ordering on complementary pairs but not in general.

## Fix 2 — eq. (4.6)

State the constraint inline rather than leaving it implicit:

> L(p) = max_{b ∈ ℝⁿ, Σ|bᵢ| ≤ 1} g(b₁, …, bₙ)

## Fix 3 — range of L

"This produces a measure L(p) ∈ [0, ∞)" → **L(p) ∈ [0, 1]**. (L is bounded because the stake budget Σ|bᵢ| ≤ 1 caps the achievable payoff; ∞ was left over from an earlier unnormalized draft.)

## Fix 4 — eq. (4.7)

Note inline that the gambles here are the bookie's, not the gambler's: "where α = −b are the bookie's stakes (SSK's convention)."

## Fix 5 — the worked example

The subsection header "Worked example: Dutch book-ability" currently has no worked example under it — just two sentences of framing before jumping to §4.1.3. Insert:

---

Take a bookie who states p(A) = 0.7 and p(¬A) = 0.5 for a statement A and its negation. These should sum to 1; they sum to 1.2, so the bookie is willing to *sell* both a $1-ticket on A and a $1-ticket on ¬A for a combined 1.20, while at most 1.00 will ever be paid out (exactly one of A, ¬A 
is true). A gambler who buys both tickets for 1.20 and collects 1.00 nets a guaranteed profit of 0.20 regardless of the outcome. Normalizing by the stake budget (here, both tickets bought at full weight, Σ|bᵢ| = 1) gives

> L(p) = 0.20 / 2 = **0.1**

matching (4.6) with the optimal stakes b = (−0.5, −0.5) (negative because the gambler is selling). By the same numbers, SSK's bookie-escrow rate is ρ = 1/6 ≈ 0.167 and the gambler-escrow rate is ψ = 0.25 — all three agree that the bookie is incoherent and rank this case the same way, because on a two-event negation pair the three rates are monotone functions of one number, d = |p(A)+p(¬A)−1| (here d = 0.2): L = d/2, ρ = d/(1+d), ψ = d/(1−d).

This single-number agreement does not extend to larger event families. For a conjunction family {A, ¬A, B, ¬B, A∧B}, consider two previsions that are each incoherent in a different way:

- p₁ = (0.70, 0.50, 0.50, 0.50, 0.25) violates only the complement constraint p(A)+p(¬A) = 1 (by 0.2, a 2-statement violation): **L(p₁) = 0.100**, ρ(p₁) = 0.167.
- p₂ = (0.80, 0.20, 0.80, 0.20, 0.27) violates only the Fréchet upper bound p(A) + p(¬B) − p(A∧¬B) ≤ 1 (by 0.33, a 3-statement violation): **L(p₂) = 0.110**, ρ(p₂) = 0.142.

L ranks p₂ as more incoherent than p₁; ρ ranks p₁ as more incoherent than p₂. Each rate divides a violation's size by a different quantity — L by the number of statements the violated constraint involves, ρ by the prices bound up in it — so the two rates are not monotone transforms of one another once more than a single complementary pair is in play. This is why the experiments in this chapter report L as the primary rate and ρ only as a secondary column on conjunction families, and never compare L across families of different shapes.

---

## Notes for whoever pastes this in

- The reconciliation paragraph (Fix 1) and the two-statement identities (worked example) are proven and checked to 1e-13 against both the LP and closed-form solutions; see the research doc for the derivation (Fourier–Motzkin elimination over the family's facets) and `DutchBook.py` / `tests/test_dutch_book.py` for the implementation used throughout the rest of this chapter's experiments (see the companion write-up notebook, `Dutch_Book_Write_Up.ipynb`).
- Equation/section numbers above (4.5–4.9, §4.1.2, §4.1.3) match the PDF as of `references/AI__Agency__and_Representation.pdf`; re-check them against the current draft before pasting, in case renumbering has happened since.
