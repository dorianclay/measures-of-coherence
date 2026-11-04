# Build the event families from the structured datasets

Type: task (AFK)
Status: resolved
Blocked by:

## Question

Turn `neg_*` and `conj_neg_*` into explicit event families with known logical structure, so previsions can be booked against them.

- **Negation pairs**: pair each statement in `neg_facts` / `neg_companies` with its negation.
- **Conjunction families**: split each `conj_neg_*` statement into its two conjuncts, match each conjunct to its `neg_*` row, and record which conjunct(s) are negated. Negation can appear in either conjunct ("A, and B isn't…", "A isn't…, and B…"), so the family's events are {A, ¬A, B, ¬B, conj} with the conjunction's polarity recorded. A rough check found both conjuncts for 523/561 facts rows and 493/550 companies rows; verify properly and decide what to do with the rest.
- Check that labels are consistent within each family, since label-previsions must give a rate of loss of 0.

Done when: a family file per dataset exists (format your choice, committed alongside a small builder script), with each family's events, their atom-membership structure, and source rows. Record the counts, the drop reasons, and any label inconsistencies in the answer. Later tickets depend on these counts.

## Answer

Done. The builder is `Build_Event_Families.py`, and it writes `event_families/{facts,companies}_families.json`. Neither is committed yet. Each family holds its events (statement, label, source file, row, `surface_negated` flag), its atom matrix, and `labels_consistent`. Conjunction events are always ordered [C1, ¬C1, C2, ¬C2, C1∧C2], so every conjunction family shares one atom matrix. Each file also carries a `summary` and a `dropped` list with reasons.

| | facts | companies |
|---|---|---|
| negation pairs (after dedupe) | 547 (14 verbatim repeats removed) | 500 (50 repeats removed) |
| negation pairs with inconsistent labels | 0 | 0 |
| conjunction families | 556 / 561 | 546 / 550 |
| dropped: a conjunct has two different negations in `neg_*` ("doesn't include" vs. "excludes") | 4 | 4 |
| dropped: both conjuncts from the same pair (a contradiction) | 1 | 0 |
| conjunction families with inconsistent labels | 2 | 0 |
| polarity C1/C2 (pos-pos, pos-neg, neg-pos, neg-neg) | 138 / 130 / 149 / 139 | 111 / 163 / 131 / 141 |

Facts later tickets depend on:
- **Conjunctions split correctly.** Splitting only at the one ", and " boundary where both halves are known statements handles conjuncts that contain lists. My rough 523/493 estimate from charting was an undercount.
- **Label inconsistencies.** Both inconsistent families (`conj_neg_facts:117`, `:153`) hinge on "The Nile River is the longest river in the world". `neg_facts` labels it true; `conj_neg_facts` labels treat it as false. The fact is genuinely contested (Nile vs. Amazon). The label-previsions control must exclude `labels_consistent: false` families, or report them separately.
- **Leakage.** About half of the negation-pair statements appear verbatim in the base datasets: 542/1092 in `facts`, 496/999 in `companies`. Those are exactly the datasets the current probes were trained on.
- **Statements recur across families.** Each negation pair appears in several conjunction families, so conjunction families are not independent samples. Every family also contains its two negation pairs, so the two experiments share statements.
