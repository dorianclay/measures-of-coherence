# Implement and validate the rate-of-loss solver

Type: task (AFK)
Status: open
Blocked by: 01, 03

## Question

Implement the formulations chosen in [Which rate-of-loss formulation(s) do we compute?](01-rate-of-loss-formulation.md): L, plus ρ for conjunction families. The LP shapes and the reference script `rate_of_loss_check.py` are on branch `research/rate-of-loss-formulation`. Implement the formulations as a solver over an event family (atoms × events membership matrix + previsions → rate of loss, plus the optimal stakes). Validate it before any real previsions touch it:

- Closed-form oracles from 01 for negation pairs and conjunction families.
- Label previsions give exactly 0 on every family with `labels_consistent: true`. The two facts families flagged false (the contested Nile fact) must give a positive rate, which checks the solver in the other direction.
- Random previsions give a positive rate, which sets the ceiling for the random control.

Build test-first (`/tdd`). Done when the solver runs over all families from 03 with label and random previsions and the results are recorded.
