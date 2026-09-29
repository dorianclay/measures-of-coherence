# How do we elicit previsions from model behavior?

Type: prototype
Status: open
Blocked by:

## Question

Choose how an **elicited prevision** is read from a model:

- **Token logprobs**: normalized P("True") / (P("True") + P("False")) after a judgment prompt. Works on base models.
- **Stated probability**: ask for a number from 0 to 100 and parse it. Needs an instruct model; output is coarse and often clustered.
- Prompt template(s), whether a fixed **role** is set per batch of queries (Andrews §2 recommends testing coherence within one role), and whether paraphrase variants count toward the same family.

Prototype: a throwaway script that runs both methods on ~20 families from a small model (SmolLM2-135M, plus a small instruct model), and prints the previsions side by side with their naive rate of loss. React to it together.
