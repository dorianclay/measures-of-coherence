"""The registry of measurement protocols.

A protocol fixes everything that decides whether two measurements are comparable: the code path, its parameters
and the dataset versions it reads. Results are comparable only within one protocol version. To change a protocol,
add a new version below and leave the old one as it is, so runs that cite it still say what they did; bumping a
version needs the author's sign-off.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Protocol:
    id: str
    description: str
    code: str
    datasets: dict[str, int] = field(default_factory=dict)
    params: dict = field(default_factory=dict)


# Experiment 001's measurements, ported unchanged into src/. The event families are built from the negation and
# conjunction datasets of the facts and companies domains.
DUTCH_BOOK_DATASETS = {"neg_facts": 1, "neg_companies": 1, "conj_neg_facts": 1, "conj_neg_companies": 1,
                       "facts_families": 1, "companies_families": 1}
# The probes also train on the six atomic datasets and the two domains' base datasets
PROBE_DATASETS = {**DUTCH_BOOK_DATASETS, "facts": 1, "companies": 1, "cities": 1, "animals": 1, "elements": 1,
                  "inventions": 1, "capitals": 1, "generated": 1}
# Activations are extracted once per model reference for every probe protocol, 001's and 002's
ACTIVATION_DATASETS = {name: 1 for name in PROBE_DATASETS if not name.endswith("_families")} | {"umbrella_towns": 1}

PROTOCOLS = {p.id: p for p in [
    Protocol("belief-probe-linear@1",
             "Logistic-regression probe on last-token activations, trained on one domain and booking the other "
             "(domain swap); headline layer chosen within the training domain.",
             code="probing.family_probes", datasets=PROBE_DATASETS,
             params={"probe": "linear", "calibration_fraction": 0.2, "split_seed": 0}),
    Protocol("belief-probe-mlp@1",
             "MLP probe (256-128-64 ReLU) on last-token activations, domain swap, one rate per seed.",
             code="probing.family_probes", datasets=PROBE_DATASETS,
             params={"probe": "mlp", "mlp_seeds": 5, "calibration_fraction": 0.2, "split_seed": 0}),
    Protocol("belief-elicit-logprob@1",
             "Next-token True/False log-probabilities, each statement in its own prompt, over three templates.",
             code="coherence.elicitation", datasets=DUTCH_BOOK_DATASETS,
             params={"method": "logprob", "calibration_fraction": 0.2, "split_seed": 0}),
    Protocol("belief-elicit-stated@1",
             "Stated 0-100 probability that the statement is true, instruct models only.",
             code="coherence.elicitation", datasets=DUTCH_BOOK_DATASETS,
             params={"method": "stated", "calibration_fraction": 0.2, "split_seed": 0}),
    # Activation extraction, the input every probe protocol cites: probing.activations' last-token residual stream
    # at 8 relative depths, float16, per dataset.
    Protocol("activations-last-token@1",
             "Residual-stream activations at the last token before the final period, at relative depths 1/8..8/8.",
             code="probing.extract", datasets=ACTIVATION_DATASETS,
             params={"depths": [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0], "batch_size": 32,
                     "dtype": "float16"}),
    # Experiment 002's in-task beliefs: 001's elicitation methods on the umbrella_towns statements, chat prompts,
    # uncalibrated (each seed hides different chances of rain, so there is no ground truth to calibrate against).
    Protocol("town-belief-elicit-logprob@1",
             "Next-token True/False log-probabilities for each town statement, own prompt each, three templates.",
             code="coherence.measure_towns", datasets={"umbrella_towns": 1},
             params={"method": "logprob", "templates": ["t0", "t1", "t2"], "chat": True, "calibrated": False}),
    Protocol("town-belief-elicit-stated@1",
             "Stated 0-100 probability that each town statement is true, greedy-decoded; unparseable left empty.",
             code="coherence.measure_towns", datasets={"umbrella_towns": 1},
             params={"method": "stated", "templates": ["t0"], "chat": True, "calibrated": False}),
    Protocol("town-belief-probe-linear@1",
             "Logistic-regression probe trained on every statement 001's probes train on (atomic datasets plus both "
             "domains' base, negation and conjunction datasets), at every layer, applied to the town statements; "
             "the selected layer has the best accuracy on a held-out 20% of those statements. Uncalibrated.",
             code="probing.probe", datasets={name: 1 for name in ACTIVATION_DATASETS},
             params={"probe": "linear", "holdout_fraction": 0.2, "split_seed": 0, "calibrated": False}),
]}


def get(protocol_id: str) -> Protocol:
    try:
        return PROTOCOLS[protocol_id]
    except KeyError:
        raise KeyError(f"Unknown protocol {protocol_id!r}; registered: {', '.join(sorted(PROTOCOLS))}") from None
