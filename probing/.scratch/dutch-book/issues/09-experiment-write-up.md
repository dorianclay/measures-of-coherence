# Write up the experiments and every decision behind them

Type: task (HITL)
Status: open
Blocked by: 05, 06, 07, 08

## Question

Once the experiments have run, write a thorough account of them that someone could reproduce and defend. It must detail **every decision on this map**: what was decided, the alternatives, and why. That covers:
- the rate-of-loss formulation;
- calibration;
- event-family construction, including drops and label inconsistencies;
- the leakage protocol;
- elicitation;
- models, layers, and seeds;
- aggregation and statistics.

The write-up also covers methods, results (figures and tables for both prevision sources, both family shapes, and all controls), limitations, and the fixes to thesis §4.1.2. Those fixes include the notation corrections found by [Which rate-of-loss formulation(s) do we compute?](01-rate-of-loss-formulation.md) and the worked "Dutch book-ability" example the thesis leaves blank.

Source every decision from the map's resolved tickets and link each one. The write-up restates their content for a reader who never saw the map.

Also blocked by the analysis tickets that will graduate from the map's fog (running elicitation at scale, aggregation and comparison); add them to `Blocked by` when they are created. Worked with the author, who reviews the draft: format and home (repo doc, thesis chapter section, or both) are decided at the start of this ticket.
