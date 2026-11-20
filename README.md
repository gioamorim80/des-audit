# des-audit

A store's chat assistant can offer a shopper nothing, a discount or free shipping. If a model reads each chat and returns a few probabilities (is price the obstacle, has shipping cost come up), the store can ask whether those readings tell it whom an offer moves, or only who was going to buy anyway. `des-audit` answers that from the store's own randomized decision log. It is three commands over one file and depends on numpy and pandas.

## Install

```bash
pip install git+https://github.com/gioamorim80/des-audit
# or, from a clone:  uv sync --extra test   (or pip install -e ".[test]")
des-audit --help
```

Python 3.11 or later. For Parquet logs add the extra: `pip install "des-audit[parquet] @ git+https://github.com/gioamorim80/des-audit"`.

## Running the test in your store

1. Before you look at any data, fix the question, the cut and the direction: for example, "free shipping raises purchase more for chats whose shipping-cost reading is 0.5 or above than for the rest".
2. On a slice of chats, make the offer at random: a coin flip between the offer and no offer, decided after the chat is read and independently of it. Log the probability the offer had (0.5 for a fair coin). Keep one chat per shopper if you can; if shoppers come back, log a shopper id and pass it as `--cluster`.
3. Log, per chat, the reading, the offer made, whether the shopper bought, the margin of the sale and what the offer cost.
4. Run `des-audit calibration` on that log with the cut you fixed (`--bins 2` cuts at 0.5).

If the offer's effect is clearly larger where the reading is high, the reading carries targeting signal. If the effect on profit changes sign between the bins, the reading should change the decision.

## The log

One CSV or Parquet file, one row per decision:

| column | type | meaning |
| --- | --- | --- |
| `decision_id` | unique | one row per decision |
| `action` | text | the offer made; the control is named by `--control`, default `no_offer` |
| `propensity` | in (0, 1] | the probability of the offer made, logged when the decision was made |
| `outcome` | 0 or 1 | did the shopper buy |
| `revenue` | number | margin of the sale, before the offer's cost (0 without a sale) |
| `cost` | number | what the offer cost on this decision; `revenue − cost` is what the decision earned |
| `ctx_*` | numbers | what the store knew before the chat: cart value, pages viewed, history |
| `j_*` | in [0, 1] | the readings, one column per question; may be missing on some rows |
| anything else | any | ignored, unless named by `--cluster` |

Randomization with logged propensities and a control arm is what makes this work. Without randomization the offer a shopper got depends on things that also move purchase, and no estimator can separate the two. Without the logged propensity there is nothing to reweight by. Without a control there is no "what would have happened with no offer". A policy that explores with the probabilities written down produces a usable log; a policy that always picks its favourite does not.

The loader refuses an empty file, a missing required column, a missing value outside the `j_*` columns, a propensity outside (0, 1], an outcome other than 0 or 1, a reading outside [0, 1], a repeated `decision_id`, a single arm, or a control with no rows, and says which. A row missing a reading is dropped by the commands that use that reading, and both commands say how many they dropped (`calibration` per arm). If one arm misses the reading more often, the reading may have been taken after the offer, and the offer may have changed it.

## `calibration`: the offer's effect by reading

```bash
des-audit calibration --log log.csv --judgment j_price_is_obstacle --treatment discount_10 [--bins 2] [--cluster shopper_id]
```

On `examples/example_log.csv`, a made-up store of 12,000 chats whose truth is stated in `examples/make_example_log.py` (10% off adds 10 points of purchase probability for a price-sensitive shopper and 2 for anyone else), from the `examples` folder:

```text
$ des-audit calibration --log example_log.csv --judgment j_price_is_obstacle --treatment discount_10 --cluster shopper_id

effect of discount_10 against no_offer by j_price_is_obstacle: 8,954 decisions, self-normalized IPW, 95% percentile-bootstrap intervals from 2,000 resamples of clusters of shopper_id (seed 0)

score         n control  n treatment          control rate        purchase effect        profit effect  flag
------------  ---------  -----------  --------------------  ---------------------  -------------------  ----
[0.00, 0.50)      3,552        3,292  0.144 [0.133, 0.156]  0.015 [-0.003, 0.033]  -0.73 [-1.91, 0.47]
[0.50, 1.00]      1,066        1,044  0.130 [0.111, 0.152]   0.088 [0.056, 0.121]    3.56 [1.38, 5.81]

last bin minus first: purchase effect 0.073 [0.035, 0.111], profit effect 4.29 [1.80, 6.79]
This log detects a difference in purchase effect of 0.053 with 80% power (two-sided, alpha 0.05); a true difference of 0.073 would need about 4,707 decisions in these two arms.
```

Where the price reading is 0.5 or above, 10% off raises purchase by 8.8 points and earns $3.56 more per chat, an interval clear of zero. Below it, the discount raises purchase by 1.5 points and earns $0.73 less, an interval that includes zero. The difference between the bins, 7.3 points of purchase and $4.29 per chat, is clear of zero: this reading should decide who gets the discount.

In each bin the purchase rate and the profit per decision under each arm are estimated by self-normalized inverse-propensity weighting: Σ (y/π) ÷ Σ (1/π) over the arm's rows. With a constant propensity, as in a coin flip, that is the plain mean, and the effect is the difference in means. The effect in a bin is a group average treatment effect, E[Y(1) − Y(0) | reading in the bin]: the effect of making the offer, within the purchase window your log uses.

All the intervals come from one percentile bootstrap that resamples decisions, or whole clusters with `--cluster`, 2,000 times by default, so the per-bin effects and their difference are drawn together. The detectable difference is (z(1 − α/2) + z(power)) times the delta-method standard error of the difference, which carries no resampling noise; with a coin flip that standard error is the square root of Σ p(1 − p)/n over the four bin-by-arm cells.

The command checks what would make those numbers wrong:

- **Rows per arm.** Adjacent bins are merged until each holds at least 30 rows of each arm. A bin with 500 control rows and one treated row would otherwise get a confident interval that only reflects the control.
- **Resamples that could not compute.** A bin's interval is withheld, with a flag, when its effect could not be computed in more than 1% of resamples.
- **Positivity.** In a two-arm log, a row whose propensity is above 0.99 had (almost) no chance of the other arm, and the arms are then compared on different shoppers; the command warns. In any log it computes, per bin and arm, Σ (1/π) over the arm's rows divided by the bin's rows (in the `--json` output). With positivity that is about 1; the command warns when it is off by more than 0.1. With more than two arms the other arms' probabilities are not in the log, and this balance is the only check.
- **Small propensities.** A propensity below 0.01 (a weight above 100) is a warning: a few decisions carry the estimate and the intervals understate it.

Equal-width bin edges are k / bins, so a reading of exactly 0.3 at `--bins 10` lands in [0.3, 0.4). `--bins 10` shows the shape of the effect across the reading; a reading that only tracks who buys shows a control rate that rises with the reading and an effect that does not.

Fix the cut before you read the log. On a log where the cut at 0.5 does not separate the effects, some other cut often will by chance, and choosing it after looking turns a test into a search. If you test several readings, correct for it (Holm's method, for example) or choose on one half of the log and test on the other.

## `value`: what deciding with the readings is worth

```bash
des-audit value --log log.csv [--judgments j_a,j_b] [--all] [--holdout 0.5] [--cluster shopper_id]
```

The decisions are split at random (by cluster with `--cluster`, so a shopper is never on both sides). On one part a greedy policy is fit per set of inputs: one ridge regression of `revenue − cost` per arm, the policy picking the arm with the highest prediction. The sets are the context alone, the context plus each reading named in `--judgments`, and with `--all` the context plus every reading. On the other part each policy is valued by self-normalized inverse-propensity weighting and by the doubly robust estimator of Dudík, Langford and Li (2011), and compared with the context-only policy by a paired bootstrap of the difference. Valuing a policy on decisions it was not fit on is what stops a reading from looking useful by fitting noise.

On the example log:

```text
$ des-audit value --log example_log.csv --all --cluster shopper_id

value of deciding with the judgments: greedy ridge policies over 3 arms, fit on 5,870 decisions and valued on the other 6,009; differences against the context-only policy in revenue − cost per decision, 95% paired bootstrap intervals from 2,000 resamples of clusters of shopper_id (seed 0)

policy                 matched      IPW difference       DR difference
---------------------  -------  ------------------  ------------------
ctx + all 3 judgments    1,687  0.14 [-0.35, 0.62]  0.16 [-0.31, 0.63]

The context-only policy is worth 9.39 (IPW) and 9.64 (DR) per decision.
121 decisions missing a judgment in use were dropped.
```

`calibration` found a clear difference in this log, and `value` cannot tell deciding with the readings from deciding without them. Both are right. The best possible policy in this store gains about $2 per decision over offering nothing, against a reward whose standard deviation is about $25, and a linear policy fit on 6,000 decisions captures only part of that gain. A difference in profit per decision is much harder to detect than a difference in a purchase rate between two bins; `power --sigma` says how much harder.

## `power`: decisions a test needs

```bash
des-audit power --delta 0.02 --rate 0.17 --high-share 0.25        # a difference in effect between bins
des-audit power --delta 1 --sigma 25 [--arms 2]                   # a difference in mean profit per decision
```

An estimate with standard error c/√n detects a true difference Δ with probability `power` in a two-sided test at level α when n = c² (z(1 − α/2) + z(power))² / Δ². For a difference in an offer's effect between the high and the low bin, with purchase rate p, a share h of decisions high and a share q treated, c² = p(1 − p) / (q(1 − q)) × (1/h + 1/(1 − h)). With p = 0.17, h = 0.25, q = 0.5 and Δ = 2 points that is 3.01 × 7.85 / 0.0004, about 59,100 chats. For a difference in mean profit between two arms with standard deviation σ, c² = 2σ² per arm. Halving Δ quadruples n.

Every command prints a table, with its warnings on stderr, or one JSON object with `--json`. Exit codes: 0 done, 1 the log or a request was refused (the reason is on stderr), 2 usage.

## Limitations

The estimators assume the store was the same across the log: the same offers, the same economics, no change to the chat in between. Purchases are whatever the log attributes to the decision in its window, so an offer that pulls a purchase forward from next month looks like a gain. Interference between shoppers is assumed away; a shopper who chats several times and is treated more than once is handled by `--cluster` for the intervals, not for carry-over between chats. The reading must come from text written before the offer, or the offer can change it. The bootstrap treats the logged propensities as known. `value`'s policies are linear in the readings, so a signal that lives in an interaction or a threshold is missed, and its result moves with the random split at a few thousand decisions; rerun it with a few seeds before trusting one. A log collected by an adaptive policy, such as a bandit, needs adaptive weights for valid intervals (Hadad et al. 2021); these intervals assume the propensities did not depend on earlier outcomes.

## References

Bottou, L., Peters, J., Quiñonero-Candela, J., Charles, D. X., Chickering, D. M., Portugaly, E., Ray, D., Simard, P. and Snelson, E. (2013). Counterfactual reasoning and learning systems: the example of computational advertising. *Journal of Machine Learning Research* 14.

Dudík, M., Langford, J. and Li, L. (2011). Doubly robust policy evaluation and learning. *Proceedings of the 28th International Conference on Machine Learning*.

Egami, N., Fong, C. J., Grimmer, J., Roberts, M. E. and Stewart, B. M. (2022). How to make causal inferences using texts. *Science Advances* 8(42).

Hadad, V., Hirshberg, D. A., Zhan, R., Wager, S. and Athey, S. (2021). Confidence intervals for policy evaluation in adaptive experiments. *Proceedings of the National Academy of Sciences* 118(15).

## License

MIT. See `LICENSE`.
