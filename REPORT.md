# Router regret report

Scored questions: 108 (all 3 rungs answered and judged).
Regret = oracle mean score − policy mean score. Oracle picks the best-scoring rung per question, cheapest on ties.

| router | mean score | regret | mean cost/q | score per $ | latency ms | n |
|---|---|---|---|---|---|---|
| oracle | 9.38 | 0.0 | $0.0433 | 216.5 | 26344 | 108 |
| always-deep | 9.343 | 0.037 | $0.0544 | 171.6 | 31888 | 108 |
| sft:claude-opus-5 | 9.167 | 0.213 | $0.0521 | 175.8 | 30707 | 108 |
| distilled:claude-opus-5 | 9.157 | 0.222 | $0.0507 | 180.7 | 29909 | 108 |
| sft:claude-haiku-4-5 | 9.12 | 0.259 | $0.0501 | 181.9 | 29635 | 108 |
| distilled:claude-haiku-4-5 | 9.037 | 0.343 | $0.0490 | 184.3 | 29084 | 108 |
| sft:claude-sonnet-5 | 8.898 | 0.481 | $0.0447 | 199.3 | 26814 | 108 |
| teacher:claude-haiku-4-5 | 8.88 | 0.5 | $0.0378 | 235.1 | 23086 | 108 |
| distilled:claude-sonnet-5 | 8.88 | 0.5 | $0.0426 | 208.4 | 25720 | 108 |
| teacher:claude-opus-5 | 8.824 | 0.556 | $0.0373 | 236.7 | 22848 | 108 |
| teacher:claude-sonnet-5 | 8.806 | 0.574 | $0.0354 | 248.9 | 21786 | 108 |
| always-thoughtful | 8.611 | 0.769 | $0.0107 | 806.0 | 9001 | 108 |
| heuristic | 8.602 | 0.778 | $0.0282 | 305.3 | 17909 | 108 |
| always-quick | 7.296 | 2.083 | $0.0028 | 2642.8 | 5779 | 108 |

## Regret by slice

| router | ambiguous | clean-deep | clean-quick | clean-thoughtful | ood |
|---|---|---|---|---|---|
| oracle | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| always-deep | 0.042 | 0.111 | 0.042 | 0.0 | 0.0 |
| sft:claude-opus-5 | 0.375 | 0.111 | 0.417 | 0.0 | 0.111 |
| distilled:claude-opus-5 | 0.417 | 0.111 | 0.417 | 0.0 | 0.111 |
| sft:claude-haiku-4-5 | 0.458 | 0.111 | 0.542 | 0.0 | 0.111 |
| distilled:claude-haiku-4-5 | 0.667 | 0.111 | 0.625 | 0.042 | 0.167 |
| sft:claude-sonnet-5 | 0.792 | 0.111 | 0.583 | 0.292 | 0.556 |
| teacher:claude-haiku-4-5 | 0.458 | 0.111 | 0.542 | 0.458 | 0.944 |
| distilled:claude-sonnet-5 | 0.792 | 0.111 | 0.625 | 0.292 | 0.611 |
| teacher:claude-opus-5 | 0.75 | 0.111 | 0.542 | 0.542 | 0.778 |
| teacher:claude-sonnet-5 | 0.625 | 0.111 | 0.542 | 0.583 | 1.0 |
| always-thoughtful | 0.833 | 1.056 | 0.458 | 0.583 | 1.056 |
| heuristic | 1.167 | 0.444 | 0.583 | 0.583 | 1.111 |
| always-quick | 1.917 | 3.222 | 0.542 | 1.958 | 3.389 |

## Reading it

- `always-deep` is the quality ceiling money can buy; its regret is what judge noise plus occasional small-model wins look like.
- The interesting rows are ambiguous/ood slice regret: clean slices are easy for everyone.
- Teacher rows are live-routing cost-free here (label cost not included in cost/q); student rows route for ~free at inference.
