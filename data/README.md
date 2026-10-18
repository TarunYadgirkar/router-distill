# Data card

All files are JSON Lines (one object per line) except `train_exclusions.json`. Everything here was produced in one run on 2026-08-22 with the Anthropic API; nothing was edited by hand afterwards except where noted under "Changes since collection".

## Files

| file | rows | what it is |
|---|---|---|
| `questions.jsonl` | 108 | The eval set. Hand-written questions, held out from all training. |
| `questions_train.jsonl` | 475 | Synthetic training pool written by Claude Opus 5 (`scripts/expand_questions.py`). 7 rows are excluded at load time, leaving 468. |
| `train_exclusions.json` | 7 | Training rows that copy or paraphrase an eval question, with the eval id and the reason. |
| `outcomes/answers.jsonl` | 324 | Every eval question answered at every rung. |
| `outcomes/judged.jsonl` | 324 | A 0-10 score for every answer from Claude Opus 5. |
| `labels/questions.<teacher>.jsonl` | 108 each | Each teacher's tier distribution for each eval question. |
| `labels/questions_train.<teacher>.jsonl` | 475 each | Each teacher's tier distribution for each training question. |

Teachers are `claude-haiku-4-5`, `claude-sonnet-5` and `claude-opus-5`.

## Fields

**Questions** (`questions.jsonl`, `questions_train.jsonl`): `id`; `text`; `slice`, which for the eval set is one of `clean-quick` (24), `clean-thoughtful` (24), `clean-deep` (18), `ambiguous` (24) or `ood` (18), and for the training pool is `train-<tier>`, the tier the generator was asked to write; `kind_hint`, the intended question kind (lookup, synthesis, comparison, reasoning, code, creative, other); `source` (`curated` or `synthetic`).

**Answers** (`outcomes/answers.jsonl`): `question_id`, `tier`, `model`, `effort` (quick: none, Haiku 4.5 does not accept it; thoughtful: medium; deep: high), `answer` (text blocks only; thinking is not stored), `input_tokens`, `output_tokens`, `latency_ms` (wall clock with four concurrent requests and SDK retries), `cost_usd` (as recorded at collection, with Sonnet 5 priced at $3/$15; the analysis reprices from tokens), `max_tokens`. Rows written by the current scripts also carry `sample`, `stop_reason`, `response_id` and `created_at`.

**Judgments** (`outcomes/judged.jsonl`): `question_id`, `tier`, `score` (0-10, integers in practice), `punt` (the judge thinks the answer dodged or asked for context), `judge_cost_usd`, `rationale` (under 15 words). Missing fields default to `judge = claude-opus-5`, `sample = 0`, `judge_pass = 0`.

**Labels** (`labels/*.jsonl`): `question_id`, `teacher`, `kind`, `tier_probs` (quick / thoughtful / deep, normalised to sum to 1), `reason` (under 8 words), `cost_usd` (as recorded; token counts were not stored).

## How it was made

- Rungs used the system prompt in `src/router_distill/rungs.py`; the judge prompt is in `judge.py`; the teacher prompt is in `teacher.py`; the generator prompt is in `scripts/expand_questions.py`. The judge sees the question and one answer, not the model name.
- No temperature was set (Sonnet 5 and Opus 5 reject one), so every row is a single sample from a non-deterministic model.
- The generator was asked for an even spread over 7 kinds x 3 tiers; exact-duplicate texts were dropped, which is why ids have gaps.

## Known issues

- **max_tokens differs by row.** 303 answers were generated with `max_tokens` 4,000. 21 (19 deep, 2 quick) truncated under that cap, were dropped and were re-run at 16,000. No kept 4,000-token answer reached the cap. `cd-018` at the deep rung still hit 16,000 tokens, is cut off mid-proof, and scored 4. The `max_tokens` values were backfilled from git history (rows present in commit `0fe0acc` ran at 4,000, rows added in `a1de1be` at 16,000); `stop_reason` was not recorded for these rows.
- **The judge is Claude Opus 5**, which also wrote the deep answers, the training questions and one set of labels. Scores bunch at 8-10. Within a question the deep answer is usually both the longest and the best, so length and rung are confounded; REPORT.md section 6 gives the length effect with rung means removed.
- **Leakage handled by exclusion, not deletion.** `questions_train.jsonl` still contains the seven excluded rows and the label files still label them, so the files match what was paid for; `router_distill.data.load_train_pool()` drops them.
- **Paraphrase detection is manual.** The generator drops copies automatically (same normalized text or character n-gram cosine of 0.8 or more). Paraphrases share few characters, so `make leaks` (`scripts/find_leaks.py`) lists the closest eval question for every training question by character n-grams and by all-MiniLM-L6-v2 embedding cosine, and a person decides. syn-0439 (cosine 0.68 to ood-001, character similarity 0.27) was found this way after the first six exclusions; no other pair in the top 40 by either signal asks the same question.
- **Near-duplicates inside the training pool.** 12 pairs of the 468 training questions have character n-gram cosine above 0.6 (for example two awk one-liner requests). They were left in; they make the cross-validation agreement in REPORT.md slightly optimistic.
- **Ambiguous slice.** Questions such as "Why did my deploy fail?" have no right answer without context. The judge has no rule for scoring a justified request for more context; 19 of the 72 answers on this slice were flagged as punts.

## Changes since collection

- 2026-10-18: added the `max_tokens` field to every row of `outcomes/answers.jsonl` (values from git history, see above) and added `train_exclusions.json`. No answer, score or label was changed.
- 2026-10-18: excluded syn-0439, a paraphrase of ood-001 found in review.

## Cost

At the prices in effect (Sonnet 5 at $2/$10) the committed rows cost about $13.65: rungs $6.95, judge $3.01, labels $3.70. Question generation was not recorded. See REPORT.md section 8.

## License

Released with the code under the MIT license. The questions were written for this project; the answers, scores and labels are model outputs.
