# router-distill

Can a tiny model imitate a frontier router's routing decisions — and is the model that's best
at a task necessarily the best *teacher* for routing that task?

Grew out of Bonsai's model+effort router and Mahogany's outcome-recorded provider routing.

## The experiment

1. **Outcome grid** (ground truth): every curated eval question answered at all three rungs
   (haiku/quick, sonnet/thoughtful, opus/deep), each answer scored 0–10 by a judge.
   Regret is measured against the per-question oracle (best score, cheapest on ties).
2. **Teachers**: haiku, sonnet, and opus each act as a routing classifier, emitting a soft
   probability distribution over tiers per question.
3. **Students**: per teacher, a TF-IDF + softmax-regression classifier trained on a synthetic
   question pool — **SFT** on the teacher's argmax label, **distilled** on the full soft
   distribution. Same features, same optimizer; the target is the only difference.
4. **Report**: regret and quality-per-dollar for fixed baselines, the heuristic router, each
   live teacher, and each student — overall and per slice (clean / ambiguous / OOD).

The multi-teacher comparison is the research question: if the strongest solver rates
everything "easy", its routing labels under-discriminate, and a mid-tier teacher may distill
into a better router.

## Setup

```
uv sync
touch .env && open -e .env   # paste: ANTHROPIC_API_KEY=sk-ant-...
```

Scripts load `.env` themselves; nothing needs exporting.

## Pipeline

```
# ground truth (curated set, data/questions.jsonl)
uv run scripts/run_rungs.py
uv run scripts/judge_answers.py

# teacher labels on the eval set (live-teacher rows in the report)
uv run scripts/label_teacher.py --teacher claude-haiku-4-5
uv run scripts/label_teacher.py --teacher claude-sonnet-5
uv run scripts/label_teacher.py --teacher claude-opus-5

# synthetic training pool + teacher labels on it + students
uv run scripts/expand_questions.py --n 500
uv run scripts/label_teacher.py --teacher claude-opus-5 --questions data/questions_train.jsonl
uv run scripts/train_students.py --teacher claude-opus-5
# (repeat the last two per teacher for the multi-teacher comparison)

uv run scripts/report.py    # writes REPORT.md
```

Every script is resumable (existing rows are skipped) and prints running cost. `--limit N`
runs a cheap subset first.

## Label space

Bonsai's: tier `quick | thoughtful | deep` (→ model + effort), kind
`lookup | synthesis | comparison | reasoning | code | creative | other`. Students predict tier;
kind is stored for slice analysis. Anthropic-only for now; the config's per-model spec table is
where an OpenAI rung/teacher would slot in.
