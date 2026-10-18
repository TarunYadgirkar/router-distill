import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import JUDGE_MODEL, TIERS


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class Question:
    id: str
    text: str
    slice: str  # clean-quick | clean-thoughtful | clean-deep | ambiguous | ood | train-<tier>
    kind_hint: str | None = None
    source: str = "curated"  # curated | synthetic


@dataclass
class TeacherLabel:
    question_id: str
    teacher: str
    kind: str
    tier_probs: dict[str, float]  # quick/thoughtful/deep, sums to 1
    reason: str
    cost_usd: float
    stop_reason: str | None = None
    response_id: str | None = None
    created_at: str | None = None

    @property
    def tier(self) -> str:
        # Ties escalate: when the teacher is split, the more capable tier is the safer route.
        return max(TIERS, key=lambda t: (self.tier_probs[t], TIERS.index(t)))


@dataclass
class RungAnswer:
    question_id: str
    tier: str
    model: str
    effort: str | None
    answer: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    cost_usd: float  # at the prices in effect when the row was written; analysis reprices from tokens
    sample: int = 0
    max_tokens: int | None = None
    stop_reason: str | None = None
    response_id: str | None = None
    created_at: str | None = None

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.question_id, self.tier, self.sample)


@dataclass
class JudgedAnswer:
    question_id: str
    tier: str
    score: float  # 0-10
    punt: bool
    judge_cost_usd: float
    rationale: str = ""
    judge: str = JUDGE_MODEL
    sample: int = 0
    judge_pass: int = 0
    stop_reason: str | None = None
    response_id: str | None = None
    created_at: str | None = None

    @property
    def key(self) -> tuple[str, str, int, str, int]:
        return (self.question_id, self.tier, self.sample, self.judge, self.judge_pass)


def _encode(row: Any) -> str:
    return json.dumps(asdict(row) if hasattr(row, "__dataclass_fields__") else row)


def write_jsonl(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(_encode(row) + "\n")


def append_jsonl(path: Path, row: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(_encode(row) + "\n")
        f.flush()


def read_jsonl(path: Path, cls: type | None = None) -> list[Any]:
    if not path.exists():
        return []
    lines = [line.strip() for line in path.read_text().splitlines()]
    lines = [line for line in lines if line]
    rows = []
    for i, line in enumerate(lines):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            # Rows are appended one at a time, so a killed run can leave a half-written last line.
            if i == len(lines) - 1:
                print(f"warning: ignoring truncated last line of {path}", file=sys.stderr)
                break
            raise
        rows.append(cls(**obj) if cls else obj)
    return rows


def drop_truncated_tail(path: Path) -> None:
    """Remove a half-written last line so appends after a killed run start on a clean line."""
    if not path.exists():
        return
    text = path.read_text()
    if not text or text.endswith("\n"):
        return
    head, _, tail = text.rpartition("\n")
    try:
        json.loads(tail)
        path.write_text(text + "\n")
    except json.JSONDecodeError:
        print(f"warning: dropping truncated last line of {path}", file=sys.stderr)
        path.write_text(head + "\n" if head else "")
