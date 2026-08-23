import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Question:
    id: str
    text: str
    slice: str  # clean-quick | clean-thoughtful | clean-deep | ambiguous | ood
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

    @property
    def tier(self) -> str:
        return max(self.tier_probs, key=lambda t: self.tier_probs[t])


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
    cost_usd: float


@dataclass
class JudgedAnswer:
    question_id: str
    tier: str
    score: float  # 0-10
    punt: bool
    judge_cost_usd: float
    rationale: str = ""


def write_jsonl(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(asdict(row) if hasattr(row, "__dataclass_fields__") else row) + "\n")


def append_jsonl(path: Path, row: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(asdict(row) if hasattr(row, "__dataclass_fields__") else row) + "\n")


def read_jsonl(path: Path, cls: type | None = None) -> list[Any]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(cls(**obj) if cls else obj)
    return rows
