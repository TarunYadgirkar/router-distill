import re

# Ported from Bonsai's kind space + Mahogany's regex classifier. Deliberately dumb — this is
# the floor every learned router must beat.

_KIND_PATTERNS = [
    ("comparison", re.compile(r"\b(vs|versus|compare|better|instead of|or should)\b", re.I)),
    ("reasoning", re.compile(r"\b(prove|why|implication|tradeoff|assess|risk|derive|complexity)", re.I)),
    ("code", re.compile(r"\b(code|function|bug|error|api|typescript|python|schema|regex|sql)\b", re.I)),
    ("creative", re.compile(r"\b(write a (story|poem|song)|rewrite|reword|tone|draft|slogan)\b", re.I)),
    ("synthesis", re.compile(r"\b(summariz|explain|overview|describe how|walk me through)", re.I)),
]

_DEEP = re.compile(
    r"\b(prove|design a|architect|trade-?offs?|multi|optimal|formally|strategy|roadmap|weigh)", re.I
)
_QUICK = re.compile(r"^(what|who|when|where|which|is|are|does|did|how (many|much|old))\b", re.I)


def classify_kind(question: str) -> str:
    for kind, pattern in _KIND_PATTERNS:
        if pattern.search(question):
            return kind
    return "lookup" if _QUICK.match(question.strip()) else "other"


def classify_tier(question: str) -> str:
    q = question.strip()
    if _DEEP.search(q) or len(q) > 300:
        return "deep"
    if _QUICK.match(q) and len(q) < 90:
        return "quick"
    return "thoughtful"


def route(question: str) -> dict:
    tier = classify_tier(question)
    return {"kind": classify_kind(question), "tier": tier}
