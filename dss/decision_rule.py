"""
Решающее правило — результат Фазы 1.
Lookup-таблица «вектор → класс», сохраняемая в JSON.
"""

from __future__ import annotations
import json
import math
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path

from .criteria import CRITERIA, CLASSES, omega, num_classes


@dataclass
class BuildMetadata:
    provider: str
    model: str
    built_at: str
    queries: int
    space_size: int
    omega: list
    num_classes: int
    notes: str = ""
    # Статистика качества LLM как ЛПР
    contradiction_count: int = 0   # суммарно обнаруженных противоречий (все итерации R)
    R_iterations: int = 0          # число итераций процедуры R
    clamp_count: int = 0           # ответов ЛПР, зажатых в [C^L, C^U]
    parse_retry_count: int = 0     # ответов, потребовавших повторного парсинга
    build_time_sec: float = 0.0    # время построения правила (секунды)


@dataclass
class DecisionRule:
    table: dict
    metadata: BuildMetadata
    history: list = field(default_factory=list)

    def lookup(self, vector: tuple) -> int:
        if vector not in self.table:
            raise KeyError(
                f"Вектор {vector} отсутствует в правиле. "
                f"Допустимые размерности: {self.metadata.omega}"
            )
        return self.table[vector]

    def save(self, path) -> None:
        path = Path(path)
        payload = {
            "metadata": asdict(self.metadata),
            "table": [
                {"vector": list(vec), "class": cls}
                for vec, cls in sorted(self.table.items())
            ],
            "history": self.history,
            "criteria": [
                {
                    "code": c.code,
                    "name": c.name,
                    "grades": [{"code": g.code, "title": g.title} for g in c.grades],
                }
                for c in CRITERIA
            ],
            "classes": [
                {"code": c.code, "name": c.name, "description": c.description}
                for c in CLASSES
            ],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "DecisionRule":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        meta = BuildMetadata(**data["metadata"])
        table = {tuple(item["vector"]): item["class"] for item in data["table"]}
        history = data.get("history", [])
        return cls(table=table, metadata=meta, history=history)

    def class_distribution(self) -> dict:
        dist: dict = {}
        for cls in self.table.values():
            dist[cls] = dist.get(cls, 0) + 1
        return dist

    def verify_consistency(self) -> list:
        violations = []
        vectors = list(self.table.keys())
        for i, x in enumerate(vectors):
            for y in vectors[i + 1:]:
                geq = all(a >= b for a, b in zip(x, y))
                strict = any(a > b for a, b in zip(x, y))
                if geq and strict:
                    if self.table[x] < self.table[y]:
                        violations.append((x, y, self.table[x], self.table[y]))
                geq2 = all(b >= a for a, b in zip(x, y))
                strict2 = any(b > a for a, b in zip(x, y))
                if geq2 and strict2:
                    if self.table[y] < self.table[x]:
                        violations.append((y, x, self.table[y], self.table[x]))
        return violations


def make_metadata(
    provider: str,
    model: str,
    queries: int,
    notes: str = "",
    contradiction_count: int = 0,
    R_iterations: int = 0,
    clamp_count: int = 0,
    parse_retry_count: int = 0,
    build_time_sec: float = 0.0,
) -> BuildMetadata:
    return BuildMetadata(
        provider=provider,
        model=model,
        built_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        queries=queries,
        space_size=math.prod(omega()) if omega() else 1,
        omega=list(omega()),
        num_classes=num_classes(),
        notes=notes,
        contradiction_count=contradiction_count,
        R_iterations=R_iterations,
        clamp_count=clamp_count,
        parse_retry_count=parse_retry_count,
        build_time_sec=build_time_sec,
    )
