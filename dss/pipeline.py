"""
Pipeline: координирует две фазы работы СППР.

Фаза 1 (build_decision_rule):
    DM = LLMDecisionMaker(llm)
    rule = TsiklAlgorithm(omega, M, DM).run()
    rule.save("rule.json")

Фаза 2 (classify_column):
    rule = DecisionRule.load("rule.json")
    assessor = LLMAssessor(llm)
    assessment = assessor.assess(column_description)
    cls = rule.lookup(assessment.vector)
"""

from __future__ import annotations
from dataclasses import dataclass

from .criteria import CRITERIA, CLASSES, omega, num_classes
from .tsikl import TsiklAlgorithm
from .decision_maker import LLMDecisionMaker
from .assessor import LLMAssessor, ColumnDescription, AssessmentResult
from .decision_rule import DecisionRule, make_metadata
from .llm_client import LLMClient

import time


# ---------------------------------------------------------------------------
# Фаза 1
# ---------------------------------------------------------------------------

def build_decision_rule(
    llm: LLMClient,
    *,
    progress=None,
    notes: str = "",
) -> DecisionRule:
    """Построить решающее правило, используя LLM как ЛПР."""
    t0 = time.time()
    dm = LLMDecisionMaker(llm)
    algo = TsiklAlgorithm(
        omega=omega(),
        M=num_classes(),
        decision_maker=dm,
        progress_callback=progress,
    )
    table = algo.run()
    elapsed = round(time.time() - t0, 1)

    rule = DecisionRule(
        table=table,
        metadata=make_metadata(
            provider=llm.name,
            model=getattr(llm, "model", "n/a"),
            queries=len(algo.state.query_log),
            notes=notes,
            contradiction_count=algo.state.contradiction_count,
            R_iterations=algo.state.R_iterations,
            clamp_count=algo.state.clamp_count,
            parse_retry_count=dm.parse_retry_count,
            build_time_sec=elapsed,
        ),
        history=dm.history,
    )
    return rule


# ---------------------------------------------------------------------------
# Фаза 2
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    column: ColumnDescription
    assessment: AssessmentResult
    sensitivity_class: int
    class_code: str
    class_name: str

    def as_dict(self) -> dict:
        return {
            "field_name": self.column.field_name,
            "table_name": self.column.table_name,
            "description": self.column.description,
            "vector": list(self.assessment.vector),
            "vector_labeled": {
                c.code: f"{c.grades[v - 1].code} ({c.grades[v - 1].title})"
                for c, v in zip(CRITERIA, self.assessment.vector)
            },
            "class": self.sensitivity_class,
            "class_code": self.class_code,
            "class_name": self.class_name,
            "reasoning": self.assessment.reasoning,
        }


def classify_column(
    column: ColumnDescription,
    rule: DecisionRule,
    assessor: LLMAssessor,
) -> ClassificationResult:
    assessment = assessor.assess(column)
    cls = rule.lookup(assessment.vector)
    cls_obj = CLASSES[cls - 1]
    return ClassificationResult(
        column=column,
        assessment=assessment,
        sensitivity_class=cls,
        class_code=cls_obj.code,
        class_name=cls_obj.name,
    )


def classify_batch(
    columns: list,
    rule: DecisionRule,
    assessor: LLMAssessor,
    progress=None,
) -> list:
    results = []
    for i, col in enumerate(columns, start=1):
        if progress:
            progress(f"[{i}/{len(columns)}] {col.field_name or col.description[:50]}")
        try:
            results.append(classify_column(col, rule, assessor))
        except Exception as e:
            if progress:
                progress(f"  ⚠ Пропущено ({col.field_name}): {e}")
        time.sleep(0.5)  # небольшая пауза между запросами
    return results
