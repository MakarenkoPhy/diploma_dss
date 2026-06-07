"""
СППР для разметки чувствительных данных в банковской среде.

Гибридная архитектура:
  Фаза 1. LLM в роли ЛПР + метод ЦИКЛ → решающее правило.
  Фаза 2. LLM-оценщик описания столбца + lookup в правиле → класс чувствительности.
"""

from .criteria import (
    CRITERIA, CLASSES, omega, num_classes, total_space_size,
    Criterion, CriterionGrade, SensitivityClass,
)
from .llm_client import (
    LLMClient, AnthropicClient, OpenAIClient, DeepSeekClient,
    ManualClient, make_client,
)
from .tsikl import TsiklAlgorithm, DecisionMaker, TsiklState
from .decision_maker import LLMDecisionMaker, build_dm_system_prompt
from .assessor import (
    LLMAssessor, ColumnDescription, AssessmentResult,
    build_assessor_system_prompt,
)
from .decision_rule import DecisionRule, BuildMetadata, make_metadata
from .pipeline import (
    build_decision_rule, classify_column, classify_batch,
    ClassificationResult,
)

__all__ = [
    "CRITERIA", "CLASSES", "omega", "num_classes", "total_space_size",
    "Criterion", "CriterionGrade", "SensitivityClass",
    "LLMClient", "AnthropicClient", "OpenAIClient", "DeepSeekClient",
    "ManualClient", "make_client",
    "TsiklAlgorithm", "DecisionMaker", "TsiklState",
    "LLMDecisionMaker", "build_dm_system_prompt",
    "LLMAssessor", "ColumnDescription", "AssessmentResult",
    "build_assessor_system_prompt",
    "DecisionRule", "BuildMetadata", "make_metadata",
    "build_decision_rule", "classify_column", "classify_batch",
    "ClassificationResult",
]
