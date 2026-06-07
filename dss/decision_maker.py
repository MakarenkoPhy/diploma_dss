"""
Фаза 1. LLM в роли ЛПР для метода ЦИКЛ.

LLM получает гипотетический объект в виде вектора вербальных оценок
по критериям и относит его к одному из M классов чувствительности.
Системный промпт собирается автоматически из системы критериев, классов
и методологических блоков (см. expert_guidance).
"""

from __future__ import annotations
import re

from .criteria import CRITERIA, CLASSES, Criterion, SensitivityClass
from .llm_client import LLMClient
from .tsikl import DecisionMaker
from . import expert_guidance as eg


SYSTEM_PROMPT_TEMPLATE = """\
Ты выступаешь в роли эксперта по информационной безопасности банка и принимаешь \
решение о классе чувствительности гипотетического объекта данных.

Объект описывается набором вербальных оценок по {n_criteria} критериям. \
Каждая оценка выбрана из упорядоченной шкалы; ПЕРВАЯ градация означает \
НАИБОЛЬШУЮ выраженность чувствительности по этому критерию, ПОСЛЕДНЯЯ — наименьшую.

═════════════════════════════════════════════════════════════════════════
КРИТЕРИИ
═════════════════════════════════════════════════════════════════════════

{criteria_block}

═════════════════════════════════════════════════════════════════════════
КЛАССЫ ЧУВСТВИТЕЛЬНОСТИ (от наиболее чувствительного к наименее)
═════════════════════════════════════════════════════════════════════════

{classes_block}

═════════════════════════════════════════════════════════════════════════
{heuristic}
═════════════════════════════════════════════════════════════════════════

{precaution}

═════════════════════════════════════════════════════════════════════════

ТВОЯ ЗАДАЧА: получив вектор оценок (К₁..К₆), отнести объект к одному из \
{n_classes} классов, применяя приведённую выше логику соотнесения.

ВАЖНО:
1. Соблюдай МОНОТОННОСТЬ: если объект A по всем критериям имеет не более \
сильные оценки чувствительности, чем объект B, то класс A должен быть \
не более чувствительным, чем класс B (т.е. номер класса A ≥ номер класса B).
2. При сомнениях применяй принцип осторожности.
3. Отвечай ТОЛЬКО ОДНИМ ЧИСЛОМ — номером класса от 1 до {n_classes}. \
Никаких пояснений, преамбул, знаков препинания.
"""


def _format_criterion(c: Criterion) -> str:
    lines = [f"{c.code}. {c.name}"]
    for i, g in enumerate(c.grades, start=1):
        lines.append(f"  {i}. [{g.code}] {g.title}. {g.description} Например: {g.examples}")
    return "\n".join(lines)


def _format_class(i: int, cls: SensitivityClass) -> str:
    return f"  Класс {i}. [{cls.code}] {cls.name}. {cls.description}"


def build_dm_system_prompt() -> str:
    criteria_block = "\n\n".join(_format_criterion(c) for c in CRITERIA)
    classes_block = "\n".join(_format_class(i + 1, c) for i, c in enumerate(CLASSES))
    return SYSTEM_PROMPT_TEMPLATE.format(
        n_criteria=len(CRITERIA),
        n_classes=len(CLASSES),
        criteria_block=criteria_block,
        classes_block=classes_block,
        heuristic=eg.HEURISTIC_RULE,
        precaution=eg.PRECAUTION_PRINCIPLE,
    )


def _format_object(vector: tuple) -> str:
    lines = ["Объект для классификации:"]
    for c, val in zip(CRITERIA, vector):
        grade = c.grades[val - 1]
        lines.append(f"  • {c.code} ({c.name}): {grade.code} — {grade.title}")
    lines.append("")
    lines.append(
        f"К какому из {len(CLASSES)} классов следует отнести этот объект? "
        f"Ответь одним числом от 1 до {len(CLASSES)}."
    )
    return "\n".join(lines)


_NUMBER_RE = re.compile(r"\d+")


def _parse_class(text: str, M: int):
    for token in _NUMBER_RE.findall(text):
        try:
            v = int(token)
        except ValueError:
            continue
        if 1 <= v <= M:
            return v
    return None


class LLMDecisionMaker(DecisionMaker):
    """LLM, выступающая в роли ЛПР для метода ЦИКЛ."""

    def __init__(self, llm: LLMClient, max_retries: int = 2):
        self.llm = llm
        self.max_retries = max_retries
        self.M = len(CLASSES)
        self.system_prompt = build_dm_system_prompt()
        self.history: list = []
        self.parse_retry_count: int = 0

    def classify(self, vector: tuple) -> int:
        prompt = _format_object(vector)

        last_response = ""
        for attempt in range(self.max_retries + 1):
            response = self.llm.complete(
                system=self.system_prompt,
                user=prompt,
                max_tokens=20,
            )
            last_response = response
            cls = _parse_class(response, self.M)
            if cls is not None:
                self.history.append({
                    "vector": list(vector),
                    "response": response,
                    "class": cls,
                    "attempt": attempt,
                })
                if attempt > 0:
                    self.parse_retry_count += 1
                return cls
            prompt = (
                _format_object(vector)
                + f"\n\nПРЕДЫДУЩИЙ ОТВЕТ '{response[:50]}' НЕ СОДЕРЖАЛ КОРРЕКТНОГО ЧИСЛА. "
                + f"Ответь СТРОГО одним целым числом от 1 до {self.M} и ничего больше."
            )

        raise ValueError(
            f"LLM не вернул корректный класс за {self.max_retries + 1} попыток. "
            f"Последний ответ: {last_response!r}"
        )
