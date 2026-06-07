"""
Фаза 2. LLM-оценщик: оценивает реальный столбец данных по критериям.

В отличие от Фазы 1 (где LLM играла роль ЛПР и принимала решение о классе),
здесь LLM выполняет ИСКЛЮЧИТЕЛЬНО роль оценщика: получает текстовое описание
столбца (4 атрибута: имя таблицы, описание таблицы, имя поля, описание поля)
→ возвращает вектор оценок по 6 критериям + reasoning. Класс определяется
детерминированно через lookup в правиле, построенном на Фазе 1.

Системный промпт собирается из методологических блоков (см. expert_guidance),
полностью соответствующих инструкции эксперта-оценщика v3.
"""

from __future__ import annotations
import json
import re
import time
from dataclasses import dataclass

from .criteria import CRITERIA, Criterion
from .llm_client import LLMClient
from . import expert_guidance as eg


SYSTEM_PROMPT_TEMPLATE = """\
Ты — эксперт по информационной безопасности банка. Твоя задача — оценить \
описание столбца банковской информационной системы по {n_criteria} критериям \
чувствительности и вернуть результат в строгом формате JSON.

═════════════════════════════════════════════════════════════════════════
{object_block}
═════════════════════════════════════════════════════════════════════════

{subject_types}

═════════════════════════════════════════════════════════════════════════
{role_block}

═════════════════════════════════════════════════════════════════════════
{operational_block}

═════════════════════════════════════════════════════════════════════════
КРИТЕРИИ ОЦЕНКИ
═════════════════════════════════════════════════════════════════════════

{criteria_block}

═════════════════════════════════════════════════════════════════════════
{precaution}

═════════════════════════════════════════════════════════════════════════
{examples}
═════════════════════════════════════════════════════════════════════════

ПОРЯДОК РАБОТЫ

1. Прочитай все четыре атрибута столбца (имя и описание таблицы, имя и описание поля).
2. Определи, какие данные ТИПИЧНО хранятся в столбце с таким описанием в \
банковской среде. При неоднозначности описания поля опирайся на описание таблицы.
3. Определи тип субъекта данных (ФЛ-взрослый, ребёнок, ИП, ЮЛ, сотрудник, \
системная сущность) — совместно по всем четырём атрибутам, с учётом маркеров \
в названиях таблицы и поля.
4. Определи роль столбца в реляционной структуре (внешний или внутренний \
идентификатор, технический идентификатор события, содержательный атрибут, \
справочный атрибут или параметр продукта).
5. При конфликте источников применяй иерархию приоритета: \
описание поля > описание таблицы > название поля > название таблицы.
6. Последовательно оцени столбец по каждому из шести критериев.

ФОРМА ОТВЕТА

Сначала КРАТКО РАССУЖДАЙ в свободной форме (несколько строк): тип субъекта \
(с маркером, если он сыграл роль) → роль столбца → проход по шести критериям \
(по одной короткой фразе на критерий). Это рассуждение помогает точности и не \
обязано быть кратким в ущерб содержанию.

ЗАТЕМ заверши ответ ЕДИНСТВЕННЫМ валидным JSON-блоком в формате ниже. \
Парсится ТОЛЬКО последний JSON-блок в ответе, поэтому рассуждение перед ним \
не помешает. JSON обязан быть синтаксически корректным и полным.

ФОРМАТ ИТОГОВОГО JSON (строго валидный JSON, без markdown-обёртки):

{{
  "K1": <число от 1 до {w1}>,
  "K2": <число от 1 до {w2}>,
  "K3": <число от 1 до {w3}>,
  "K4": <число от 1 до {w4}>,
  "K5": <число от 1 до {w5}>,
  "K6": <число от 1 до {w6}>,
  "reasoning": "<итоговое обоснование на русском: тип субъекта, роль столбца, ключевые аргументы по критериям>"
}}

Помни: оценка 1 — НАИБОЛЬШАЯ выраженность чувствительности по критерию. \
Не подгоняй оценки под заранее выбранный класс — класс определит решающее \
правило по твоему вектору.
"""


def _format_criterion_for_assessor(c: Criterion) -> str:
    lines = [f"{c.code}. {c.name}"]
    for i, g in enumerate(c.grades, start=1):
        lines.append(f"  {i} = [{g.code}] {g.title}. {g.description}")
        lines.append(f"      Например: {g.examples}")
    return "\n".join(lines)


def build_assessor_system_prompt() -> str:
    criteria_block = "\n\n".join(_format_criterion_for_assessor(c) for c in CRITERIA)
    o = tuple(c.omega for c in CRITERIA)
    return SYSTEM_PROMPT_TEMPLATE.format(
        n_criteria=len(CRITERIA),
        object_block=eg.OBJECT_OF_ASSESSMENT,
        subject_types=eg.SUBJECT_TYPES,
        role_block=eg.ROLE_OF_COLUMN,
        operational_block=eg.OPERATIONAL_CRITICALITY,
        criteria_block=criteria_block,
        precaution=eg.PRECAUTION_PRINCIPLE,
        examples=eg.EXAMPLES_FULL,
        w1=o[0], w2=o[1], w3=o[2], w4=o[3], w5=o[4], w6=o[5],
    )


# ---------------------------------------------------------------------------
# Описание столбца — структура входа на Фазе 2
# ---------------------------------------------------------------------------

@dataclass
class ColumnDescription:
    """Описание оцениваемого столбца — 4 атрибута из инструкции."""
    description: str                # описание поля (field_describe) — основной
    table_name: str | None = None   # название таблицы (mart_name)
    table_description: str | None = None  # описание таблицы (table_desc / mart_desc)
    field_name: str | None = None   # название поля (field_nm)
    sample_values: str | None = None  # примеры значений (опционально)

    def to_user_prompt(self) -> str:
        parts = ["Столбец данных для оценки:"]
        parts.append(f"  • Название таблицы: {self.table_name or '(не указано)'}")
        parts.append(f"  • Описание таблицы: {self.table_description or '(не указано)'}")
        parts.append(f"  • Название поля: {self.field_name or '(не указано)'}")
        parts.append(f"  • Описание поля: {self.description}")
        if self.sample_values:
            parts.append(f"  • Примеры значений: {self.sample_values}")
        parts.append("")
        parts.append("Оцени столбец по 6 критериям и верни ответ в формате JSON, как описано в инструкции.")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Парсер ответа
# ---------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # Сначала пробуем весь текст целиком (идеальный случай)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Берём ПОСЛЕДНИЙ валидный JSON-блок — модель могла написать рассуждение
    # с фрагментами JSON до финального ответа, нам нужен именно финальный.
    last_valid = None
    for match in _JSON_BLOCK_RE.finditer(text):
        try:
            last_valid = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
    if last_valid is not None:
        return last_valid
    raise ValueError(f"Не удалось извлечь JSON из ответа: {text[:200]!r}")


@dataclass
class AssessmentResult:
    vector: tuple
    reasoning: str
    raw_response: str

    def vector_dict(self) -> dict:
        return {c.code: v for c, v in zip(CRITERIA, self.vector)}


class LLMAssessor:
    """LLM в роли оценщика на этапе оперативной классификации."""

    def __init__(self, llm: LLMClient, max_retries: int = 2):
        self.llm = llm
        self.max_retries = max_retries
        self.system_prompt = build_assessor_system_prompt()

    def assess(self, column: ColumnDescription) -> AssessmentResult:
        prompt = column.to_user_prompt()
        last_error = None
        last_response = ""

        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                time.sleep(3)
            response = self.llm.complete(
                system=self.system_prompt,
                user=prompt,
                # 1600 (а не 1200): теперь модель сначала кратко рассуждает,
                # затем выдаёт финальный JSON; запас гарантирует, что последний
                # JSON-блок не будет обрезан по лимиту токенов.
                max_tokens=1600,
            )
            last_response = response
            try:
                payload = _extract_json(response)
                vector = self._validate_vector(payload)
                reasoning = str(payload.get("reasoning", "")).strip()
                return AssessmentResult(
                    vector=vector,
                    reasoning=reasoning,
                    raw_response=response,
                )
            except (ValueError, KeyError, TypeError) as e:
                last_error = e
                prompt = (
                    column.to_user_prompt()
                    + f"\n\nПРЕДЫДУЩИЙ ОТВЕТ НЕ УДАЛОСЬ РАСПАРСИТЬ: {e}. "
                    "Верни СТРОГО валидный JSON с ключами K1..K6 (целые числа в допустимых "
                    "диапазонах) и reasoning. Никакой markdown-обёртки."
                )

        raise ValueError(
            f"Оценщик не вернул корректный JSON за {self.max_retries + 1} попыток. "
            f"Последняя ошибка: {last_error}. Ответ: {last_response[:300]!r}"
        )

    @staticmethod
    def _validate_vector(payload: dict) -> tuple:
        omega = tuple(c.omega for c in CRITERIA)
        keys = ("K1", "K2", "K3", "K4", "K5", "K6")
        vector = []
        for k, w in zip(keys, omega):
            if k not in payload:
                raise KeyError(f"В JSON отсутствует ключ {k}")
            v = payload[k]
            if not isinstance(v, int):
                try:
                    v = int(v)
                except (TypeError, ValueError):
                    raise ValueError(f"Ключ {k} не является целым числом: {v!r}")
            if not 1 <= v <= w:
                raise ValueError(f"Значение {k}={v} вне диапазона [1, {w}]")
            vector.append(v)
        return tuple(vector)
