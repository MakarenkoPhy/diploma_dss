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

Сначала — МАКСИМАЛЬНО СЖАТОЕ рассуждение (3–5 коротких строк, телеграфным \
стилем). Структура:
- одна строка: тип субъекта + роль столбца в реляционной структуре;
- шесть строк по критериям в формате `K1=N — обоснование ≤6 слов`.

ЗАПРЕЩЕНО: повторять атрибуты столбца, цитировать описание поля или таблицы, \
раскрывать определения критериев, использовать markdown-форматирование (жирный, \
заголовки, маркированные списки), писать вводные фразы («Анализирую…», \
«Рассмотрим…»). Допускается только тире-разделитель внутри строки критерия.

ЗАТЕМ заверши ответ ЕДИНСТВЕННЫМ валидным JSON-блоком в формате ниже. \
Парсится ТОЛЬКО последний JSON-блок в ответе. JSON обязан быть синтаксически \
корректным и полным.

ПРИМЕР ИТОГОВОГО ФОРМАТА (соблюдай эту лаконичность):

```
Субъект: ФЛ-взрослый (маркер `retail`). Роль: внешний идентификатор.
K1=2 — ИНН, ПДн (152-ФЗ)
K2=2 — массовая утечка, штрафы
K3=2 — соц. инженерия, не для прямого хищения
K4=3 — дискомфорт раскрытия
K5=3 — не влияет на операции
K6=1 — внешний идентификатор

{{"K1": 2, "K2": 2, "K3": 2, "K4": 3, "K5": 3, "K6": 1}}
```

ФОРМАТ ИТОГОВОГО JSON (строго валидный JSON, без markdown-обёртки):

{{
  "K1": <число от 1 до {w1}>,
  "K2": <число от 1 до {w2}>,
  "K3": <число от 1 до {w3}>,
  "K4": <число от 1 до {w4}>,
  "K5": <число от 1 до {w5}>,
  "K6": <число от 1 до {w6}>
}}

JSON содержит ТОЛЬКО шесть числовых полей K1..K6. Поле с обоснованием в JSON \
не выводится — обоснование уже изложено в свободном рассуждении выше и \
сохраняется системой как есть.

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


def _extract_json_and_preamble(text: str) -> tuple[dict, str]:
    """
    Возвращает (распарсенный JSON, preamble — текст ДО финального JSON-блока).
    Preamble используется как reasoning: модель сначала свободно рассуждает,
    затем выдаёт JSON только с числовыми полями K1..K6.
    """
    original = text
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # Сначала пробуем весь текст целиком (идеальный случай — JSON без preamble)
    try:
        return json.loads(text), ""
    except json.JSONDecodeError:
        pass
    # Берём ПОСЛЕДНИЙ валидный JSON-блок. Всё, что до него — preamble (reasoning).
    last_valid = None
    last_span = None
    for match in _JSON_BLOCK_RE.finditer(text):
        try:
            parsed = json.loads(match.group(0))
            last_valid = parsed
            last_span = match.span()
        except json.JSONDecodeError:
            continue
    if last_valid is not None:
        preamble = text[: last_span[0]].strip() if last_span else ""
        # Чистим возможные markdown-обёртки и хвостовые ``` в начале JSON-блока
        preamble = re.sub(r"```(?:json)?\s*$", "", preamble).strip()
        return last_valid, preamble
    raise ValueError(f"Не удалось извлечь JSON из ответа: {original[:200]!r}")


def _extract_json(text: str) -> dict:
    """Обратная совместимость: возвращает только JSON."""
    payload, _ = _extract_json_and_preamble(text)
    return payload


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
                # 600: телеграфный preamble (~3-5 строк, target ~150 токенов)
                # + компактный JSON только с шестью числовыми полями (~30 токенов).
                # Запас x3 от целевого размера — на случай если модель чуть
                # расширит обоснование по сложным объектам.
                max_tokens=600,
            )
            last_response = response
            try:
                payload, preamble = _extract_json_and_preamble(response)
                vector = self._validate_vector(payload)
                # reasoning теперь берётся из preamble (текст до JSON-блока),
                # а не из поля внутри JSON. Это устраняет дублирование текста
                # и существенно сокращает выходные токены.
                reasoning = preamble
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
                    "диапазонах). JSON содержит только эти шесть полей, никакого reasoning внутри JSON, "
                    "никакой markdown-обёртки."
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
