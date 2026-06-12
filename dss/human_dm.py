"""
Фаза 1, вариант с ЧЕЛОВЕКОМ-ЛПР и подсказками от LLM-классификаций.

`AssistedHumanDecisionMaker` реализует интерфейс `DecisionMaker`
(`classify(vector) -> int`), но решение принимает человек. Чтобы облегчить
задачу, рядом с гипотетическим вектором, который предъявляет метод ЦИКЛ,
выводятся реальные объекты из результатов Фазы 2 разных LLM
(файлы `results_<llm>_*.csv`), чьи векторы оценок максимально близки к
предъявляемому (расстояние L1 = сумма модулей разниц координат ≤ threshold).
Для каждого такого объекта показываются все его атрибуты, присвоенный
конкретной LLM класс и обоснование. Если в пределах порога объектов для
какой-то LLM нет — выводится один ближайший (как fallback) с пометкой.

Источники подсказок — это выходы СИСТЕМ конкретных LLM (правило этой LLM,
применённое к вектору её же оценщика на реальном столбце), то есть это
референсные «якоря», а не эталон. Человек принимает решение самостоятельно.
"""

from __future__ import annotations
import csv
import glob
from dataclasses import dataclass, field
from pathlib import Path

from .criteria import CRITERIA, CLASSES
from .tsikl import DecisionMaker

CRIT_CODES = [c.code for c in CRITERIA]   # ['К₁', ..., 'К₆']
M = len(CLASSES)


# ---------------------------------------------------------------------------
# Загрузка референсных объектов из result-файлов
# ---------------------------------------------------------------------------

@dataclass
class RefObject:
    vector: tuple          # (К₁..К₆)
    row: dict              # все исходные столбцы строки CSV
    llm: str               # метка LLM (из имени файла)


def _l1(x: tuple, y: tuple) -> int:
    """Сумма модулей разниц координат (то же, что rho в tsikl.py)."""
    return sum(abs(a - b) for a, b in zip(x, y))


def _derive_label(path: Path) -> str:
    """results_anthropic_test_columns_440_clean.csv -> anthropic."""
    stem = path.stem
    for pref in ("results_", "result_"):
        if stem.startswith(pref):
            stem = stem[len(pref):]
            break
    # отрезаем хвост с именем датасета, если он начинается со знакомого маркера
    for marker in ("_test", "_440", "_columns", "_sample"):
        idx = stem.find(marker)
        if idx > 0:
            stem = stem[:idx]
            break
    return stem or path.stem


def resolve_result_paths(spec: str) -> list[Path]:
    """
    spec может быть:
      - директорией  -> берём results_*.csv и result_*.csv внутри;
      - glob-шаблоном -> разворачиваем;
      - списком файлов через запятую.
    """
    parts = [s.strip() for s in spec.split(",") if s.strip()]
    paths: list[Path] = []
    for part in parts:
        p = Path(part)
        if p.is_dir():
            found = sorted(p.glob("results_*.csv")) + sorted(p.glob("result_*.csv"))
            paths.extend(found)
        elif any(ch in part for ch in "*?[]"):
            paths.extend(sorted(Path(x) for x in glob.glob(part)))
        else:
            paths.append(p)
    # дедупликация с сохранением порядка
    seen, uniq = set(), []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


class ReferenceBank:
    """Хранилище референсных объектов, сгруппированных по LLM."""

    def __init__(self, paths):
        self.by_llm: dict[str, list[RefObject]] = {}
        self.sources: dict[str, str] = {}
        for p in paths:
            p = Path(p)
            if not p.exists():
                continue
            label = _derive_label(p)
            objs = self._load(p, label)
            if objs:
                self.by_llm.setdefault(label, []).extend(objs)
                self.sources[label] = str(p)

    @staticmethod
    def _load(path: Path, label: str) -> list[RefObject]:
        out: list[RefObject] = []
        with path.open(encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    vec = tuple(int(float(row[code])) for code in CRIT_CODES)
                except (KeyError, ValueError, TypeError):
                    continue
                out.append(RefObject(vector=vec, row=row, llm=label))
        return out

    @property
    def labels(self) -> list[str]:
        return list(self.by_llm.keys())

    def total(self) -> int:
        return sum(len(v) for v in self.by_llm.values())

    def nearest(self, vector: tuple, threshold: int = 2, max_per_llm: int = 8) -> dict:
        """
        Для каждой LLM возвращает:
          {label: {"matches": [(dist, RefObject), ...],
                   "total_within": int,   # сколько всего в пределах порога
                   "fallback": bool}}     # True, если порог пуст и показан 1 ближайший
        matches отсортированы по возрастанию расстояния.
        """
        result: dict[str, dict] = {}
        for label, objs in self.by_llm.items():
            scored = sorted(((_l1(vector, o.vector), o) for o in objs), key=lambda t: t[0])
            within = [(d, o) for d, o in scored if d <= threshold]
            if within:
                result[label] = {
                    "matches": within[:max_per_llm],
                    "total_within": len(within),
                    "fallback": False,
                }
            elif scored:
                result[label] = {
                    "matches": scored[:1],
                    "total_within": 0,
                    "fallback": True,
                }
            else:
                result[label] = {"matches": [], "total_within": 0, "fallback": False}
        return result


# ---------------------------------------------------------------------------
# Человек-ЛПР с подсказками
# ---------------------------------------------------------------------------

class AssistedHumanDecisionMaker(DecisionMaker):
    """Человек принимает решение, видя предъявляемый вектор и ближайшие
    реальные объекты из result-файлов каждой LLM."""

    def __init__(
        self,
        bank: ReferenceBank,
        *,
        threshold: int = 2,
        max_per_llm: int = 8,
        descr_len: int = 120,
        reasoning_len: int = 240,
        input_fn=input,
        output_fn=print,
    ):
        self.bank = bank
        self.threshold = threshold
        self.max_per_llm = max_per_llm
        self.descr_len = descr_len
        self.reasoning_len = reasoning_len
        self._in = input_fn
        self._out = output_fn
        # атрибуты для совместимости с pipeline.build_decision_rule
        self.history: list = []
        self.parse_retry_count: int = 0

    # --- вывод ----------------------------------------------------------

    def _show_object(self, vector: tuple) -> None:
        self._out("\n" + "═" * 78)
        self._out(f"ОБЪЕКТ НА ОЦЕНКУ — вектор {vector}")
        self._out("  (градация 1 = МАКСИМАЛЬНАЯ чувствительность по критерию)")
        for c, v in zip(CRITERIA, vector):
            g = c.grades[v - 1]
            self._out(f"  {c.code} {c.name}: {v} = [{g.code}] {g.title}")

    def _trim(self, s, n) -> str:
        s = (s or "").strip().replace("\n", " ")
        return s if len(s) <= n else s[: n - 1] + "…"

    def _show_refs(self, vector: tuple) -> None:
        if self.bank.total() == 0:
            self._out("\n  ⚠ Не загружено ни одного result-файла LLM — подсказок нет.")
            return
        near = self.bank.nearest(vector, self.threshold, self.max_per_llm)
        self._out("\n" + "─" * 78)
        self._out(f"БЛИЖАЙШИЕ КЛАССИФИЦИРОВАННЫЕ ОБЪЕКТЫ (ρ = сумма |разниц|, порог ≤ {self.threshold})")
        for label in sorted(near.keys()):
            info = near[label]
            matches = info["matches"]
            if not matches:
                self._out(f"\n  [{label}] — нет объектов.")
                continue
            if info["fallback"]:
                tag = "в пределах порога нет; показан 1 ближайший"
            else:
                extra = info["total_within"] - len(matches)
                tag = f"в пределах порога: {info['total_within']}" + (f" (показаны первые {len(matches)})" if extra > 0 else "")
            self._out(f"\n  [{label}] {tag}")
            for d, o in matches:
                r = o.row
                ccode = r.get("class_code", "") or f"C{r.get('class','?')}"
                cname = r.get("class_name", "") or ""
                vec_str = ",".join(str(r.get(code, "?")) for code in CRIT_CODES)
                self._out(
                    f"    ρ={d}  {self._trim(r.get('field_name'), 40)}"
                    f" | {self._trim(r.get('table_name'), 40)}"
                )
                descr = self._trim(r.get("description"), self.descr_len)
                if descr:
                    self._out(f"          описание: {descr}")
                self._out(f"          вектор=({vec_str})  →  КЛАСС {ccode} {cname}".rstrip())
                reasoning = self._trim(r.get("reasoning"), self.reasoning_len)
                if reasoning:
                    self._out(f"          обоснование LLM: {reasoning}")

    # --- интерфейс DecisionMaker ---------------------------------------

    def classify(self, vector: tuple) -> int:
        self._show_object(vector)
        self._show_refs(vector)
        while True:
            ans = self._in(f"\n  Ваш класс для вектора {vector} (1..{M}): ").strip()
            try:
                v = int(ans)
            except ValueError:
                self.parse_retry_count += 1
                self._out(f"  Введите целое число от 1 до {M}.")
                continue
            if 1 <= v <= M:
                self.history.append({"vector": list(vector), "class": v})
                return v
            self.parse_retry_count += 1
            self._out(f"  Число должно быть в диапазоне 1..{M}.")
