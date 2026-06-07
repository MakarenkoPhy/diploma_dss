"""
Ядро метода ЦИКЛ (Цепная Интерактивная Классификация).

Источник: Асанов А.А., Борисенков П.В., Ларичев О.И., Нарыжный Е.В., Ройзензон Г.В.
"Метод многокритериальной классификации ЦИКЛ и его применение для анализа
кредитного риска" (2001).

Метод строит ПОЛНУЮ непротиворечивую порядковую классификацию пространства
многокритериальных альтернатив, поэтапно опрашивая ЛПР (в гибридной СППР — LLM).

Соглашение о направлении доминирования
--------------------------------------
Шкалы критериев упорядочены так, что градация 1 — наибольшая выраженность
чувствительности, ω_q — наименьшая. Тогда:

    x доминирует y  ⟺  x_q ≥ y_q ∀q (и хотя бы одно строго)
                    ⟹  x менее чувствителен, чем y
                    ⟹  class(x) ≥ class(y)

где class(·) ∈ {1, ..., M} и большее значение класса = меньшая чувствительность.

Структура алгоритма
-------------------
Шаг 1.  Классификация крайних точек y_min = (1,...,1) и y_max = (ω_1,...,ω_n).
Шаг 2.  procedure_D(y_max, y_min) — рекурсивное разбиение цепи между
        крайними точками с поиском границ между классами. После каждого
        получения класса от ЛПР работает propagate (S) и _resolve_contradictions (R).
Шаг 2.5 _resolve_class_gaps — для всех пар (a, b) с a P b и
        class(a) − class(b) > 1 запускаем D, чтобы сузить области с
        ненулевым разрывом C^U − C^L на остальном пространстве.
Шаг 3.  Достраивание — для каждого вектора v ∈ Y:
            если v ∈ classified — берём class(v);
            если C^L(v) = C^U(v) — берём это значение (определено propagate);
            иначе — используем эвристику ⌊(C^L + C^U)/2⌋ БЕЗ обращения к ЛПР
            (область неопределённости границы между классами).
"""

from __future__ import annotations
import itertools
from dataclasses import dataclass, field
from typing import Callable, Protocol


# ---------------------------------------------------------------------------
# Абстрактный интерфейс ЛПР: вектор → класс
# ---------------------------------------------------------------------------

class DecisionMaker(Protocol):
    """Лицо, принимающее решения. В гибридной СППР это LLM."""

    def classify(self, vector: tuple) -> int:
        """Принимает вектор оценок (vector[i] ∈ [1, ω_i]), возвращает номер класса (1..M)."""
        ...


# ---------------------------------------------------------------------------
# Состояние алгоритма
# ---------------------------------------------------------------------------

@dataclass
class TsiklState:
    omega: tuple
    M: int
    # Границы классов: для y, для которых ничего ещё не известно,
    # C_lower(y) = 1, C_upper(y) = M.
    C_upper: dict = field(default_factory=dict)
    C_lower: dict = field(default_factory=dict)
    # Векторы, для которых ЛПР дал ответ
    classified: dict = field(default_factory=dict)
    # Журнал обращений: [(vector, class), ...]
    query_log: list = field(default_factory=list)
    # Счётчик зажатых ответов ЛПР (ответ вышел за [C^L, C^U])
    clamp_count: int = 0
    # Суммарное число обнаруженных противоречий (все итерации R)
    contradiction_count: int = 0
    # Суммарное число итераций процедуры R (каждый проход цикла)
    R_iterations: int = 0


# ---------------------------------------------------------------------------
# Алгоритм
# ---------------------------------------------------------------------------

class TsiklAlgorithm:

    def __init__(
        self,
        omega: tuple,
        M: int,
        decision_maker: DecisionMaker,
        progress_callback: Callable | None = None,
    ):
        self.omega = omega
        self.N = len(omega)
        self.M = M
        self.dm = decision_maker
        self.state = TsiklState(omega=omega, M=M)
        self._progress = progress_callback or (lambda msg: None)
        self._max_R_iters = 5  # защита от зацикливания procedure_R

    # --- Доминирование и расстояния -----------------------------------------

    @staticmethod
    def dominates(x: tuple, y: tuple) -> bool:
        """Строгое доминирование P: x ≥ y по всем, хотя бы по одному строго."""
        return all(a >= b for a, b in zip(x, y)) and any(a > b for a, b in zip(x, y))

    @staticmethod
    def weak_dominates(x: tuple, y: tuple) -> bool:
        """Слабое доминирование Q: x ≥ y по всем критериям."""
        return all(a >= b for a, b in zip(x, y))

    @staticmethod
    def rho(x: tuple, y: tuple) -> int:
        return sum(abs(a - b) for a, b in zip(x, y))

    @staticmethod
    def norm(y: tuple) -> int:
        return sum(y)

    # --- Множества Λ и L ----------------------------------------------------

    def lambda_set(self, x: tuple, y: tuple) -> list:
        """Λ(x, y) = {v ∈ Y | x Q v Q y}. Предполагается x Q y."""
        ranges = [range(yq, xq + 1) for xq, yq in zip(x, y)]
        return list(itertools.product(*ranges))

    def L_set(self, x: tuple, y: tuple) -> list:
        """L(x, y) = {v ∈ Λ(x, y) | ‖v‖ близко к (‖x‖+‖y‖)/2}."""
        target = (self.norm(x) + self.norm(y)) / 2
        if target == int(target):
            t = int(target)
            return [v for v in self.lambda_set(x, y) if self.norm(v) == t]
        lo, hi = int(target), int(target) + 1
        return [v for v in self.lambda_set(x, y) if self.norm(v) in (lo, hi)]

    # --- Управление границами C^L и C^U -------------------------------------

    def C_lower(self, y: tuple) -> int:
        return self.state.C_lower.get(y, 1)

    def C_upper(self, y: tuple) -> int:
        return self.state.C_upper.get(y, self.M)

    def _set_C_lower(self, y, value: int):
        if value > self.C_lower(y):
            self.state.C_lower[y] = value

    def _set_C_upper(self, y, value: int):
        if value < self.C_upper(y):
            self.state.C_upper[y] = value

    # --- Конусы доминирования -----------------------------------------------

    def _cone_below(self, x: tuple):
        return itertools.product(*(range(1, xq + 1) for xq in x))

    def _cone_above(self, x: tuple):
        return itertools.product(*(range(xq, wq + 1) for xq, wq in zip(x, self.omega)))

    # --- Процедура S: распространение по доминированию ----------------------

    def propagate(self, x: tuple):
        """
        После классификации x в класс cls:
          y ≤ x: C_upper(y) ← cls   (y более чувствителен ⇒ class ≤ cls)
          z ≥ x: C_lower(z) ← cls   (z менее чувствителен ⇒ class ≥ cls)
        """
        cls = self.state.classified[x]
        for y in self._cone_below(x):
            if y != x:
                self._set_C_upper(y, cls)
        for z in self._cone_above(x):
            if z != x:
                self._set_C_lower(z, cls)

    # --- Запрос к ЛПР -------------------------------------------------------

    def classify_vector(self, x: tuple) -> int:
        """
        Если вектор уже классифицирован — возвращаем сохранённое значение.
        Если C^L(x) == C^U(x) — класс однозначно определён доминированием.
        Иначе — обращаемся к ЛПР, фиксируем результат и распространяем.
        """
        if x in self.state.classified:
            return self.state.classified[x]

        cl, cu = self.C_lower(x), self.C_upper(x)
        if cl == cu:
            self.state.classified[x] = cl
            return cl

        raw_cls = self.dm.classify(x)
        # Защита: ответ ЛПР, выходящий за границы, ограничиваем ими
        cls = max(cl, min(cu, raw_cls))
        if raw_cls != cls:
            self.state.clamp_count += 1

        self.state.classified[x] = cls
        self.state.query_log.append((x, cls))
        self._progress(f"  [#{len(self.state.query_log):>3}] {x} → класс {cls}")

        self.propagate(x)
        return cls

    # --- Процедура D: рекурсивное разбиение интервала классов ---------------

    def procedure_D(self, a: tuple, b: tuple):
        """
        D(a, b): a Q b, классы a и b известны.
        Если cls_a > cls_b и расстояние > 1, выбирается опорный вектор x ∈ L(a, b),
        запрашивается у ЛПР, и рекурсивно вызывается D(a, x) и D(x, b)
        там, где остался разрыв классов.
        """
        cls_a = self.state.classified[a]
        cls_b = self.state.classified[b]

        if cls_a == cls_b:
            return
        if self.rho(a, b) <= 1:
            return
        # Оптимизация для гибридной СППР с дорогим ЛПР:
        # при разрыве классов = 1 граница уже локализована с точностью
        # до области, заведомо помещающейся в один класс с обеих сторон,
        # и достроится эвристикой ⌊(C^L+C^U)/2⌋ на Шаге 3.
        if cls_a - cls_b <= 1:
            return

        L = self.L_set(a, b)
        x = self._select_pivot(L)
        if x is None:
            return

        cls_x = self.classify_vector(x)
        self._resolve_contradictions()

        # Получаем актуальные классы (могли измениться при процедуре R)
        cls_a = self.state.classified.get(a, cls_a)
        cls_b = self.state.classified.get(b, cls_b)
        cls_x = self.state.classified.get(x, cls_x)

        # Разбиваем только там, где остался разрыв классов
        if cls_a > cls_x:
            self.procedure_D(a, x)
        if cls_x > cls_b:
            self.procedure_D(x, b)

    def _select_pivot(self, L: list):
        """
        Эвристика: среди неклассифицированных векторов из L выбираем тот,
        у которого максимален текущий разрыв C^U − C^L. Там больше всего
        неопределённости, и фиксация класса даст максимум информации
        для propagate.
        При равенстве разрывов — берём по серединной норме.
        """
        if not L:
            return None
        unclass = [v for v in L if v not in self.state.classified]
        if not unclass:
            return None
        return max(
            unclass,
            key=lambda v: (
                self.C_upper(v) - self.C_lower(v),
                -abs(self.norm(v) * 2 - sum(self.norm(u) for u in (L[0], L[-1])))
            ),
        )

    # --- Процедура R: устранение противоречий -------------------------------

    def _resolve_contradictions(self):
        """
        Ищем пары непосредственно классифицированных векторов x, y,
        для которых x P y, но class(x) < class(y) — противоречие условию (2)
        непротиворечивости. Простое разрешение: «срединное» значение
        между двумя классами. Защита от зацикливания: max N итераций.
        """
        for _ in range(self._max_R_iters):
            contradictions = []
            classified = list(self.state.classified.keys())
            for i, x in enumerate(classified):
                for y in classified[i + 1:]:
                    if self.dominates(x, y):
                        if self.state.classified[x] < self.state.classified[y]:
                            contradictions.append((x, y))
                    elif self.dominates(y, x):
                        if self.state.classified[y] < self.state.classified[x]:
                            contradictions.append((y, x))
            if not contradictions:
                return
            self.state.contradiction_count += len(contradictions)
            self.state.R_iterations += 1
            for x, y in contradictions:
                cx = self.state.classified[x]
                cy = self.state.classified[y]
                if cx < cy:
                    new_cls = (cx + cy + 1) // 2
                    self.state.classified[x] = new_cls
                    self.state.classified[y] = min(cy, new_cls)
                    self.propagate(x)
                    self.propagate(y)

    # --- Шаг 2.5: разрешение оставшихся разрывов классов через D ------------

    def _resolve_class_gaps(self):
        """
        После procedure_D(y_max, y_min) ещё могут оставаться пары
        классифицированных векторов (a, b) с a P b и cls_a − cls_b > 1
        и rho(a, b) > 1: D-процедура их не «увидела», поскольку они не
        лежат на одной обработанной цепи.

        Запускаем D для каждой такой пары, пока такие пары есть.
        Сортируем по убыванию разрыва классов: чем больше разрыв,
        тем больше потенциал для propagate после нахождения границы.
        """
        max_iters = 50
        for _ in range(max_iters):
            classified = list(self.state.classified.keys())
            gaps = []
            for i, x in enumerate(classified):
                for y in classified:
                    if x is y:
                        continue
                    if self.dominates(x, y):
                        cx = self.state.classified[x]
                        cy = self.state.classified[y]
                        if cx - cy > 1 and self.rho(x, y) > 1:
                            gaps.append((cx - cy, x, y))
            if not gaps:
                return
            gaps.sort(key=lambda t: -t[0])
            _, a, b = gaps[0]
            queries_before = len(self.state.query_log)
            self.procedure_D(a, b)
            # Если D не сделал ни одного нового запроса — выходим
            if len(self.state.query_log) == queries_before:
                return

    # --- Запуск алгоритма ---------------------------------------------------

    def run(self, *, use_heuristic_fillin: bool = True) -> dict:
        """
        Запускает алгоритм ЦИКЛ.

        Параметры
        ---------
        use_heuristic_fillin : bool, default True
            Если True (рекомендуется для гибридной СППР), то на Шаге 3
            оставшиеся неоднозначные векторы достраиваются эвристикой
            ⌊(C^L + C^U) / 2⌋ — без обращений к ЛПР.
            Если False, по каждому неоднозначному вектору запрашивается ЛПР
            (соответствует оригинальной формулировке Асанова и др.).
            Этот режим даёт максимальную точность ценой бо́льшего числа
            обращений и нужен прежде всего для тестирования корректности.
        """
        total = 1
        for w in self.omega:
            total *= w
        self._progress(f"ЦИКЛ: |Y| = {total}, N = {self.N}, M = {self.M}, ω = {self.omega}")

        self._progress("")
        self._progress("Шаг 1. Классификация крайних точек")
        y_min = tuple(1 for _ in range(self.N))
        y_max = tuple(self.omega)
        cls_min = self.classify_vector(y_min)
        cls_max = self.classify_vector(y_max)

        if cls_max != cls_min:
            self._progress("")
            self._progress(f"Шаг 2. Процедура D({y_max}, {y_min})")
            self.procedure_D(y_max, y_min)

            self._progress("")
            self._progress("Шаг 2.5. Разрешение оставшихся разрывов классов")
            self._resolve_class_gaps()

        self._progress("")
        self._progress("Шаг 3. Достраивание (без обращений к ЛПР)")

        result: dict = {}
        ambig_count = 0
        ambiguous: list = []
        for vec in itertools.product(*(range(1, w + 1) for w in self.omega)):
            if vec in self.state.classified:
                result[vec] = self.state.classified[vec]
                continue
            cl, cu = self.C_lower(vec), self.C_upper(vec)
            if cl == cu:
                result[vec] = cl
            elif use_heuristic_fillin:
                # Эвристика для области неопределённости границы классов.
                # Округление в сторону БОЛЬШЕЙ ЧУВСТВИТЕЛЬНОСТИ (меньший номер
                # класса) безопасно для задачи защиты данных: лучше
                # переоценить уровень доступа, чем оставить столбец
                # с заниженной защитой.
                result[vec] = (cl + cu) // 2
                ambig_count += 1
            else:
                ambiguous.append(vec)

        # В режиме без эвристики опрашиваем ЛПР по каждому неоднозначному вектору
        if not use_heuristic_fillin and ambiguous:
            self._progress(
                f"  Опрашиваем ЛПР по {len(ambiguous)} неоднозначным векторам "
                f"(C^L < C^U)…"
            )
            for vec in ambiguous:
                result[vec] = self.classify_vector(vec)

        queries = len(self.state.query_log)
        self._progress("")
        self._progress(
            f"Готово. Обращений к ЛПР: {queries} ({queries * 100 / total:.1f}% от |Y|). "
            f"Векторов через эвристику середины: {ambig_count}."
        )
        return result
