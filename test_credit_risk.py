"""
Тестовая задача для проверки корректности реализации метода ЦИКЛ.

Задача: анализ кредитного риска (отсылка к оригинальной статье
Асанова, Борисенкова, Ларичева, Нарыжного, Ройзензона, 2001).
  3 критерия × 3 градации × 3 класса = пространство |Y| = 27.
  Эталонное правило: class(v) = max(v) — «слабое звено» определяет класс.
  (Хоть один плохой показатель ⇒ заявка попадает в более рискованную категорию.)

Сравниваются два режима реализации ЦИКЛ:
  A) с эвристикой ⌊(C^L+C^U)/2⌋ на Шаге 3 — экономит запросы;
  B) без эвристики (опрос ЛПР по каждому неоднозначному вектору) —
     соответствует оригинальной формулировке Асанова и др.

Для обоих проверяются:
  – покрытие пространства Y,
  – непротиворечивость классификации (условие монотонности (2)),
  – соответствие эталонному правилу.
"""

from collections import Counter
import time

from dss.tsikl import TsiklAlgorithm, DecisionMaker


class WeakestLinkExpert(DecisionMaker):
    """Эталонный «эксперт»: класс заявки определяется худшим из её критериев."""

    def __init__(self):
        self.calls = 0
        self.history = []

    def classify(self, vector):
        self.calls += 1
        cls = max(vector)
        self.history.append((vector, cls))
        return cls


def reference_class(vector):
    return max(vector)


OMEGA = (3, 3, 3)
M = 3
TOTAL = 27

CRIT_NAMES = ["Залог", "Ист.", "Доход"]
CLASS_NAMES = {1: "C₁ отказать", 2: "C₂ доп.расс.", 3: "C₃ одобрить"}


def analyse(label, result, expert, elapsed):
    print(f"\n{'─'*72}")
    print(f"РЕЗУЛЬТАТЫ: {label}")
    print(f"{'─'*72}")

    print(f"  Время работы: {elapsed*1000:.1f} мс")
    print(f"  Обращений к ЛПР: {expert.calls} из {TOTAL} ({expert.calls*100/TOTAL:.1f}%)")
    print(f"  Сэкономлено опросов: {TOTAL - expert.calls} ({(TOTAL-expert.calls)*100/TOTAL:.1f}%)")

    dist = Counter(result.values())
    print(f"\n  Распределение классов:")
    for cls in sorted(dist):
        print(f"    {CLASS_NAMES[cls]:>16}: {dist[cls]:>2}")

    mismatches = [(v, result[v], reference_class(v))
                  for v in result if result[v] != reference_class(v)]
    if not mismatches:
        print(f"\n  ✓ Все 27 векторов совпадают с эталоном.")
    else:
        print(f"\n  ⚠ Расхождений с эталоном: {len(mismatches)}")
        for vec, cls, ref in mismatches:
            print(f"      {vec}: получен {cls}, эталон {ref}")

    items = list(result.items())
    violations = []
    for i, (x, cx) in enumerate(items):
        for y, cy in items[i + 1:]:
            x_dom_y = all(a >= b for a, b in zip(x, y)) and any(a > b for a, b in zip(x, y))
            y_dom_x = all(b >= a for a, b in zip(x, y)) and any(b > a for a, b in zip(x, y))
            if x_dom_y and cx < cy:
                violations.append((x, y, cx, cy))
            if y_dom_x and cy < cx:
                violations.append((y, x, cy, cx))
    if not violations:
        print(f"  ✓ Условие непротиворечивости выполнено.")
    else:
        print(f"  ⚠ Нарушений непротиворечивости: {len(violations)}")

    return len(mismatches), len(violations)


def run_one(use_heuristic):
    expert = WeakestLinkExpert()
    algo = TsiklAlgorithm(omega=OMEGA, M=M, decision_maker=expert,
                          progress_callback=lambda m: None)
    t0 = time.time()
    result = algo.run(use_heuristic_fillin=use_heuristic)
    elapsed = time.time() - t0
    return result, expert, algo, elapsed


def main():
    print("=" * 72)
    print("ТЕСТ КОРРЕКТНОСТИ РЕАЛИЗАЦИИ ЦИКЛ")
    print("Задача: анализ кредитного риска (Асанов и др., 2001)")
    print("Эталонное правило: class(v) = max(v₁, v₂, v₃)")
    print("Пространство Y: 3 × 3 × 3 = 27 векторов, 3 класса")
    print("=" * 72)

    res_a, exp_a, algo_a, t_a = run_one(use_heuristic=True)
    mm_a, vv_a = analyse("Режим A — с эвристикой ⌊(C^L+C^U)/2⌋", res_a, exp_a, t_a)

    res_b, exp_b, algo_b, t_b = run_one(use_heuristic=False)
    mm_b, vv_b = analyse("Режим B — без эвристики (как у Асанова и др.)", res_b, exp_b, t_b)

    print(f"\n{'─'*72}")
    print(f"Полная таблица решающего правила (режим B, эталонный):")
    print(f"{'─'*72}")
    queried_b = {tuple(v) for v, _ in exp_b.history}
    print(f"  {'K1':>3} {'K2':>3} {'K3':>3} | класс             | источник")
    print(f"  {'-'*3} {'-'*3} {'-'*3} + {'-'*17} + {'-'*30}")
    for vec in sorted(res_b.keys()):
        cls = res_b[vec]
        src = "опрос ЛПР" if vec in queried_b else "propagate (S)"
        marker = "✓" if cls == reference_class(vec) else "✗"
        print(f"  {vec[0]:>3} {vec[1]:>3} {vec[2]:>3} | {marker} {CLASS_NAMES[cls]:<15} | {src}")

    print(f"\n{'─'*72}")
    print(f"Журнал опросов ЛПР в режиме B ({exp_b.calls} запросов):")
    print(f"{'─'*72}")
    for i, (vec, cls) in enumerate(exp_b.history, 1):
        labelled = ", ".join(f"{n}={v}" for n, v in zip(CRIT_NAMES, vec))
        print(f"  #{i:>2}  ({labelled})  →  {CLASS_NAMES[cls]}")

    print(f"\n{'='*72}")
    print(f"СРАВНЕНИЕ РЕЖИМОВ")
    print(f"{'='*72}")
    print(f"  {'Метрика':<35} {'Режим A':>15} {'Режим B':>15}")
    print(f"  {'-'*35} {'-'*15} {'-'*15}")
    print(f"  {'Запросов к ЛПР':<35} {exp_a.calls:>15} {exp_b.calls:>15}")
    print(f"  {'Доля от полного перебора':<35} {exp_a.calls*100/TOTAL:>14.1f}% {exp_b.calls*100/TOTAL:>14.1f}%")
    print(f"  {'Расхождений с эталоном':<35} {mm_a:>15} {mm_b:>15}")
    print(f"  {'Нарушений непротиворечивости':<35} {vv_a:>15} {vv_b:>15}")
    print(f"  {'Время работы (мс)':<35} {t_a*1000:>15.1f} {t_b*1000:>15.1f}")

    print(f"\n{'='*72}")
    print(f"ИНТЕРПРЕТАЦИЯ")
    print(f"{'='*72}")
    print("Оба режима строят непротиворечивое решающее правило (это инвариант")
    print("метода ЦИКЛ, гарантируемый процедурами S и R).")
    print()
    print("Режим B соответствует оригинальной формулировке Асанова и др. (2001):")
    print("при необходимости опрашивает ЛПР по каждому вектору, для которого")
    print("границы C^L и C^U после распространения не сошлись. Даёт точное")
    print("совпадение с эталоном.")
    print()
    print("Режим A — наша оптимизация для гибридной СППР, где обращение к LLM")
    print("стоит дорого. На пограничной области (где C^L < C^U) используется")
    print("эвристическое округление к большей чувствительности. Это снижает")
    print("число запросов, но в редких случаях даёт класс, отличающийся от")
    print("ответа ЛПР на 1.")
    print()
    if mm_b == 0 and vv_a == 0 and vv_b == 0:
        print("✓ Реализация ЦИКЛ работает КОРРЕКТНО:")
        print("  – пространство Y покрыто полностью в обоих режимах;")
        print("  – непротиворечивость не нарушается;")
        print("  – режим B даёт результат, идентичный прямому применению эталона.")


if __name__ == "__main__":
    main()
