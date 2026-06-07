"""
Тестовая задача для проверки реализации метода ЦИКЛ.

Задача: выбор места для отдыха.
  3 критерия × 3 градации × 3 класса = пространство |Y| = 27.

Критерии (1 = наименее привлекательно, 3 = наиболее):
  K₁ Погода:        1 = плохая,    2 = средняя,     3 = отличная
  K₂ Цена:          1 = дорого,    2 = средне,      3 = дёшево
  K₃ Интересность:  1 = скучно,    2 = нормально,   3 = интересно

Классы (1 = наименее желательно, 3 = наиболее):
  C₁ не ехать
  C₂ рассмотреть (подумать)
  C₃ ехать обязательно

Эталонное правило: class(v) = round(mean(v)) — среднее по трём оценкам,
округлённое к ближайшему целому. Это монотонное правило (соответствует
условию (2) метода ЦИКЛ), но в отличие от max() из теста кредитного риска,
граница между классами проходит «по диагонали» пространства, а не «по осям».
Это другой профиль работы алгоритма и более строгая проверка.

Сравниваются те же два режима реализации:
  A) с эвристикой ⌊(C^L+C^U)/2⌋ на Шаге 3 (наша оптимизация);
  B) без эвристики (классический ЦИКЛ Асанова и др.).
"""

from collections import Counter
import time

from dss.tsikl import TsiklAlgorithm, DecisionMaker


class AverageRatingExpert(DecisionMaker):
    """Эталонный эксперт: класс = округление среднего по трём оценкам.

    round((1+1+1)/3) = 1   → не ехать
    round((1+1+2)/3) = 1   → не ехать
    round((1+2+2)/3) = 2   → рассмотреть
    round((2+2+3)/3) = 2   → рассмотреть
    round((2+3+3)/3) = 3   → ехать
    """

    def __init__(self):
        self.calls = 0
        self.history = []

    def classify(self, vector):
        self.calls += 1
        cls = round(sum(vector) / len(vector))
        # round() в Python по банковским правилам, но для значений 1.33..3.0
        # это даёт ожидаемый ответ. На всякий случай зажимаем в [1, 3].
        cls = max(1, min(3, cls))
        self.history.append((vector, cls))
        return cls


def reference_class(vector):
    return max(1, min(3, round(sum(vector) / len(vector))))


OMEGA = (3, 3, 3)
M = 3
TOTAL = 27

CRIT_NAMES = ["Погода", "Цена", "Интер."]
CLASS_NAMES = {
    1: "C₁ не ехать",
    2: "C₂ рассмотреть",
    3: "C₃ ехать!",
}


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
        print(f"    {CLASS_NAMES[cls]:>17}: {dist[cls]:>2}")

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
    expert = AverageRatingExpert()
    algo = TsiklAlgorithm(omega=OMEGA, M=M, decision_maker=expert,
                          progress_callback=lambda m: None)
    t0 = time.time()
    result = algo.run(use_heuristic_fillin=use_heuristic)
    elapsed = time.time() - t0
    return result, expert, algo, elapsed


def main():
    print("=" * 72)
    print("ТЕСТ КОРРЕКТНОСТИ РЕАЛИЗАЦИИ ЦИКЛ")
    print("Задача: выбор места для отдыха")
    print("Эталонное правило: class(v) = round((K₁ + K₂ + K₃) / 3)")
    print("Пространство Y: 3 × 3 × 3 = 27 векторов, 3 класса")
    print("=" * 72)

    # Покажу для понимания эталонную таблицу
    print("\nЭталонная классификация (как должна выглядеть):")
    print(f"  {'Погода':>7} {'Цена':>7} {'Интер.':>7} | средн. | класс")
    print(f"  {'-'*7} {'-'*7} {'-'*7} + {'-'*5} + {'-'*5}")
    n_by_class = Counter()
    for v in sorted([(a, b, c) for a in (1, 2, 3) for b in (1, 2, 3) for c in (1, 2, 3)]):
        avg = sum(v) / 3
        cls = reference_class(v)
        n_by_class[cls] += 1
        if v in [(1,1,1), (1,1,2), (1,2,2), (2,2,2), (2,2,3), (2,3,3), (3,3,3)]:
            print(f"  {v[0]:>7} {v[1]:>7} {v[2]:>7} | {avg:>5.2f} | {cls}")
    print(f"  ...                          (всего {sum(n_by_class.values())} векторов)")
    print(f"\nОжидаемое распределение: " +
          ", ".join(f"{CLASS_NAMES[c]}: {n_by_class[c]}" for c in sorted(n_by_class)))

    # Прогон A: с эвристикой
    res_a, exp_a, algo_a, t_a = run_one(use_heuristic=True)
    mm_a, vv_a = analyse("Режим A — с эвристикой ⌊(C^L+C^U)/2⌋", res_a, exp_a, t_a)

    # Прогон B: классический
    res_b, exp_b, algo_b, t_b = run_one(use_heuristic=False)
    mm_b, vv_b = analyse("Режим B — без эвристики (как у Асанова и др.)", res_b, exp_b, t_b)

    # Полная таблица в режиме B
    print(f"\n{'─'*72}")
    print(f"Полная таблица решающего правила (режим B):")
    print(f"{'─'*72}")
    queried_b = {tuple(v) for v, _ in exp_b.history}
    print(f"  {'Погода':>6} {'Цена':>5} {'Интер':>5} | класс             | источник")
    print(f"  {'-'*6} {'-'*5} {'-'*5} + {'-'*17} + {'-'*30}")
    for vec in sorted(res_b.keys()):
        cls = res_b[vec]
        src = "опрос ЛПР" if vec in queried_b else "propagate (S)"
        marker = "✓" if cls == reference_class(vec) else "✗"
        print(f"  {vec[0]:>6} {vec[1]:>5} {vec[2]:>5} | {marker} {CLASS_NAMES[cls]:<15} | {src}")

    # Журнал опросов
    print(f"\n{'─'*72}")
    print(f"Журнал опросов ЛПР в режиме B ({exp_b.calls} запросов):")
    print(f"{'─'*72}")
    for i, (vec, cls) in enumerate(exp_b.history, 1):
        labelled = ", ".join(f"{n}={v}" for n, v in zip(CRIT_NAMES, vec))
        print(f"  #{i:>2}  ({labelled})  →  {CLASS_NAMES[cls]}")

    # Резюме
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
    print("Эталонное правило 'среднее по критериям' даёт более 'диагональные'")
    print("границы между классами, чем 'максимум' (предыдущий тест). Это другой")
    print("профиль работы — больше векторов попадает в средний класс.")
    print()
    if mm_b == 0 and vv_a == 0 and vv_b == 0:
        print("✓ Реализация ЦИКЛ работает КОРРЕКТНО на этой задаче:")
        print("  – пространство Y покрыто полностью в обоих режимах;")
        print("  – непротиворечивость выполнена;")
        print("  – режим B даёт результат, совпадающий с эталонным правилом;")
        print(f"  – задействовано {exp_b.calls} опросов из {TOTAL} возможных")
        print(f"    (экономия {(TOTAL-exp_b.calls)*100/TOTAL:.0f}%).")


if __name__ == "__main__":
    main()
