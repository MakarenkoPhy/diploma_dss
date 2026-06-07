"""
Проверка полноты решающего правила.
Запуск: py check_rule.py --rule rules/rule_deepseek_deepseek-chat.json

Скрипт проверяет:
1. Все ли 1296 векторов покрыты в правиле
2. Нет ли нарушений непротиворечивости
3. Распределение классов
"""
import argparse
import json
import math
from itertools import product
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule", required=True)
    args = parser.parse_args()

    data = json.loads(Path(args.rule).read_text(encoding="utf-8"))
    meta = data["metadata"]
    omega = meta["omega"]           # [4, 3, 3, 4, 3, 3]
    M = meta["num_classes"]         # 4
    expected = math.prod(omega)     # 1296

    table = {tuple(item["vector"]): item["class"] for item in data["table"]}

    print(f"Правило: {args.rule}")
    print(f"Провайдер/модель: {meta['provider']}/{meta['model']}")
    print(f"Построено: {meta['built_at']}, обращений: {meta['queries']}")
    print(f"omega = {omega}, M = {M}, |Y| = {expected}")
    print()

    # 1. Полнота
    all_vectors = list(product(*[range(1, w + 1) for w in omega]))
    missing = [v for v in all_vectors if v not in table]
    extra   = [v for v in table if v not in set(all_vectors)]

    print(f"Векторов в правиле: {len(table)}")
    print(f"Ожидается:          {expected}")
    if missing:
        print(f"⚠ ОТСУТСТВУЮТ {len(missing)} векторов!")
        for v in missing[:10]:
            print(f"  {v}")
        if len(missing) > 10:
            print(f"  ... и ещё {len(missing) - 10}")
    else:
        print("✓ Все векторы покрыты")

    if extra:
        print(f"⚠ Лишних векторов: {len(extra)}")
    else:
        print("✓ Посторонних векторов нет")

    # 2. Непротиворечивость (только нарушения)
    vectors = list(table.keys())
    violations = []
    for i, x in enumerate(vectors):
        for y in vectors[i + 1:]:
            geq = all(a >= b for a, b in zip(x, y))
            strict = any(a > b for a, b in zip(x, y))
            if geq and strict and table[x] < table[y]:
                violations.append((x, y, table[x], table[y]))
            geq2 = all(b >= a for a, b in zip(x, y))
            strict2 = any(b > a for a, b in zip(x, y))
            if geq2 and strict2 and table[y] < table[x]:
                violations.append((y, x, table[y], table[x]))

    print()
    if violations:
        print(f"⚠ Нарушений непротиворечивости: {len(violations)}")
        for x, y, cx, cy in violations[:5]:
            print(f"  {x}(C{cx}) доминирует {y}(C{cy})")
    else:
        print("✓ Условие непротиворечивости выполнено")

    # 3. Распределение
    from collections import Counter
    dist = Counter(table.values())
    print()
    print("Распределение по классам:")
    for cls in sorted(dist):
        n = dist[cls]
        print(f"  C{cls}: {n:4d} ({n / expected * 100:.1f}%)")


if __name__ == "__main__":
    main()
