"""
CLI для Фазы 1: построение решающего правила.

Примеры:
    python build_rule.py --provider anthropic
    python build_rule.py --provider openai --model gpt-4o
    python build_rule.py --provider deepseek --output rules/rule_ds_v4.json
    python build_rule.py --provider manual
"""

import argparse
import sys
from pathlib import Path

from dss import (
    make_client, build_decision_rule, total_space_size, CRITERIA, CLASSES,
)

# Папка для хранения решающих правил (создаётся автоматически при необходимости)
RULES_DIR = Path("rules")


def main():
    parser = argparse.ArgumentParser(description="Построение решающего правила (Фаза 1).")
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "deepseek", "manual", "human"],
        default="anthropic",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--output", default=None,
        help=(
            "Путь к выходному JSON-файлу. "
            "По умолчанию: rules/rule_<provider>.json"
        ),
    )
    parser.add_argument(
        "--results", default="results",
        help=(
            "Только для --provider human: где брать классификации LLM для подсказок. "
            "Директория (берутся results_*.csv / result_*.csv), glob-шаблон или "
            "список файлов через запятую. По умолчанию: results/"
        ),
    )
    parser.add_argument(
        "--threshold", type=int, default=2,
        help="Только для human: порог L1-расстояния для «близких» объектов (по умолчанию 2).",
    )
    parser.add_argument(
        "--max-matches", dest="max_matches", type=int, default=8,
        help="Только для human: максимум показываемых соседей на одну LLM (по умолчанию 8).",
    )
    parser.add_argument("--notes", default="")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    progress = (lambda m: None) if args.quiet else print

    # --- Ветка: человек-ЛПР с подсказками от LLM-классификаций ------------
    if args.provider == "human":
        from dss import (
            ReferenceBank, AssistedHumanDecisionMaker, resolve_result_paths,
        )

        paths = resolve_result_paths(args.results)
        bank = ReferenceBank(paths)

        out_path = Path(args.output) if args.output else RULES_DIR / "rule_human.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        print("Провайдер: human (ЛПР — человек, подсказки от LLM)")
        print(f"Критерии: {len(CRITERIA)}, классы: {len(CLASSES)}, |Y| = {total_space_size()}")
        if bank.total() == 0:
            print(f"⚠ В '{args.results}' не найдено ни одного result-файла — подсказок не будет.")
        else:
            print(f"Загружено подсказок: {bank.total()} объектов из {len(bank.labels)} LLM:")
            for label in bank.labels:
                print(f"    {label}: {len(bank.by_llm[label])}  ({bank.sources[label]})")
        print(f"Порог близости (L1): ≤ {args.threshold}; максимум соседей на LLM: {args.max_matches}")
        print(f"Будет сохранено в: {out_path.resolve()}")
        print()

        dm = AssistedHumanDecisionMaker(
            bank, threshold=args.threshold, max_per_llm=args.max_matches,
        )
        rule = build_decision_rule(
            decision_maker=dm, provider="human", model="human",
            progress=progress, notes=args.notes,
        )

        rule.save(out_path)
        print(f"\nРешающее правило сохранено: {out_path.resolve()}")

        dist = rule.class_distribution()
        print("\nРаспределение по классам:")
        for cls in sorted(dist.keys()):
            cls_obj = CLASSES[cls - 1]
            print(f"  Класс {cls} [{cls_obj.code}] {cls_obj.name}: {dist[cls]} векторов")

        violations = rule.verify_consistency()
        if violations:
            print(f"\n⚠ Найдено {len(violations)} нарушений непротиворечивости.")
            for x, y, cx, cy in violations[:5]:
                print(f"  {x}(класс {cx}) доминирует {y}(класс {cy})")
            sys.exit(1)
        else:
            print("\n✓ Условие непротиворечивости выполнено для всех пар векторов.")
        return

    # --- Ветка: LLM в роли ЛПР (исходное поведение) ----------------------
    client_kwargs = {}
    if args.model:
        client_kwargs["model"] = args.model
    llm = make_client(args.provider, **client_kwargs)

    # Определяем путь сохранения: --output или rules/rule_<provider>.json
    if args.output:
        out_path = Path(args.output)
    else:
        model_tag = getattr(llm, "model", "model").replace("/", "-").replace(":", "-")
        out_path = RULES_DIR / f"rule_{llm.name}_{model_tag}.json"

    # Создаём папку, если не существует
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Провайдер: {llm.name}, модель: {getattr(llm, 'model', 'n/a')}")
    print(f"Критерии: {len(CRITERIA)}, классы: {len(CLASSES)}, |Y| = {total_space_size()}")
    print(f"Будет сохранено в: {out_path.resolve()}")
    print()

    rule = build_decision_rule(llm, progress=progress, notes=args.notes)

    rule.save(out_path)
    print(f"\nРешающее правило сохранено: {out_path.resolve()}")

    dist = rule.class_distribution()
    print("\nРаспределение по классам:")
    for cls in sorted(dist.keys()):
        cls_obj = CLASSES[cls - 1]
        print(f"  Класс {cls} [{cls_obj.code}] {cls_obj.name}: {dist[cls]} векторов")

    violations = rule.verify_consistency()
    if violations:
        print(f"\n⚠ Найдено {len(violations)} нарушений непротиворечивости.")
        for x, y, cx, cy in violations[:5]:
            print(f"  {x}(класс {cx}) доминирует {y}(класс {cy})")
        sys.exit(1)
    else:
        print("\n✓ Условие непротиворечивости выполнено для всех пар векторов.")


if __name__ == "__main__":
    main()
