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
        choices=["anthropic", "openai", "deepseek", "manual"],
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
    parser.add_argument("--notes", default="")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

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

    progress = (lambda m: None) if args.quiet else print

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
