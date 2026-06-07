"""
CLI для Фазы 2: оперативная классификация столбцов.

Режимы:
  1) Один столбец интерактивно:
        py classify.py --rule rules/rule_deepseek_deepseek-chat.json --provider deepseek
  2) Один столбец по аргументам:
        py classify.py --rule rules/rule_deepseek_deepseek-chat.json --provider deepseek \
            --field "client_inn" \
            --table "crm_retail.client_phys" \
            --table-desc "Витрина данных физических лиц для CRM розничного блока" \
            --description "ИНН клиента — физического лица"
  3) Пакетный из CSV:
        py classify.py --rule rules/rule_deepseek_deepseek-chat.json --provider deepseek \
            --input columns.csv --output results/results_deepseek.csv

Формат CSV: field_name, table_name, table_description, description, sample_values
"""

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

from dss import (
    make_client, DecisionRule, LLMAssessor, ColumnDescription,
    classify_column, classify_batch, CRITERIA,
)

# Папки для результатов (создаются автоматически при необходимости)
RULES_DIR   = Path("rules")
RESULTS_DIR = Path("results")


def read_csv(path: Path) -> list:
    columns = []
    with path.open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            columns.append(ColumnDescription(
                description=row.get("description", "").strip(),
                table_name=(row.get("table_name") or "").strip() or None,
                table_description=(row.get("table_description") or "").strip() or None,
                field_name=(row.get("field_name") or "").strip() or None,
                sample_values=(row.get("sample_values") or "").strip() or None,
            ))
    return columns


def write_csv(path: Path, results) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = (
            ["field_name", "table_name", "table_description", "description"]
            + [c.code for c in CRITERIA]
            + ["class", "class_code", "class_name", "reasoning"]
        )
        writer.writerow(header)
        for r in results:
            writer.writerow([
                r.column.field_name or "",
                r.column.table_name or "",
                r.column.table_description or "",
                r.column.description,
                *r.assessment.vector,
                r.sensitivity_class,
                r.class_code,
                r.class_name,
                r.assessment.reasoning,
            ])


def write_stats(path: Path, stats: dict) -> None:
    """Сохраняет суммарную статистику прогона в JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def append_jsonl_log(path: Path, result) -> None:
    """Дописывает полный результат (включая raw_response) в JSONL-лог."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        **result.as_dict(),
        "raw_response": result.assessment.raw_response,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_result(result) -> None:
    print(f"\nСтолбец: {result.column.field_name or '(без имени)'}")
    if result.column.table_name:
        print(f"Таблица: {result.column.table_name}")
    if result.column.table_description:
        print(f"Описание таблицы: {result.column.table_description}")
    print(f"Описание поля: {result.column.description}")
    print(f"\nВектор оценок:")
    for c, v in zip(CRITERIA, result.assessment.vector):
        g = c.grades[v - 1]
        print(f"  {c.code} ({c.name}): {v} = {g.code} {g.title}")
    print(f"\nКЛАСС: {result.class_code} — {result.class_name}")
    if result.assessment.reasoning:
        print(f"\nОбоснование:\n  {result.assessment.reasoning}")


def _print_cache_stats(llm) -> None:
    """Если провайдер собирал статистику prompt caching — показать её."""
    stats = getattr(llm, "cache_stats", None)
    if not stats:
        return
    created = stats.get("created", 0)
    read = stats.get("read", 0)
    uncached = stats.get("uncached_input", 0)
    if created == 0 and read == 0:
        return
    print("\nСтатистика prompt caching:")
    print(f"  Создано кэша (1-й запрос):  {created:>8} токенов")
    print(f"  Прочитано из кэша:          {read:>8} токенов (дешевле ~в 10 раз)")
    print(f"  Некэшированный вход:        {uncached:>8} токенов")
    if read > 0:
        saved = read * 0.9
        print(f"  Эквивалент экономии:        ~{saved:>7.0f} входных токенов")


def _resolve_rule_path(arg: str) -> Path:
    """
    Ищет файл правила:
    1. Точный путь как указан.
    2. Если не найден — ищет в папке rules/.
    """
    p = Path(arg)
    if p.exists():
        return p
    in_rules = RULES_DIR / p.name
    if in_rules.exists():
        return in_rules
    # Вернём оригинальный путь — DecisionRule.load выдаст понятную ошибку
    return p


def main():
    parser = argparse.ArgumentParser(description="Оперативная классификация столбцов (Фаза 2).")
    parser.add_argument(
        "--rule", required=True,
        help="Путь к JSON-файлу с решающим правилом (или имя файла — ищется в rules/)",
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "deepseek", "manual"],
        default="anthropic",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--input", help="CSV со столбцами для пакетного режима")
    parser.add_argument(
        "--output", default=None,
        help=(
            "CSV с результатами. "
            "По умолчанию: results/results_<provider>_<input_stem>.csv "
            "(в пакетном режиме) или results/results_<provider>.csv (в одиночном)"
        ),
    )
    parser.add_argument("--field", help="Имя поля")
    parser.add_argument("--table", help="Имя таблицы")
    parser.add_argument("--table-desc", dest="table_desc", help="Описание таблицы")
    parser.add_argument("--description", help="Описание поля")
    parser.add_argument("--samples", help="Примеры значений")
    parser.add_argument("--json", action="store_true", help="Вывод в JSON")
    parser.add_argument(
        "--log", default=None,
        help=(
            "JSONL-файл для дозаписи полных результатов. "
            "По умолчанию: results/log_<provider>_<input_stem>.jsonl (пакетный режим)"
        ),
    )
    args = parser.parse_args()

    rule_path = _resolve_rule_path(args.rule)
    rule = DecisionRule.load(rule_path)
    print(f"Загружено правило: {rule_path}")
    print(f"  Провайдер/модель: {rule.metadata.provider}/{rule.metadata.model}, "
          f"построено {rule.metadata.built_at}, обращений: {rule.metadata.queries}")

    client_kwargs = {}
    if args.model:
        client_kwargs["model"] = args.model
    llm = make_client(args.provider, **client_kwargs)
    assessor = LLMAssessor(llm)

    # Пакетный режим
    if args.input:
        in_path = Path(args.input)

        # Определяем путь выходного CSV
        if args.output:
            out_path = Path(args.output)
        else:
            stem = in_path.stem
            out_path = RESULTS_DIR / f"results_{llm.name}_{stem}.csv"

        # Определяем путь лога
        if args.log:
            log_path = Path(args.log)
        else:
            stem = in_path.stem
            log_path = RESULTS_DIR / f"log_{llm.name}_{stem}.jsonl"

        # Создаём папки
        out_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        columns = read_csv(in_path)
        print(f"Прочитано {len(columns)} столбцов из {in_path}")
        print(f"Результаты → {out_path.resolve()}")
        print(f"Лог        → {log_path.resolve()}")
        print()

        t0 = time.time()
        results = classify_batch(columns, rule, assessor, progress=print)
        classify_time_sec = round(time.time() - t0, 1)

        write_csv(out_path, results)
        print(f"\nРезультаты сохранены: {out_path.resolve()}")
        for r in results:
            append_jsonl_log(log_path, r)
        print(f"Полный лог дописан в: {log_path.resolve()}")
        _print_cache_stats(llm)

        # Суммарная статистика прогона
        from collections import Counter
        class_dist = Counter(r.sensitivity_class for r in results)
        stats = {
            "run_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "provider": llm.name,
            "model": getattr(llm, "model", "n/a"),
            "rule_provider": rule.metadata.provider,
            "rule_model": rule.metadata.model,
            "rule_built_at": rule.metadata.built_at,
            "input_file": str(in_path),
            "total_columns": len(columns),
            "classified": len(results),
            "classify_time_sec": classify_time_sec,
            "avg_sec_per_column": round(classify_time_sec / len(results), 2) if results else 0,
            "class_distribution": {f"C{k}": v for k, v in sorted(class_dist.items())},
            "cache_stats": getattr(llm, "cache_stats", {}),
        }
        stats_path = out_path.with_name(out_path.stem.replace("results_", "stats_") + ".json")
        write_stats(stats_path, stats)
        print(f"Статистика прогона  → {stats_path.resolve()}")
        print(f"Время классификации: {classify_time_sec:.1f} сек "
              f"({stats['avg_sec_per_column']:.2f} сек/столбец)")
        return

    # Одиночный режим — из аргументов
    if args.description:
        column = ColumnDescription(
            description=args.description,
            table_name=args.table,
            table_description=args.table_desc,
            field_name=args.field,
            sample_values=args.samples,
        )
        result = classify_column(column, rule, assessor)
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
        else:
            print_result(result)

        # Лог в одиночном режиме — только если явно указан
        if args.log:
            log_path = Path(args.log)
            append_jsonl_log(log_path, result)
            print(f"\nЗапись дописана в: {log_path.resolve()}")
        return

    # Интерактивный режим
    log_path = Path(args.log) if args.log else None
    print("\nИнтерактивная классификация. Ctrl+C для выхода.")
    while True:
        try:
            print("\n" + "─" * 60)
            field_name = input("Имя поля (опционально): ").strip() or None
            table_name = input("Имя таблицы (опционально): ").strip() or None
            table_desc = input("Описание таблицы (опционально): ").strip() or None
            description = input("Описание поля: ").strip()
            if not description:
                continue
            column = ColumnDescription(
                description=description,
                table_name=table_name,
                table_description=table_desc,
                field_name=field_name,
            )
            result = classify_column(column, rule, assessor)
            print_result(result)
            if log_path:
                append_jsonl_log(log_path, result)
        except KeyboardInterrupt:
            print("\nВыход.")
            return


if __name__ == "__main__":
    main()
