"""
Streamlit-фронтенд гибридной СППР.

Запуск:
    streamlit run app.py
"""

import io
import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from dss import (
    CRITERIA, CLASSES, omega, total_space_size, num_classes,
    make_client, build_decision_rule,
    DecisionRule, LLMAssessor, ColumnDescription, classify_column,
)


st.set_page_config(
    page_title="Гибридная СППР: ЦИКЛ + LLM",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _init_state():
    if "rule" not in st.session_state:
        st.session_state.rule = None


_init_state()


# ---------------------------------------------------------------------------
# Сайдбар
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙ Конфигурация")

    provider = st.selectbox(
        "LLM-провайдер",
        ["anthropic", "openai", "deepseek", "manual"],
        index=0,
    )

    default_models = {
        "anthropic": "claude-sonnet-4-5",
        "openai": "gpt-4o",
        "deepseek": "deepseek-chat",
        "manual": "manual",
    }
    model = st.text_input(
        "Модель",
        value=default_models[provider],
        disabled=(provider == "manual"),
    )

    st.divider()
    st.subheader("📂 Решающее правило")
    upload = st.file_uploader("Загрузить из JSON", type="json")
    if upload is not None:
        try:
            data = json.loads(upload.read().decode("utf-8"))
            tmp_path = Path("/tmp/uploaded_rule.json")
            tmp_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            st.session_state.rule = DecisionRule.load(tmp_path)
            st.success(f"Загружено: {len(st.session_state.rule.table)} векторов")
        except Exception as e:
            st.error(f"Ошибка загрузки: {e}")

    if st.session_state.rule:
        meta = st.session_state.rule.metadata
        st.caption(
            f"Активное правило:\n\n"
            f"{meta.provider}/{meta.model}\n\n"
            f"Построено: {meta.built_at}\n\n"
            f"Обращений к ЛПР: {meta.queries}"
        )

    st.divider()
    st.caption(
        f"|Y| = {total_space_size()} = " + " × ".join(str(w) for w in omega())
        + f" \nКлассов: {num_classes()}"
    )


tab_build, tab_classify, tab_rule, tab_help = st.tabs([
    "🛠 Фаза 1. Построение правила",
    "🔍 Фаза 2. Классификация",
    "📊 Решающее правило",
    "📖 Критерии",
])


# =========================================================================
# Вкладка 1. Построение правила
# =========================================================================

with tab_build:
    st.header("Построение решающего правила методом ЦИКЛ")

    st.markdown(
        f"""
**Что происходит на этой фазе.** Метод ЦИКЛ предъявляет LLM-ЛПР гипотетические объекты
в виде векторов вербальных оценок (K₁..K₆) и получает от неё номер класса. Алгоритм выбирает
векторы стратегически: за счёт распространения по доминированию (процедура S) и
рекурсивного разбиения цепей (процедура D) полная классификация {total_space_size()}
векторов строится за **значительно меньшее число обращений**.

**Замечание об оптимизации.** В реализации введено раннее прекращение `procedure_D` при
разрыве классов = 1 (граница уже локализована); оставшиеся неоднозначные векторы
получают класс по эвристике ⌊(C^L+C^U)/2⌋ — округление в сторону большей чувствительности
(безопасное для задачи защиты данных).
        """
    )

    notes = st.text_input("Примечание для metadata (опционально)", value="")

    if provider == "manual":
        st.warning("Manual-режим не работает в Streamlit (нужен stdin). "
                   "Используйте `python build_rule.py --provider manual` в терминале.")

    if st.button("🚀 Запустить построение", type="primary", disabled=(provider == "manual")):
        log_box = st.empty()
        log_lines: list = []

        def progress(msg: str):
            log_lines.append(msg)
            log_box.code("\n".join(log_lines[-50:]), language="text")

        try:
            llm = make_client(provider, model=model)
            with st.spinner("Идёт построение решающего правила…"):
                t0 = time.time()
                rule = build_decision_rule(llm, progress=progress, notes=notes)
                elapsed = time.time() - t0

            st.session_state.rule = rule
            st.success(
                f"Готово за {elapsed:.1f} с. "
                f"Обращений к LLM: {rule.metadata.queries}, |Y| = {len(rule.table)}"
            )

            payload = io.BytesIO()
            payload.write(json.dumps({
                "metadata": rule.metadata.__dict__,
                "table": [{"vector": list(v), "class": c}
                          for v, c in sorted(rule.table.items())],
                "history": rule.history,
            }, ensure_ascii=False, indent=2).encode("utf-8"))
            payload.seek(0)
            st.download_button(
                "💾 Скачать решающее правило (JSON)",
                data=payload,
                file_name=f"rule_{provider}_{model}.json",
                mime="application/json",
            )

            dist = rule.class_distribution()
            df_dist = pd.DataFrame([
                {"Класс": cls, "Код": CLASSES[cls - 1].code,
                 "Название": CLASSES[cls - 1].name, "Векторов": dist.get(cls, 0)}
                for cls in range(1, num_classes() + 1)
            ])
            st.subheader("Распределение векторов по классам")
            st.dataframe(df_dist, use_container_width=True, hide_index=True)
            st.bar_chart(df_dist.set_index("Код")["Векторов"])

        except Exception as e:
            st.error(f"Ошибка: {e}")


# =========================================================================
# Вкладка 2. Классификация
# =========================================================================

with tab_classify:
    st.header("Оперативная классификация")

    if st.session_state.rule is None:
        st.info("Сначала постройте или загрузите решающее правило.")
    else:
        rule = st.session_state.rule

        sub_single, sub_batch = st.tabs(["Один столбец", "Пакет (CSV)"])

        with sub_single:
            with st.form("single_form"):
                c1, c2 = st.columns(2)
                with c1:
                    field_name = st.text_input("Имя поля", value="client_inn")
                    table_name = st.text_input("Имя таблицы", value="clients")
                with c2:
                    samples = st.text_input("Примеры значений (опционально)", value="")
                description = st.text_area(
                    "Описание столбца",
                    value="ИНН клиента — физического лица",
                    height=80,
                )
                submit = st.form_submit_button("🔍 Классифицировать", type="primary")

            if submit:
                if not description.strip():
                    st.warning("Введите описание столбца.")
                else:
                    try:
                        llm = make_client(provider, model=model)
                        assessor = LLMAssessor(llm)
                        column = ColumnDescription(
                            description=description.strip(),
                            field_name=field_name.strip() or None,
                            table_name=table_name.strip() or None,
                            sample_values=samples.strip() or None,
                        )
                        with st.spinner("LLM оценивает столбец…"):
                            result = classify_column(column, rule, assessor)

                        st.success(f"**{result.class_code} — {result.class_name}**")

                        col_v, col_r = st.columns([2, 3])
                        with col_v:
                            st.subheader("Вектор оценок")
                            df = pd.DataFrame([
                                {
                                    "Критерий": f"{c.code} {c.name}",
                                    "Оценка": v,
                                    "Градация": f"{c.grades[v-1].code} — {c.grades[v-1].title}",
                                }
                                for c, v in zip(CRITERIA, result.assessment.vector)
                            ])
                            st.dataframe(df, use_container_width=True, hide_index=True)
                        with col_r:
                            st.subheader("Обоснование LLM")
                            st.write(result.assessment.reasoning or "—")

                        with st.expander("Детали (raw response от LLM)"):
                            st.code(result.assessment.raw_response, language="json")
                    except Exception as e:
                        st.error(f"Ошибка: {e}")

        with sub_batch:
            st.markdown(
                "Формат CSV: `field_name, table_name, table_description, description, sample_values`. "
                "Обязательная колонка — `description`."
            )
            csv_file = st.file_uploader("CSV-файл", type=["csv"], key="batch_csv")
            if csv_file is not None:
                df_in = pd.read_csv(csv_file).fillna("")
                st.dataframe(df_in.head(20), use_container_width=True)

                if st.button("Классифицировать пакет", type="primary"):
                    try:
                        llm = make_client(provider, model=model)
                        assessor = LLMAssessor(llm)
                        columns = [
                            ColumnDescription(
                                description=str(row.get("description", "")).strip(),
                                table_name=str(row.get("table_name", "")).strip() or None,
                                table_description=str(row.get("table_description", "")).strip() or None,
                                field_name=str(row.get("field_name", "")).strip() or None,
                                sample_values=str(row.get("sample_values", "")).strip() or None,
                            )
                            for _, row in df_in.iterrows()
                            if str(row.get("description", "")).strip()
                        ]

                        progress_bar = st.progress(0.0)
                        log_box = st.empty()
                        results = []
                        for i, col in enumerate(columns):
                            log_box.write(
                                f"[{i+1}/{len(columns)}] "
                                f"{col.field_name or col.description[:60]}"
                            )
                            results.append(classify_column(col, rule, assessor))
                            progress_bar.progress((i + 1) / len(columns))

                        out_rows = []
                        for r in results:
                            row = {
                                "field_name": r.column.field_name or "",
                                "table_name": r.column.table_name or "",
                                "table_description": r.column.table_description or "",
                                "description": r.column.description,
                            }
                            for c, v in zip(CRITERIA, r.assessment.vector):
                                row[c.code] = v
                            row["class"] = r.sensitivity_class
                            row["class_code"] = r.class_code
                            row["class_name"] = r.class_name
                            row["reasoning"] = r.assessment.reasoning
                            out_rows.append(row)
                        df_out = pd.DataFrame(out_rows)
                        st.dataframe(df_out, use_container_width=True)

                        csv_buf = io.StringIO()
                        df_out.to_csv(csv_buf, index=False)
                        st.download_button(
                            "💾 Скачать результаты (CSV)",
                            data=csv_buf.getvalue().encode("utf-8"),
                            file_name="classification_results.csv",
                            mime="text/csv",
                        )

                        st.subheader("Распределение по классам")
                        st.bar_chart(df_out.groupby("class_code").size())
                    except Exception as e:
                        st.error(f"Ошибка: {e}")


# =========================================================================
# Вкладка 3. Решающее правило
# =========================================================================

with tab_rule:
    st.header("Текущее решающее правило")

    if st.session_state.rule is None:
        st.info("Правило не загружено.")
    else:
        rule = st.session_state.rule
        meta = rule.metadata

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Векторов в Y", len(rule.table))
        c2.metric("Обращений к ЛПР", meta.queries)
        c3.metric("Эффективность", f"{meta.queries*100/max(len(rule.table),1):.1f}%")
        c4.metric("Классов", meta.num_classes)

        st.caption(f"Провайдер: {meta.provider} / Модель: {meta.model} / "
                   f"Построено: {meta.built_at}")

        st.subheader("Распределение по классам")
        dist = rule.class_distribution()
        df = pd.DataFrame([
            {"Класс": f"{CLASSES[cls-1].code} {CLASSES[cls-1].name}",
             "Векторов": dist.get(cls, 0)}
            for cls in range(1, num_classes() + 1)
        ])
        st.bar_chart(df.set_index("Класс")["Векторов"])

        st.subheader("Полная таблица")
        rows = []
        for vec, cls in sorted(rule.table.items()):
            row = {f"K{i+1}": v for i, v in enumerate(vec)}
            row["class"] = cls
            row["class_code"] = CLASSES[cls - 1].code
            rows.append(row)
        df_full = pd.DataFrame(rows)
        st.dataframe(df_full, use_container_width=True, height=400)

        st.subheader("Журнал обращений к ЛПР")
        if rule.history:
            df_hist = pd.DataFrame([
                {**{f"K{i+1}": v for i, v in enumerate(h["vector"])},
                 "class": h["class"]}
                for h in rule.history
            ])
            st.dataframe(df_hist, use_container_width=True, height=300)
        else:
            st.caption("Журнал пуст (правило загружено без истории).")

        violations = rule.verify_consistency()
        if violations:
            st.error(f"⚠ Найдено {len(violations)} нарушений непротиворечивости")
            st.dataframe(pd.DataFrame([
                {"x": str(x), "y": str(y), "class(x)": cx, "class(y)": cy}
                for x, y, cx, cy in violations[:20]
            ]))
        else:
            st.success("✓ Условие непротиворечивости выполнено")


# =========================================================================
# Вкладка 4. Критерии
# =========================================================================

with tab_help:
    st.header("Система критериев и классов")
    st.markdown("Подробное описание см. §4 диплома (criteria_system_v3.md).")

    st.subheader("Классы чувствительности")
    df_cls = pd.DataFrame([
        {"#": i + 1, "Код": c.code, "Название": c.name, "Описание": c.description}
        for i, c in enumerate(CLASSES)
    ])
    st.dataframe(df_cls, use_container_width=True, hide_index=True)

    st.subheader("Критерии")
    for c in CRITERIA:
        with st.expander(f"{c.code}. {c.name} (шкала: {c.omega} градаций)"):
            for i, g in enumerate(c.grades, start=1):
                st.markdown(
                    f"**{i}. [{g.code}] {g.title}**\n\n{g.description}\n\n"
                    f"_Примеры:_ {g.examples}"
                )
