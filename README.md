# Гибридная СППР: ЦИКЛ + LLM для разметки чувствительных данных

Реализация архитектуры, описанной в Главе 5 диплома: интеграция метода
многокритериальной классификации **ЦИКЛ** с большой языковой моделью для
классификации столбцов банковского хранилища по уровням конфиденциальности.

## Архитектура

**Принципиальное разделение ролей LLM:**
- Фаза 1 (offline, один раз). LLM выступает в роли **ЛПР**: получает
  гипотетический вектор оценок (K₁..K₆) и относит его к одному из 4 классов.
  Метод ЦИКЛ обеспечивает непротиворечивость через процедуры S
  (распространение по доминированию) и R (устранение противоречий).
  На выходе — полная таблица (вектор → класс) на 1296 векторов.
- Фаза 2 (online). LLM выступает в роли **оценщика**: получает описание
  реального столбца и возвращает вектор оценок + reasoning. Класс
  определяется детерминированно через lookup в таблице, построенной
  на Фазе 1.

## Структура

```
sensitive_data_dss/
├── dss/
│   ├── criteria.py            — 6 критериев и 4 класса (§4 диплома)
│   ├── tsikl.py               — ядро метода ЦИКЛ
│   ├── llm_client.py          — Anthropic / OpenAI / DeepSeek / Manual
│   ├── decision_maker.py      — Фаза 1: LLM как ЛПР
│   ├── assessor.py            — Фаза 2: LLM как оценщик
│   ├── decision_rule.py       — модель правила, save/load/verify
│   └── pipeline.py            — связка двух фаз
├── build_rule.py              — CLI Фазы 1
├── classify.py                — CLI Фазы 2 (одиночный + batch)
├── app.py                     — Streamlit-фронтенд
├── example_columns.csv        — пример входа для batch-классификации
└── requirements.txt
```

## Установка

```bash
cd sensitive_data_dss
pip install -r requirements.txt

# Один из:
export ANTHROPIC_API_KEY="..."
export OPENAI_API_KEY="..."
export DEEPSEEK_API_KEY="..."
```

## Использование

### Фаза 1. Построение решающего правила

```bash
python build_rule.py --provider anthropic --output rule.json
python build_rule.py --provider openai --model gpt-4o --output rule_gpt.json
python build_rule.py --provider manual --output rule_human.json
```

### Фаза 2. Оперативная классификация

Один столбец интерактивно:
```bash
python classify.py --rule rule.json
```

Один столбец через аргументы:
```bash
python classify.py --rule rule.json \
    --field client_inn_fl \
    --description "ИНН клиента — физического лица"
```

Пакетная обработка из CSV:
```bash
python classify.py --rule rule.json \
    --input example_columns.csv \
    --output results.csv
```

### Веб-интерфейс

```bash
streamlit run app.py
```

## Программное использование

```python
from dss import (
    make_client, build_decision_rule,
    DecisionRule, LLMAssessor, ColumnDescription, classify_column,
)

# Фаза 1 (один раз)
llm = make_client("anthropic")
rule = build_decision_rule(llm, progress=print)
rule.save("rule.json")

# Фаза 2 (многократно)
rule = DecisionRule.load("rule.json")
assessor = LLMAssessor(make_client("anthropic"))

result = classify_column(
    ColumnDescription(
        field_name="client_inn",
        description="ИНН клиента — физического лица",
    ),
    rule, assessor,
)
print(result.class_code, result.class_name)
print(result.assessment.reasoning)
```

## Замечания по реализации алгоритма ЦИКЛ

Стандартная реализация Asanov et al. предусматривает рекурсивное углубление
`procedure_D` до тех пор, пока есть любой разрыв классов между крайними
точками подцепи. Для гибридной СППР это нерационально, поскольку каждый
запрос к LLM имеет фиксированную стоимость (время и деньги). Поэтому
введены две оптимизации:

1. **Раннее прекращение `procedure_D`** при разрыве классов = 1: граница
    уже локализована, дальнейшее углубление не даёт нового знания о
    структуре правила.
2. **Эвристическое достраивание** оставшихся неоднозначных векторов через
    `⌊(C^L + C^U) / 2⌋` без обращения к ЛПР. Округление к меньшему номеру
    класса соответствует более чувствительному классу — безопасное
    поведение для задачи защиты данных.

На тестах с детерминированным mock-ЛПР алгоритм опрашивает порядка
**2.5–4 % от |Y|** (для 1296 векторов — 30–55 запросов), что соответствует
ожидаемому диапазону по [Asanov et al., 2001] и обоснованию §4.6 диплома.
Реальная цифра при работе с LLM может варьироваться в зависимости от
поведения конкретной модели.

Корректность (отсутствие нарушений условия непротиворечивости (2))
проверяется автоматически после построения правила.
