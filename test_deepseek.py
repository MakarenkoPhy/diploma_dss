import os
import sys
sys.path.insert(0, '.')
from openai import OpenAI

client = OpenAI(
    api_key=os.environ['DEEPSEEK_API_KEY'],
    base_url='https://api.artemox.com/v1'
)

# Тест 1: короткий промпт
print("=== Тест 1: короткий промпт ===")
resp = client.chat.completions.create(
    model='deepseek-chat',
    messages=[
        {'role': 'system', 'content': 'Ты эксперт. Отвечай только JSON.'},
        {'role': 'user', 'content': 'Верни json: {"K1": 1, "K2": 2}'}
    ],
    max_tokens=50
)
print(f"Ответ: {repr(resp.choices[0].message.content)}")
print(f"finish_reason: {resp.choices[0].finish_reason}")

# Тест 2: полный промпт оценщика
print()
print("=== Тест 2: полный промпт оценщика ===")
from dss.assessor import build_assessor_system_prompt
prompt = build_assessor_system_prompt()
print(f"Длина системного промпта: {len(prompt)} символов, ~{len(prompt)//4} токенов")

resp2 = client.chat.completions.create(
    model='deepseek-chat',
    messages=[
        {'role': 'system', 'content': prompt},
        {'role': 'user', 'content': (
            "Столбец данных для оценки:\n"
            "  * Название таблицы: sdp_digital_profile.kid_rf_brth_cert_info\n"
            "  * Описание таблицы: Содержит сведения об элементе kidRfBrthCert\n"
            "  * Название поля: registry_office_name\n"
            "  * Описание поля: Наименование органа ЗАГС\n\n"
            "Оцени столбец по 6 критериям и верни ответ в формате JSON."
        )}
    ],
    max_tokens=800
)
content = resp2.choices[0].message.content
print(f"finish_reason: {resp2.choices[0].finish_reason}")
print(f"Длина ответа: {len(content) if content else 0} символов")
print(f"Первые 300 символов: {repr(content[:300]) if content else 'ПУСТОЙ ОТВЕТ'}")
