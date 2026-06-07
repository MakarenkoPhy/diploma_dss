import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ['DEEPSEEK_API_KEY'],
    base_url='https://api.artemox.com/v1',
    timeout=15.0,
)

resp = client.chat.completions.create(
    model='deepseek-chat',
    messages=[{'role': 'user', 'content': 'Привет! Ответь одним словом.'}],
    max_tokens=10,
)

print('Ответ:', resp.choices[0].message.content)
print('finish_reason:', resp.choices[0].finish_reason)