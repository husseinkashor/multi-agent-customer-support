
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:12434/engines/v1",
    api_key="not-needed",
)

response = client.chat.completions.create(
    model="ai/qwen2.5:3B-Q4_K_M",
    messages=[
        {
            "role": "system",
            "content": "You are an assistant in a customer support system."
        },
        {
            "role": "user",
            "content": "A customer says their order is 10 days late. How would you respond?"
        }
    ],
    temperature=0.3,
    max_tokens=200,
)

print("\nQwen response:")
print(response.choices[0].message.content)
