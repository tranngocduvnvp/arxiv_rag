from openai import OpenAI


def chat_with_llm(messages, base_url="http://0.0.0.0:23333/v1"):
    client = OpenAI(base_url=base_url, api_key="EMPTY")

    completion = client.chat.completions.create(
        model="Qwen/Qwen3-4B-Instruct-2507",
        messages=messages
    )
    return completion.choices[0].message.content
