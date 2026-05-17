"""
Example 03: System prompt / conversation history pre-seeding
Pass an initial messages list to set the model's persona or context
before the first user turn.
"""

import litert_lm

MODEL_PATH = "gemma-4-E2B-it.litertlm"

litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)

SYSTEM_PROMPT = (
    "You are a concise Python expert. "
    "Always respond with short, clear explanations and include a code example when relevant."
)

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": SYSTEM_PROMPT}],
    }
]

with litert_lm.Engine(MODEL_PATH) as engine:
    with engine.create_conversation(messages=messages) as conversation:
        questions = [
            "What is a list comprehension?",
            "How do I read a file line by line?",
        ]
        for q in questions:
            print(f"Q: {q}")
            response = conversation.send_message(q)
            print(f"A: {response['content'][0]['text']}\n")
