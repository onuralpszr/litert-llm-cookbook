"""
Example 09: Streaming output + system prompt combined
Sets a persona via system message and streams the response token-by-token.
"""

import litert_lm

MODEL_PATH = "gemma-4-E2B-it.litertlm"

litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)

messages = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": (
                    "You are a senior software engineer at Google specialising in "
                    "on-device AI and TensorFlow Lite. Give concise, technical answers."
                ),
            }
        ],
    }
]

with litert_lm.Engine(MODEL_PATH) as engine:
    with engine.create_conversation(messages=messages) as conversation:
        user_input = "What are the main advantages of running LLMs on-device vs in the cloud?"
        print(f"Q: {user_input}\nA: ", end="", flush=True)
        for chunk in conversation.send_message_async(user_input):
            for item in chunk.get("content", []):
                if item.get("type") == "text":
                    print(item["text"], end="", flush=True)
        print()
