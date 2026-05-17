"""
Example 10: Kitchen-sink demo — GPU + speculative decoding + tools + streaming
Combines multiple features in one script to demonstrate a production-like setup.
"""

import litert_lm

MODEL_PATH = "gemma-4-E2B-it.litertlm"

litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)


def celsius_to_fahrenheit(celsius: float) -> float:
    """Converts a temperature from Celsius to Fahrenheit.

    Args:
        celsius: Temperature in Celsius.
    """
    return celsius * 9 / 5 + 32


def word_count(text: str) -> int:
    """Counts the number of words in a text string.

    Args:
        text: The input text.
    """
    return len(text.split())


system_messages = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are a helpful assistant. Use tools when they help answer the question precisely.",
            }
        ],
    }
]

tools = [celsius_to_fahrenheit, word_count]

with litert_lm.Engine(
    MODEL_PATH,
    backend=litert_lm.Backend.GPU,
    enable_speculative_decoding=True,
    cache_dir="/tmp/litert-lm-cache",
) as engine:
    with engine.create_conversation(messages=system_messages, tools=tools) as conversation:
        print("=== Kitchen-sink demo: GPU + MTP + tools + streaming ===\n")

        queries = [
            "Convert 37 degrees Celsius to Fahrenheit.",
            "How many words are in the sentence: 'The quick brown fox jumps over the lazy dog'?",
            "Tell me a fun fact about on-device AI. Keep it to two sentences.",
        ]

        for query in queries:
            print(f"Q: {query}")
            print("A: ", end="", flush=True)
            for chunk in conversation.send_message_async(query):
                for item in chunk.get("content", []):
                    if item.get("type") == "text":
                        print(item["text"], end="", flush=True)
            print("\n")
