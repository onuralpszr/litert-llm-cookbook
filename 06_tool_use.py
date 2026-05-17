"""
Example 06: Tool use / function calling
Define Python functions and pass them as tools. The model can call
them when it decides the question requires computation.
"""

import litert_lm

MODEL_PATH = "gemma-4-E2B-it.litertlm"

litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)


def add_numbers(a: float, b: float) -> float:
    """Adds two numbers.

    Args:
        a: The first number.
        b: The second number.
    """
    return a + b


def multiply_numbers(a: float, b: float) -> float:
    """Multiplies two numbers.

    Args:
        a: The first number.
        b: The second number.
    """
    return a * b


def get_current_weather(city: str) -> str:
    """Returns a mock weather report for a given city.

    Args:
        city: The name of the city.
    """
    # Stub — replace with a real weather API call
    return f"The weather in {city} is sunny and 22°C."


tools = [add_numbers, multiply_numbers, get_current_weather]

with litert_lm.Engine(MODEL_PATH) as engine:
    with engine.create_conversation(tools=tools) as conversation:
        queries = [
            "What is 123 + 456?",
            "What is 7 multiplied by 8?",
            "What is the weather in Istanbul?",
        ]
        for query in queries:
            print(f"Q: {query}")
            response = conversation.send_message(query)
            print(f"A: {response['content'][0]['text']}\n")
