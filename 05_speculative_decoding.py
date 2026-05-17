"""
Example 05: Multi-Token Prediction / Speculative Decoding (MTP)
Speeds up inference by predicting multiple tokens at once.
Requires GPU backend and a model that supports MTP.
"""

import litert_lm

MODEL_PATH = "gemma-4-E2B-it.litertlm"

litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)

with litert_lm.Engine(
    MODEL_PATH,
    backend=litert_lm.Backend.GPU,
    enable_speculative_decoding=True,
) as engine:
    with engine.create_conversation() as conversation:
        prompts = [
            "Write a short poem about the ocean.",
            "What are the key differences between Python lists and tuples?",
        ]
        for prompt in prompts:
            print(f"Prompt: {prompt}")
            response = conversation.send_message(prompt)
            print(f"Response: {response['content'][0]['text']}\n")
