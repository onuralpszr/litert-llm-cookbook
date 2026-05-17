"""
Example 01: Basic synchronous chat
Run the model once, get a single response.

Download the model first:
  litert-lm run \
    --from-huggingface-repo=litert-community/gemma-4-E2B-it-litert-lm \
    gemma-4-E2B-it.litertlm \
    --prompt="What is the capital of Turkey?"
"""

import litert_lm

MODEL_PATH = "gemma-4-E2B-it.litertlm"

litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)

with litert_lm.Engine(MODEL_PATH) as engine:
    with engine.create_conversation() as conversation:
        response = conversation.send_message("What is the capital of Turkey?")
        print(response["content"][0]["text"])
