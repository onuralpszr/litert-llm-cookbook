"""
Example 04: GPU backend
Run inference on GPU instead of the default CPU backend.
Requires a compatible GPU and the GPU-enabled litert-lm build.
"""

import litert_lm

MODEL_PATH = "gemma-4-E2B-it.litertlm"

litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)

with litert_lm.Engine(
    MODEL_PATH,
    backend=litert_lm.Backend.GPU,
    cache_dir="/tmp/litert-lm-cache",
) as engine:
    with engine.create_conversation() as conversation:
        response = conversation.send_message("Explain attention mechanism in transformers.")
        print(response["content"][0]["text"])
