"""
Example 08: Multimodal input — Vision (image)
Pass an image file alongside a text prompt.
Requires a model with vision support and vision_backend configured.

Note: gemma-4-E2B-it.litertlm may not support vision — use a
multimodal variant (e.g. gemma-4-it-multimodal.litertlm) if available.
"""

import litert_lm

MODEL_PATH = "gemma-4-E2B-it.litertlm"
IMAGE_FILE = "image.jpg"  # replace with a real image path

litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)

with litert_lm.Engine(
    MODEL_PATH,
    vision_backend=litert_lm.Backend.GPU,
) as engine:
    with engine.create_conversation() as conversation:
        user_message = {
            "role": "user",
            "content": [
                {"type": "image", "path": IMAGE_FILE},
                {"type": "text", "text": "Describe what you see in this image."},
            ],
        }
        response = conversation.send_message(user_message)
        print(response["content"][0]["text"])
