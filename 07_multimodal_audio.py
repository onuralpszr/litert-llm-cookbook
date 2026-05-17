"""
Example 07: Multimodal input — Audio
Pass an audio file alongside a text prompt.
Requires a model with audio support and audio_backend configured.

Note: gemma-4-E2B-it.litertlm may not support audio — use a
multimodal variant (e.g. gemma-4-it-multimodal.litertlm) if available.
"""

import litert_lm

MODEL_PATH = "gemma-4-E2B-it.litertlm"
AUDIO_FILE = "audio.wav"  # replace with a real .wav path

litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)

with litert_lm.Engine(
    MODEL_PATH,
    audio_backend=litert_lm.Backend.CPU,
) as engine:
    with engine.create_conversation() as conversation:
        user_message = {
            "role": "user",
            "content": [
                {"type": "audio", "path": AUDIO_FILE},
                {"type": "text", "text": "Transcribe and summarise this audio."},
            ],
        }
        response = conversation.send_message(user_message)
        print(response["content"][0]["text"])
