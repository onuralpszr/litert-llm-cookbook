"""
Example 02: Streaming / async interactive chat loop
Tokens are printed as they are generated (streaming output).
Type 'exit' or 'quit' to stop.
"""

import litert_lm

MODEL_PATH = "gemma-4-E2B-it.litertlm"

litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)

with litert_lm.Engine(MODEL_PATH) as engine:
    with engine.create_conversation() as conversation:
        print("Gemma-4 E2B chat (streaming) — type 'exit' to quit\n")
        while True:
            user_input = input(">>> ").strip()
            if user_input.lower() in {"exit", "quit"}:
                break
            if not user_input:
                continue

            print("", end="")
            for chunk in conversation.send_message_async(user_input):
                for item in chunk.get("content", []):
                    if item.get("type") == "text":
                        print(item["text"], end="", flush=True)
            print()
