from __future__ import annotations

import json


class ManualLLM:
    def __init__(self, show_last: int = 6) -> None:
        self.show_last = show_last

    async def chat(self, messages: list[dict[str, str]], temperature: float = 0.4) -> str:
        print("\n=== CONTEXT ===")
        for msg in messages[-self.show_last :]:
            role = msg.get("role")
            content = msg.get("content")
            if role == "user" and content and content.startswith("TOOL_RESULT"):
                try:
                    prefix, payload = content.split(":", 1)
                    data = json.loads(payload.strip())
                    content = f"{prefix}: {json.dumps(data, ensure_ascii=False)[:800]}"
                except Exception:
                    pass
            print(f"[{role}] {content}")
        print("=== END ===\n")
        return input("LLM JSON > ")
