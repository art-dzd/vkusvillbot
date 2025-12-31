import json

from vkusvillbot.sgr_agent import FinalAnswer, ToolCall, parse_llm_output


def test_parse_tool_call() -> None:
    payload = {
        "action": "tool_call",
        "tool": "vkusvill_products_search",
        "args": {"q": "молоко", "page": 1},
        "reason": "нужно найти товары",
    }
    parsed = parse_llm_output(json.dumps(payload, ensure_ascii=False))
    assert isinstance(parsed, ToolCall)
    assert parsed.tool == "vkusvill_products_search"


def test_parse_final() -> None:
    payload = {
        "action": "final",
        "answer": "Готово",
        "cart_items": [{"xml_id": 1, "q": 1}],
    }
    parsed = parse_llm_output(json.dumps(payload, ensure_ascii=False))
    assert isinstance(parsed, FinalAnswer)
    assert parsed.answer == "Готово"


def test_parse_json_with_noise() -> None:
    payload = {"action": "final", "answer": "ok"}
    text = f"текст\n{json.dumps(payload, ensure_ascii=False)}\nещё"
    parsed = parse_llm_output(text)
    assert isinstance(parsed, FinalAnswer)
