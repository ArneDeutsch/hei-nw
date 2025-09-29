from hei_nw.models.base import DEFAULT_MEMORY_SYSTEM_PROMPT, build_prompt, load_base
from hei_nw.testing import DUMMY_MODEL_ID


def test_memory_hint_in_chat_prompt() -> None:
    tokenizer, _, _ = load_base(model_id=DUMMY_MODEL_ID, quant_4bit=False)
    messages = [
        {"role": "system", "content": "Follow the episode instructions."},
        {"role": "user", "content": "Question: Who left the red bag?"},
    ]
    rendered = build_prompt(
        tokenizer,
        messages,
        "chat",
        template_policy="plain",
        memory_prompt="who: Dana",
        memory_system_prompt=DEFAULT_MEMORY_SYSTEM_PROMPT,
    )
    lines = [line.strip() for line in rendered.splitlines() if line.strip()]
    assert lines[0].startswith("SYSTEM:")
    assert DEFAULT_MEMORY_SYSTEM_PROMPT in lines[0]
    assert any("Follow the episode instructions." in line for line in lines)
