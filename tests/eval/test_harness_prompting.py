from pathlib import Path

from hei_nw.eval.harness import QAPromptSettings, _build_prompt
from hei_nw.models.base import build_prompt, generate, load_base

TINY_MODEL = Path(__file__).resolve().parents[2] / "models" / "tiny-gpt2"


def test_prompt_styles_and_stop_behavior() -> None:
    tok, _, _ = load_base(model_id=str(TINY_MODEL), quant_4bit=False)
    record = {
        "episode_text": "Yesterday, Alice bought a red apple from the market.",
        "cues": ["Who bought the apple?"],
        "answers": ["Alice"],
    }

    qa_plain = QAPromptSettings(prompt_style="plain", max_new_tokens=8, stop="\n", answer_hint=True)
    plain_prompt, _ = _build_prompt(
        record,
        prompt_style=qa_plain.prompt_style,
        answer_hint=qa_plain.answer_hint,
    )
    plain_output = generate(
        plain_prompt,
        max_new_tokens=qa_plain.max_new_tokens,
        stop=qa_plain.stop_value(),
        prompt_style=qa_plain.prompt_style,
    )
    assert "\n" not in plain_output["text"]
    assert plain_output["generated_tokens"] <= qa_plain.max_new_tokens

    qa_chat = QAPromptSettings(prompt_style="chat", max_new_tokens=8, stop="\n", answer_hint=True)
    chat_prompt, _ = _build_prompt(
        record,
        prompt_style=qa_chat.prompt_style,
        answer_hint=qa_chat.answer_hint,
    )
    assert isinstance(chat_prompt, list)
    rendered = build_prompt(tok, chat_prompt, qa_chat.prompt_style)
    assert "Question:" in rendered
    chat_output = generate(
        chat_prompt,
        max_new_tokens=qa_chat.max_new_tokens,
        stop=qa_chat.stop_value(),
        prompt_style=qa_chat.prompt_style,
    )
    assert "\n" not in chat_output["text"]
    assert chat_output["generated_tokens"] <= qa_chat.max_new_tokens
