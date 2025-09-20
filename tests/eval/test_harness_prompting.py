from pathlib import Path

from hei_nw.eval.harness import (
    ModelGeometry,
    QAPromptSettings,
    _build_prompt,
    _evaluate_records,
    _qa_settings_from_args,
    parse_args,
)
from hei_nw.models.base import build_prompt, generate, load_base

TINY_MODEL = Path(__file__).resolve().parents[2] / "models" / "tiny-gpt2"


def test_prompt_styles_and_stop_behavior() -> None:
    tok, _, _ = load_base(model_id=str(TINY_MODEL), quant_4bit=False)
    record = {
        "episode_text": "Yesterday, Alice bought a red apple from the market.",
        "cues": ["Who bought the apple?"],
        "answers": ["Alice"],
    }

    qa_plain = QAPromptSettings(
        prompt_style="plain", max_new_tokens=8, stop="\n", answer_hint=True
    )
    plain_prompt, _ = _build_prompt(
        record,
        prompt_style=qa_plain.prompt_style,
        answer_hint=qa_plain.answer_hint,
        omit_episode=qa_plain.omit_episode,
    )
    plain_output = generate(
        plain_prompt,
        max_new_tokens=qa_plain.max_new_tokens,
        stop=qa_plain.stop_value(),
        prompt_style=qa_plain.prompt_style,
        stop_mode=qa_plain.stop_mode,
        template_policy=qa_plain.template_policy,
    )
    assert "\n" not in plain_output["text"]
    assert plain_output["generated_tokens"] <= qa_plain.max_new_tokens
    assert "ASSISTANT:" not in plain_output["text"]

    qa_chat = QAPromptSettings(
        prompt_style="chat", max_new_tokens=8, stop="\n", answer_hint=True
    )
    chat_prompt, _ = _build_prompt(
        record,
        prompt_style=qa_chat.prompt_style,
        answer_hint=qa_chat.answer_hint,
        omit_episode=qa_chat.omit_episode,
    )
    assert isinstance(chat_prompt, list)
    assert any("no Markdown" in msg.get("content", "") for msg in chat_prompt if msg.get("role") == "user")
    rendered = build_prompt(
        tok, chat_prompt, qa_chat.prompt_style, template_policy=qa_chat.template_policy
    )
    assert "Question:" in rendered
    chat_output = generate(
        chat_prompt,
        max_new_tokens=qa_chat.max_new_tokens,
        stop=qa_chat.stop_value(),
        prompt_style=qa_chat.prompt_style,
        stop_mode=qa_chat.stop_mode,
        template_policy=qa_chat.template_policy,
    )
    assert "\n" not in chat_output["text"]
    assert chat_output["generated_tokens"] <= qa_chat.max_new_tokens
    assert chat_output["text"].strip() != ""

    qa_chat_plain_policy = QAPromptSettings(
        prompt_style="chat",
        max_new_tokens=8,
        stop="\n",
        answer_hint=True,
        template_policy="plain",
    )
    fallback_output = generate(
        chat_prompt,
        max_new_tokens=qa_chat_plain_policy.max_new_tokens,
        stop=qa_chat_plain_policy.stop_value(),
        prompt_style=qa_chat_plain_policy.prompt_style,
        stop_mode=qa_chat_plain_policy.stop_mode,
        template_policy=qa_chat_plain_policy.template_policy,
    )
    assert fallback_output["text"].strip() != ""
    assert "ASSISTANT:" not in fallback_output["text"]


def test_scenario_a_defaults_apply(monkeypatch, tmp_path) -> None:
    args = parse_args(
        [
            "--mode",
            "B0",
            "--scenario",
            "A",
            "-n",
            "1",
            "--outdir",
            str(tmp_path),
        ]
    )
    qa_settings = _qa_settings_from_args(args)
    assert qa_settings.prompt_style == "chat"
    assert qa_settings.max_new_tokens == 16
    assert qa_settings.stop is None
    assert qa_settings.answer_hint is True
    assert qa_settings.template_policy == "auto"
    assert qa_settings.stop_mode == "none"
    assert qa_settings.omit_episode is True
    assert qa_settings.memory_dependent_baseline is False

    records = [
        {
            "episode_text": "Yesterday, Alice bought a red apple from the market.",
            "cues": ["Who bought the apple?"],
            "answers": ["Alice"],
        }
    ]

    captured: dict[str, object] = {}

    def fake_generate(
        prompt,
        *,
        max_new_tokens,
        adapter=None,
        mem_tokens=None,
        stop,
        prompt_style,
        stop_mode,
        template_policy,
        **_: object,
    ) -> dict[str, object]:
        captured["prompt"] = prompt
        captured["max_new_tokens"] = max_new_tokens
        captured["stop"] = stop
        captured["prompt_style"] = prompt_style
        captured["stop_mode"] = stop_mode
        captured["template_policy"] = template_policy
        return {"text": "Alice", "generated_tokens": 1, "prompt_tokens": 1}

    monkeypatch.setattr("hei_nw.models.base.generate", fake_generate)

    geom = ModelGeometry(layers=1, hidden=1, heads=1, dtype="float32")
    items, _ = _evaluate_records(records, geom, qa_settings)

    assert captured["max_new_tokens"] == 16
    assert captured["stop"] is None
    assert captured["prompt_style"] == "chat"
    assert captured["stop_mode"] == "none"
    assert captured["template_policy"] == "auto"
    assert isinstance(captured["prompt"], list)
    assert items[0].prediction == "Alice"


def test_omit_episode_flag_changes_prompt() -> None:
    record = {
        "episode_text": "Yesterday, Alice bought a red apple from the market.",
        "cues": ["Who bought the apple?"],
        "answers": ["Alice"],
    }

    qa_settings = QAPromptSettings(
        prompt_style="plain",
        max_new_tokens=8,
        stop=None,
        answer_hint=True,
        omit_episode=True,
    )
    prompt, _ = _build_prompt(
        record,
        prompt_style=qa_settings.prompt_style,
        answer_hint=qa_settings.answer_hint,
        omit_episode=qa_settings.omit_episode,
    )
    assert "Episode:" not in prompt
    assert "Alice bought" not in prompt


def test_memory_dependent_default_differs_between_modes(tmp_path) -> None:
    records = [
        {
            "episode_text": "Yesterday, Alice bought a red apple from the market.",
            "cues": ["Who bought the apple?"],
            "answers": ["Alice"],
        }
    ]

    b0_args = parse_args(
        [
            "--mode",
            "B0",
            "--scenario",
            "A",
            "-n",
            "1",
            "--outdir",
            str(tmp_path / "b0"),
        ]
    )
    b0_settings = _qa_settings_from_args(b0_args)
    b0_prompt, _ = _build_prompt(
        records[0],
        prompt_style=b0_settings.prompt_style,
        answer_hint=b0_settings.answer_hint,
        omit_episode=b0_settings.omit_episode,
    )
    assert "Episode:" not in b0_prompt if isinstance(b0_prompt, str) else "Episode:" not in b0_prompt[1]["content"]

    b1_args = parse_args(
        [
            "--mode",
            "B1",
            "--scenario",
            "A",
            "-n",
            "1",
            "--outdir",
            str(tmp_path / "b1"),
        ]
    )
    b1_settings = _qa_settings_from_args(b1_args)
    b1_prompt, _ = _build_prompt(
        records[0],
        prompt_style=b1_settings.prompt_style,
        answer_hint=b1_settings.answer_hint,
        omit_episode=b1_settings.omit_episode,
    )
    if isinstance(b1_prompt, str):
        assert "Alice bought" in b1_prompt
    else:
        assert any("Alice bought" in msg.get("content", "") for msg in b1_prompt)

    b1_baseline_args = parse_args(
        [
            "--mode",
            "B1",
            "--scenario",
            "A",
            "-n",
            "1",
            "--outdir",
            str(tmp_path / "b1-baseline"),
            "--qa.memory_dependent_baseline",
        ]
    )
    b1_baseline_settings = _qa_settings_from_args(b1_baseline_args)
    assert b1_baseline_settings.memory_dependent_baseline is True
    assert b1_baseline_settings.omit_episode is True


def test_parse_args_supports_omit_episode(tmp_path) -> None:
    args = parse_args(
        [
            "--mode",
            "B0",
            "--scenario",
            "A",
            "-n",
            "1",
            "--outdir",
            str(tmp_path),
            "--qa.omit_episode",
        ]
    )
    qa_settings = _qa_settings_from_args(args)
    assert qa_settings.omit_episode is True
