from types import SimpleNamespace

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from kg_pipeline.llm.factory import (
    PARSE_REPAIR_ROUNDS,
    get_chat_model,
    with_retrying_structured_output,
)
from kg_pipeline.settings import DEFAULT_OPENAI_BASE_URL, Settings


def _settings(**overrides) -> Settings:
    defaults = {
        "OPENAI_API_KEY": "fake-key",
        "KG_VISION_MODEL": "fake-vision-model",
        "KG_TEXT_MODEL": "fake-text-model",
    }
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


def test_openai_provider_builds_chat_openai():
    chat = get_chat_model("openai", _settings())
    assert type(chat).__name__ == "ChatOpenAI"
    # The endpoint defaults to the one the key name implies, so a user pasting
    # their own key into .env never sends it to a third party by accident.
    assert chat.openai_api_base == DEFAULT_OPENAI_BASE_URL
    assert "academiccloud" not in chat.openai_api_base


def test_openai_key_env_var_is_read():
    """The conventional name works, so most users configure nothing."""
    assert Settings(_env_file=None, OPENAI_API_KEY="k").api_key_for("openai") == "k"


def test_missing_api_key_message_carries_the_fix():
    """"Check your .env file" names neither the variable, the value, nor the
    endpoint the key would be sent to. The message must do all three."""
    settings = _settings(OPENAI_API_KEY=None)
    with pytest.raises(ValueError) as exc:
        get_chat_model("openai", settings)
    msg = str(exc.value)
    assert "OPENAI_API_KEY" in msg
    assert ".env" in msg
    assert DEFAULT_OPENAI_BASE_URL in msg
    assert "main.py check" in msg


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown provider"):
        get_chat_model("bogus", _settings())


def test_models_come_from_settings_and_are_required():
    """There are no built-in model names: they depend on the endpoint."""
    assert get_chat_model("openai", _settings()).model_name == "fake-vision-model"
    assert get_chat_model("openai", _settings(), role="text").model_name == "fake-text-model"
    for role, var in (("vision", "KG_VISION_MODEL"), ("text", "KG_TEXT_MODEL")):
        with pytest.raises(ValueError, match=var):
            get_chat_model("openai", _settings(**{var: None}), role=role)
    # diagram and tests build the pipeline without any configuration at all.
    bare = Settings(_env_file=None)
    assert get_chat_model("openai", bare, allow_missing_key=True).model_name


# --- Parse-repair loop -------------------------------------------------------
# The chat model is stubbed at the with_structured_output seam: the stub
# replays the include_raw=True-shaped dicts the real runnable would return,
# so no network access happens.


class Point(BaseModel):
    x: int
    y: int


def _stub_model(outputs: list[dict], calls: list[list]):
    replies = iter(outputs)

    def with_structured_output(schema, method=None, include_raw=False):
        assert include_raw, "repair loop depends on include_raw=True"

        def run(messages, config=None):
            calls.append(list(messages))
            return next(replies)

        return RunnableLambda(run)

    return SimpleNamespace(with_structured_output=with_structured_output)


def _invalid(error="1 validation error for Point"):
    return {"raw": AIMessage('{"x": "not-an-int"}'), "parsed": None, "parsing_error": error}


def _valid():
    return {"raw": AIMessage('{"x": 1, "y": 2}'), "parsed": Point(x=1, y=2)}


def test_valid_completion_passes_through():
    calls: list[list] = []
    runnable = with_retrying_structured_output(_stub_model([_valid()], calls), Point)

    assert runnable.invoke([HumanMessage("go")]) == Point(x=1, y=2)
    assert len(calls) == 1


def test_invalid_completion_is_repaired_with_error_feedback():
    calls: list[list] = []
    runnable = with_retrying_structured_output(_stub_model([_invalid(), _valid()], calls), Point)

    assert runnable.invoke([HumanMessage("go")]) == Point(x=1, y=2)
    assert len(calls) == 2
    # The repair round sees the original input, the bad completion, and a
    # feedback message quoting the validation errors.
    repair_msgs = calls[1]
    assert repair_msgs[0].content == "go"
    assert repair_msgs[1].content == '{"x": "not-an-int"}'
    assert "1 validation error for Point" in repair_msgs[2].content


def test_repair_budget_exhaustion_raises():
    calls: list[list] = []
    runnable = with_retrying_structured_output(
        _stub_model([_invalid()] * (PARSE_REPAIR_ROUNDS + 1), calls), Point
    )

    with pytest.raises(OutputParserException, match="repair rounds"):
        runnable.invoke([HumanMessage("go")])
    assert len(calls) == PARSE_REPAIR_ROUNDS + 1


def test_openai_is_the_default_provider(monkeypatch):
    """`openai` is the default in settings and the only provider the CLI offers."""
    import importlib.util
    from pathlib import Path

    monkeypatch.delenv("KG_VISION_PROVIDER", raising=False)
    monkeypatch.delenv("KG_TEXT_PROVIDER", raising=False)
    assert Settings(_env_file=None).vision_provider == "openai"
    assert Settings(_env_file=None).text_provider == "openai"

    main_py = Path(__file__).resolve().parents[1] / "main.py"
    spec = importlib.util.spec_from_file_location("main_cli", main_py)
    main_cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main_cli)
    parser = main_cli.build_parser()
    args = parser.parse_args(["generate", "-i", "x", "--provider", "openai"])
    assert args.provider == "openai"
    assert args.text_provider is None
    with pytest.raises(SystemExit):
        parser.parse_args(["generate", "-i", "x", "--provider", "other"])
