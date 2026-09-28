from typing import TypeVar

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel

from kg_pipeline.settings import DEFAULT_OPENAI_BASE_URL, KEY_ENV_VARS, Settings

RETRY_ATTEMPTS = 5

# How many times a schema-invalid completion is sent back to the model with
# its validation errors for repair (distinct from RETRY_ATTEMPTS, which
# covers transport failures and retries the *same* input).
PARSE_REPAIR_ROUNDS = 2

REPAIR_PROMPT = (
    "Your previous JSON response failed schema validation with the following "
    "errors:\n\n{errors}\n\nReturn the FULL corrected JSON object again, "
    "changing only what is needed to fix these errors. Do not drop any nodes, "
    "edges, or groups."
)

SchemaT = TypeVar("SchemaT", bound=BaseModel)


def missing_key_message(provider: str) -> str:
    """The error a first-time user actually sees, so it carries the fix.

    "Check your .env file" is true and useless: it names neither the variable
    nor the value nor the endpoint the key would be sent to.
    """
    return (
        f"No API key found. Set {KEY_ENV_VARS['openai'][0]}.\n\n"
        "  cp .env.example .env\n"
        "  # then, in .env or your shell:\n"
        "  OPENAI_API_KEY=sk-...\n"
        f"  OPENAI_BASE_URL={DEFAULT_OPENAI_BASE_URL}   # or any OpenAI-compatible endpoint\n\n"
        "Run `python main.py check` to verify the configuration."
    )


def missing_model_message(role: str) -> str:
    """The error for an unset model name, naming the variable to set."""
    return (
        f"No {role} model set. Set KG_{role.upper()}_MODEL to a model your endpoint "
        f"serves{' (it must accept images)' if role == 'vision' else ''}.\n\n"
        "  # in .env or your shell:\n"
        "  KG_VISION_MODEL=your-vision-model\n"
        "  KG_TEXT_MODEL=your-text-model\n\n"
        "Run `python main.py check` to verify the configuration."
    )


def get_chat_model(
    provider: str,
    settings: Settings,
    model: str | None = None,
    temperature: float = 0.0,
    *,
    allow_missing_key: bool = False,
    role: str = "vision",
) -> BaseChatModel:
    """Builds a plain LangChain chat model for the given provider.

    Use :func:`with_retrying_structured_output` to get a JSON-schema-enforcing,
    retrying runnable from this model.

    ``model`` defaults to the ``vision_model`` or ``text_model`` setting for
    ``role``. ``allow_missing_key=True`` substitutes a placeholder key and model
    name so the model can be *constructed* without configuration -- enough to
    build and visualize the pipeline graph (``main.py diagram``, tests); any
    actual invoke would fail.
    """
    model_id = model or getattr(settings, f"{role}_model")
    if not model_id:
        if not allow_missing_key:
            raise ValueError(missing_model_message(role))
        model_id = "placeholder-model-not-set"
    api_key = settings.api_key_for(provider)

    if not api_key:
        if not allow_missing_key:
            raise ValueError(missing_key_message(provider))
        api_key = "placeholder-key-not-set"

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_id,
            api_key=api_key,
            base_url=settings.openai_base_url,
            temperature=temperature,
            timeout=settings.request_timeout,
            # The OpenAI client retries internally (default 2); layered under our
            # own .with_retry() that multiplied out to ~35 min on one hung call.
            # Disable the inner layer so retries don't compound -- a single
            # transport-retry policy lives in with_retrying_structured_output.
            max_retries=0,
        )

    raise ValueError(f"Unknown provider: {provider}")


def with_retrying_structured_output(
    chat_model: BaseChatModel, schema: type[SchemaT]
) -> Runnable[list, SchemaT]:
    """Binds a Pydantic schema for JSON-mode structured output, wrapped in
    two distinct recovery layers:

    - ``.with_retry()`` re-sends the same input on *transport* failures
      (timeouts, 5xx) with exponential backoff.
    - A *repair loop* handles schema-invalid completions: retrying the same
      input at temperature 0 just reproduces the same bad JSON, so instead the
      invalid completion plus its Pydantic errors are appended to the
      conversation and the model is asked to fix them (``include_raw=True``
      surfaces parse failures as data rather than exceptions). Benchmark
      run_122b lost 3/73 figures to exactly this: one symbolic property value
      failing validation and five identical-input retries changing nothing.

    Retry must wrap the *structured* runnable, not the raw chat model —
    ``with_retry()`` returns a ``RunnableRetry`` that doesn't proxy
    chat-model-specific methods like ``with_structured_output``.
    """
    structured = chat_model.with_structured_output(
        schema, method="json_mode", include_raw=True
    ).with_retry(
        stop_after_attempt=RETRY_ATTEMPTS,
        wait_exponential_jitter=True,
    )

    def _invoke(messages: list, config=None) -> SchemaT:
        msgs = list(messages)
        error = None
        for _ in range(PARSE_REPAIR_ROUNDS + 1):
            out = structured.invoke(msgs, config=config)
            if out.get("parsed") is not None:
                return out["parsed"]
            error = out.get("parsing_error")
            msgs = msgs + [out["raw"], HumanMessage(REPAIR_PROMPT.format(errors=error))]
        raise OutputParserException(
            f"Completion still failed {schema.__name__} validation after "
            f"{PARSE_REPAIR_ROUNDS} repair rounds: {error}"
        )

    return RunnableLambda(_invoke)
