from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = REPO_ROOT / "prompts"
MODELS_DIR = REPO_ROOT / "models"


#: Sending a key to an endpoint the user did not choose is the one failure mode
#: worth designing against, so the default is the endpoint that the key name
#: implies. Any other endpoint must be named explicitly.
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"

#: Which environment variables supply each provider's key. Used by messages
#: that have to tell a user exactly what to set; first name is the one to
#: recommend.
KEY_ENV_VARS = {
    "openai": ("OPENAI_API_KEY",),
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_base_url: str = Field(default=DEFAULT_OPENAI_BASE_URL, alias="OPENAI_BASE_URL")
    # Per-request timeout (seconds) for a single LLM call. Healthy calls finish
    # in well under a minute even for the 122b vision model; a call that runs
    # past this is a hung connection, so fail it fast and let the retry layer
    # (with_retrying_structured_output) re-send rather than block for 15 min.
    request_timeout: float = Field(default=300.0, alias="KG_REQUEST_TIMEOUT")

    # Which provider/model each half of the pipeline uses. The vision model
    # serves the image-facing agent (vision_extract) and must accept images; the
    # text model serves the text-only agents (normalize, correct). There are no
    # default model names: which models exist depends on your endpoint, so both
    # must be set (via .env, the environment, or the CLI's --model flags).
    vision_provider: str = Field(default="openai", alias="KG_VISION_PROVIDER")
    vision_model: str | None = Field(default=None, alias="KG_VISION_MODEL")
    text_provider: str = Field(default="openai", alias="KG_TEXT_PROVIDER")
    text_model: str | None = Field(default=None, alias="KG_TEXT_MODEL")

    pdffigures2_jar: Path = MODELS_DIR / "pdffigures2" / "pdffigures2.jar"
    classifier_weights: Path = MODELS_DIR / "classification" / "resnet50_vanilla.pth"
    classifier_classes: Path = MODELS_DIR / "classification" / "classes.txt"

    def api_key_for(self, provider: str) -> str | None:
        if provider == "openai":
            return self.openai_api_key
        raise ValueError(f"Unknown provider: {provider}")


def get_settings() -> Settings:
    return Settings()
