import base64
import io

import PIL.Image
from langchain_core.messages import HumanMessage, SystemMessage


def encode_image_safely(image_path: str, max_size: int = 2048) -> tuple[str, str]:
    """Resizes the image if it's too large, then returns (base64_str, mime_type).

    Downscales oversized images and standardizes to JPEG. The cap is 2048 so
    300-DPI figures keep their resolution: small tensor-shape labels only become
    legible above ~1500px. JPEG quality is 92 for the same reason: at 85,
    compression artifacts smear the smallest text.
    """
    with PIL.Image.open(image_path) as img:
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, PIL.Image.Resampling.LANCZOS)

        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=92)
        b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return b64_str, "image/jpeg"


def build_messages(
    system_prompt: str,
    user_prompt: str,
    image_path: str,
) -> list[SystemMessage | HumanMessage]:
    """Builds a (SystemMessage, HumanMessage) pair with the image inline.

    The content-block format below is the OpenAI-compatible one ``ChatOpenAI``
    sends.
    """
    b64_image, mime_type = encode_image_safely(image_path)

    content: list[dict] = [
        {"type": "text", "text": user_prompt},
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{b64_image}"},
        },
    ]

    return [SystemMessage(content=system_prompt), HumanMessage(content=content)]
