from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ImageDetail = Literal["auto", "high", "low"]
ApiPref = Literal["responses", "chat_completions", "either"]


@dataclass(frozen=True, slots=True)
class Capabilities:
    """
    Minimal registry of model capabilities to gate Responses payload features.
    """

    model: str
    family: str

    supports_responses_api: bool
    supports_chat_completions: bool
    api_preference: ApiPref = "responses"

    is_reasoning_model: bool = False
    supports_reasoning_effort: bool = False
    supports_developer_messages: bool = True

    supports_image_input: bool = False
    supports_image_detail: bool = False
    default_ocr_detail: ImageDetail = "high"

    supports_structured_outputs: bool = True
    supports_function_calling: bool = True

    supports_sampler_controls: bool = True


def _norm(name: str) -> str:
    return name.strip().lower()


def detect_capabilities(model_name: str) -> Capabilities:
    m = _norm(model_name)

    # GPT-5 family (reasoning, vision, structured outputs, no sampler controls)
    if m.startswith("gpt-5"):
        return Capabilities(
            model=model_name,
            family="gpt-5",
            supports_responses_api=True,
            supports_chat_completions=False,
            api_preference="responses",
            is_reasoning_model=True,
            supports_reasoning_effort=True,
            supports_developer_messages=True,
            supports_image_input=True,
            supports_image_detail=True,
            default_ocr_detail="high",
            supports_structured_outputs=True,
            supports_function_calling=True,
            supports_sampler_controls=False,
        )

    # o3 (reasoning, vision, avoid sampler controls)
    if m == "o3" or m.startswith("o3-") and not m.startswith("o3-mini"):
        return Capabilities(
            model=model_name,
            family="o3",
            supports_responses_api=True,
            supports_chat_completions=True,
            api_preference="responses",
            is_reasoning_model=True,
            supports_reasoning_effort=False,
            supports_developer_messages=True,
            supports_image_input=True,
            supports_image_detail=True,
            default_ocr_detail="high",
            supports_structured_outputs=False,
            supports_function_calling=True,
            supports_sampler_controls=False,
        )

    # o3-mini (reasoning, no vision)
    if m.startswith("o3-mini"):
        return Capabilities(
            model=model_name,
            family="o3-mini",
            supports_responses_api=True,
            supports_chat_completions=True,
            api_preference="responses",
            is_reasoning_model=True,
            supports_reasoning_effort=False,
            supports_developer_messages=True,
            supports_image_input=False,
            supports_image_detail=False,
            default_ocr_detail="high",
            supports_structured_outputs=True,
            supports_function_calling=True,
            supports_sampler_controls=False,
        )

    # o1 (full reasoning; no sampler controls; allow Responses)
    if m == "o1" or m.startswith("o1-20") or (m.startswith("o1") and not m.startswith("o1-mini")):
        return Capabilities(
            model=model_name,
            family="o1",
            supports_responses_api=True,
            supports_chat_completions=True,
            api_preference="either",
            is_reasoning_model=True,
            supports_reasoning_effort=False,
            supports_developer_messages=True,
            supports_image_input=True,
            supports_image_detail=True,
            default_ocr_detail="high",
            supports_structured_outputs=False,
            supports_function_calling=True,
            supports_sampler_controls=False,
        )

    # o1-mini (small reasoning; prefer Chat Completions; no vision)
    if m.startswith("o1-mini"):
        return Capabilities(
            model=model_name,
            family="o1-mini",
            supports_responses_api=False,
            supports_chat_completions=True,
            api_preference="chat_completions",
            is_reasoning_model=True,
            supports_reasoning_effort=False,
            supports_developer_messages=False,
            supports_image_input=False,
            supports_image_detail=False,
            default_ocr_detail="high",
            supports_structured_outputs=False,
            supports_function_calling=False,
            supports_sampler_controls=False,
        )

    # GPT-4o family (multimodal; structured outputs; sampler controls)
    if m.startswith("gpt-4o"):
        return Capabilities(
            model=model_name,
            family="gpt-4o",
            supports_responses_api=True,
            supports_chat_completions=True,
            api_preference="responses",
            is_reasoning_model=False,
            supports_reasoning_effort=False,
            supports_developer_messages=True,
            supports_image_input=True,
            supports_image_detail=True,
            default_ocr_detail="high",
            supports_structured_outputs=True,
            supports_function_calling=True,
            supports_sampler_controls=True,
        )

    # GPT-4.1 family (multimodal; structured outputs; sampler controls)
    if m.startswith("gpt-4.1"):
        return Capabilities(
            model=model_name,
            family="gpt-4.1",
            supports_responses_api=True,
            supports_chat_completions=True,
            api_preference="responses",
            is_reasoning_model=False,
            supports_reasoning_effort=False,
            supports_developer_messages=True,
            supports_image_input=True,
            supports_image_detail=True,
            default_ocr_detail="high",
            supports_structured_outputs=True,
            supports_function_calling=True,
            supports_sampler_controls=True,
        )

    # Fallback conservative text-only
    return Capabilities(
        model=model_name,
        family="unknown",
        supports_responses_api=True,
        supports_chat_completions=True,
        api_preference="responses",
        is_reasoning_model=False,
        supports_reasoning_effort=False,
        supports_developer_messages=True,
        supports_image_input=False,
        supports_image_detail=False,
        default_ocr_detail="high",
        supports_structured_outputs=True,
        supports_function_calling=True,
        supports_sampler_controls=True,
    )
