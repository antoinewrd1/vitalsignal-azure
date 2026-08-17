"""One place that knows how to talk to a model.

Two backends:

  azure_openai -- the real thing. Entra ID (`DefaultAzureCredential`) is the
                  default auth path because a managed identity beats a key in
                  an app setting; the key path exists only as a fallback for
                  local debugging.

  fake         -- a deterministic, rule-based stand-in. It exists so that
                  `pytest` and CI run with no network, no key and no cost, and
                  so the eval harness itself can be tested.

Be blunt about what the fake is: it is regex over the note text. Any eval score
produced against the fake measures the *harness*, not a model. Real numbers
require VS_LLM_BACKEND=azure_openai. The eval report labels which backend
produced it for exactly this reason.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, ClassVar

from vitalsignal.config import get_settings


@dataclass
class LLMResponse:
    text: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    attempts: int = 1


class BaseLLM:
    name = "base"

    def complete(self, system: str, user: str, json_mode: bool = False) -> LLMResponse:
        raise NotImplementedError


class AzureOpenAILLM(BaseLLM):
    """Thin wrapper over the Azure OpenAI chat completions API."""

    name = "azure_openai"

    def __init__(self) -> None:
        from openai import AzureOpenAI

        endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")
        self.deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")

        api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
        if api_key:
            self.client = AzureOpenAI(
                azure_endpoint=endpoint, api_key=api_key, api_version=api_version
            )
        else:
            from azure.identity import (
                DefaultAzureCredential,
                get_bearer_token_provider,
            )

            token_provider = get_bearer_token_provider(
                DefaultAzureCredential(),
                "https://cognitiveservices.azure.com/.default",
            )
            self.client = AzureOpenAI(
                azure_endpoint=endpoint,
                azure_ad_token_provider=token_provider,
                api_version=api_version,
            )

    def complete(self, system: str, user: str, json_mode: bool = False) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.deployment,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            # Extraction is not a creative task. Temperature 0 also makes the
            # eval harness reproducible enough to compare prompt versions.
            "temperature": 0,
            "max_tokens": 800,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = self.client.chat.completions.create(**kwargs)
        usage = resp.usage
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            model=self.deployment,
            usage={
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
            },
        )


class FakeLLM(BaseLLM):
    """Deterministic stub. Regex, not intelligence -- see the module docstring."""

    name = "fake"

    # Includes both canonical terms (what the generator writes) and common
    # synonyms, so the extraction layer's vocabulary mapping actually gets
    # exercised by the offline backend instead of only the happy path.
    SYMPTOM_LEXICON: ClassVar[list[str]] = [
        "bloody diarrhea", "bloody stools", "loose stools", "watery stools",
        "diarrhea", "abdominal cramps", "stomach cramps", "abdominal pain",
        "fever", "febrile", "nausea", "jaundice", "fatigue", "dark urine",
        "paroxysmal cough", "coughing fits", "whoop", "post-tussive vomiting",
        "cough", "myalgia", "muscle aches", "body aches", "sore throat",
        "loss of taste", "shortness of breath", "dyspnea",
    ]

    def complete(self, system: str, user: str, json_mode: bool = False) -> LLMResponse:
        if "EXTRACTION_TASK" in system:
            return LLMResponse(json.dumps(self._extract(user)), model="fake-extract")
        if "GROUNDED_ANSWER_TASK" in system:
            return LLMResponse(self._answer(user), model="fake-rag")
        return LLMResponse("{}", model="fake")

    def _extract(self, user: str) -> dict:
        note = user.lower()
        found: list[str] = []
        for term in self.SYMPTOM_LEXICON:
            # "diarrhea" is a substring of "bloody diarrhea"; keep the
            # longest match only, which is what the lexicon order buys us.
            if term in note and not any(term in f and term != f for f in found):
                found.append(term)

        onset = re.search(r"onset reported as (\d{4}-\d{2}-\d{2})", note)
        exposure = re.search(r"exposure history: ([^.]+)\.", note)
        travel = "reports recent travel" in note
        return {
            "symptoms": sorted(set(found)),
            "onset_date": onset.group(1) if onset else None,
            "exposure_setting": exposure.group(1).strip() if exposure else None,
            "recent_travel": travel,
            "confidence": "medium",
        }

    def _answer(self, user: str) -> str:
        """Echo the first retrieved passage with its citation, or refuse."""
        docs = re.findall(r"\[(doc-[\w-]+)\]\s*(.+)", user)
        if not docs:
            return "INSUFFICIENT_CONTEXT"
        doc_id, body = docs[0]
        first_sentence = body.split(". ")[0].strip()
        return f"{first_sentence}. [{doc_id}]"


def get_llm() -> BaseLLM:
    backend = get_settings().llm_backend
    if backend == "azure_openai":
        return AzureOpenAILLM()
    if backend == "fake":
        return FakeLLM()
    raise ValueError(f"unknown VS_LLM_BACKEND={backend!r}")
