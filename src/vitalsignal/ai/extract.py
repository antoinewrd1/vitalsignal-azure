"""Free-text clinical note -> structured, vocabulary-mapped surveillance fields.

This is the AI-engineering piece that actually earns its place in the pipeline:
onset date and exposure setting exist only in the NTE narrative, so no amount
of SQL recovers them. The engineering around the model matters more than the
prompt:

  1. SCHEMA FIRST. The model's output is parsed into a Pydantic model. Anything
     that does not validate is not "close enough" -- it is a failure with an
     error message we feed back on retry.
  2. BOUNDED RETRIES. Two retries, each one showing the model its own
     validation error. If it still fails we return a null-filled record with
     `extraction_failed=True` rather than guessing.
  3. CONTROLLED VOCABULARY. An LLM will happily emit "tummy ache". Surveillance
     needs a fixed term set. Unmapped terms are preserved in `unmapped_terms`
     for human review instead of being silently dropped -- that column is how
     you discover the vocabulary needs extending.
  4. NO SILENT HALLUCINATION OF DATES. An onset date that does not appear in
     the source text is rejected in post-validation, because a plausible-looking
     invented date is worse than a null.
"""

from __future__ import annotations

import json
import re
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from vitalsignal.ai.llm import BaseLLM, get_llm

MAX_ATTEMPTS = 3

# Canonical surveillance terms -> accepted synonyms the model might emit.
SYMPTOM_VOCABULARY: dict[str, set[str]] = {
    "diarrhea": {"diarrhea", "diarrhoea", "loose stools", "watery stools"},
    "bloody diarrhea": {"bloody diarrhea", "bloody stools", "hematochezia"},
    "abdominal cramps": {"abdominal cramps", "abdominal pain", "stomach cramps", "tummy ache"},
    "fever": {"fever", "febrile", "pyrexia", "elevated temperature"},
    "nausea": {"nausea", "queasiness"},
    "vomiting": {"vomiting", "emesis", "post-tussive vomiting"},
    "jaundice": {"jaundice", "icterus", "yellowing of skin"},
    "fatigue": {"fatigue", "tiredness", "malaise"},
    "dark urine": {"dark urine", "tea-colored urine"},
    "cough": {"cough", "coughing"},
    "paroxysmal cough": {"paroxysmal cough", "coughing fits", "whooping cough"},
    "whoop": {"whoop", "inspiratory whoop"},
    "myalgia": {"myalgia", "muscle aches", "body aches"},
    "sore throat": {"sore throat", "pharyngitis"},
    "loss of taste": {"loss of taste", "ageusia", "loss of taste or smell"},
    "shortness of breath": {"shortness of breath", "dyspnea", "difficulty breathing"},
}
_SYNONYM_INDEX = {
    syn: canon for canon, syns in SYMPTOM_VOCABULARY.items() for syn in syns
}

SYSTEM_PROMPT = """EXTRACTION_TASK
You are a public health data abstractor. Extract structured fields from a
clinical note that accompanies an electronic case report.

Rules:
- Return ONE JSON object and nothing else. No prose, no markdown fences.
- Use only information stated in the note. If a field is not stated, use null.
- Never infer or estimate a date that is not written in the note.
- symptoms: array of lowercase symptom phrases exactly as described.
- onset_date: ISO 8601 (YYYY-MM-DD) or null.
- exposure_setting: short phrase describing the stated exposure, or null.
- recent_travel: true only if travel outside the country is stated.
- confidence: "high" | "medium" | "low" -- your confidence in the extraction.

Schema:
{"symptoms": [str], "onset_date": str|null, "exposure_setting": str|null,
 "recent_travel": bool, "confidence": "high"|"medium"|"low"}"""


class ClinicalExtraction(BaseModel):
    symptoms: list[str] = Field(default_factory=list)
    onset_date: date | None = None
    exposure_setting: str | None = None
    recent_travel: bool = False
    confidence: Literal["high", "medium", "low"] = "medium"

    @field_validator("symptoms")
    @classmethod
    def _clean_symptoms(cls, v: list[str]) -> list[str]:
        return [s.strip().lower() for s in v if s and s.strip()]

    @field_validator("exposure_setting")
    @classmethod
    def _clean_exposure(cls, v: str | None) -> str | None:
        return v.strip().lower() if v and v.strip() else None


class ExtractionResult(BaseModel):
    """What the pipeline actually stores: mapped terms plus provenance."""

    symptoms_mapped: list[str] = Field(default_factory=list)
    unmapped_terms: list[str] = Field(default_factory=list)
    onset_date: date | None = None
    exposure_setting: str | None = None
    recent_travel: bool = False
    confidence: str = "medium"
    attempts: int = 0
    extraction_failed: bool = False
    failure_reason: str | None = None
    model: str = ""


def map_symptoms(raw: list[str]) -> tuple[list[str], list[str]]:
    """Map free text to the controlled vocabulary; keep the misses."""
    mapped, unmapped = set(), []
    for term in raw:
        t = term.strip().lower()
        if t in _SYNONYM_INDEX:
            mapped.add(_SYNONYM_INDEX[t])
            continue
        # Longest-match containment, so "severe bloody diarrhea" still maps.
        hit = None
        for syn, canon in sorted(_SYNONYM_INDEX.items(), key=lambda kv: -len(kv[0])):
            if syn in t:
                hit = canon
                break
        if hit:
            mapped.add(hit)
        else:
            unmapped.append(t)
    return sorted(mapped), unmapped


def _strip_fences(text: str) -> str:
    """Models add ```json fences even when told not to. Tolerate it."""
    return re.sub(r"^\s*```(?:json)?|```\s*$", "", text.strip()).strip()


def _date_is_grounded(extracted: date | None, note: str) -> bool:
    """Reject a date the model produced that does not appear in the source."""
    if extracted is None:
        return True
    return extracted.isoformat() in note


def extract_note(note: str, llm: BaseLLM | None = None) -> ExtractionResult:
    llm = llm or get_llm()
    user = f"Clinical note:\n{note}"
    last_error = ""

    for attempt in range(1, MAX_ATTEMPTS + 1):
        prompt = user if attempt == 1 else (
            f"{user}\n\nYour previous response was rejected: {last_error}\n"
            "Return corrected JSON only."
        )
        resp = llm.complete(SYSTEM_PROMPT, prompt, json_mode=True)
        try:
            payload = json.loads(_strip_fences(resp.text))
            parsed = ClinicalExtraction.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = str(exc)[:300]
            continue

        if not _date_is_grounded(parsed.onset_date, note):
            last_error = (
                f"onset_date {parsed.onset_date} does not appear in the note text"
            )
            continue

        mapped, unmapped = map_symptoms(parsed.symptoms)
        return ExtractionResult(
            symptoms_mapped=mapped,
            unmapped_terms=unmapped,
            onset_date=parsed.onset_date,
            exposure_setting=parsed.exposure_setting,
            recent_travel=parsed.recent_travel,
            confidence=parsed.confidence,
            attempts=attempt,
            model=resp.model,
        )

    return ExtractionResult(
        attempts=MAX_ATTEMPTS,
        extraction_failed=True,
        failure_reason=last_error,
        model=getattr(llm, "name", "unknown"),
    )


if __name__ == "__main__":
    demo = (
        "Patient is a 34-year-old female presenting with loose stools, stomach cramps "
        "and febrile illness. Symptom onset reported as 2026-03-04. "
        "Exposure history: church potluck. Reports recent travel to Mexico."
    )
    print(extract_note(demo).model_dump_json(indent=2))
