"""extract.py's contract: validate-or-retry, ground dates, map vocabulary."""

import json

from vitalsignal.ai.extract import ExtractionResult, extract_note, map_symptoms
from vitalsignal.ai.llm import LLMResponse


class ScriptedLLM:
    """Returns queued responses; lets tests script failure -> recovery."""

    name = "scripted"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def complete(self, system, user, json_mode=False):
        self.calls += 1
        return LLMResponse(self.responses.pop(0), model="scripted")


NOTE = (
    "Patient presenting with fever and cough. Symptom onset reported as "
    "2026-02-10. Exposure history: daycare center. Denies recent international travel."
)


def test_vocabulary_maps_synonyms_and_keeps_misses():
    mapped, unmapped = map_symptoms(["loose stools", "febrile", "glowing aura"])
    assert mapped == ["diarrhea", "fever"]
    assert unmapped == ["glowing aura"]  # preserved for vocab review, not dropped


def test_longest_match_wins():
    mapped, _ = map_symptoms(["severe bloody diarrhea"])
    assert mapped == ["bloody diarrhea"]  # not plain "diarrhea"


def test_invalid_json_then_recovery_uses_two_attempts():
    good = json.dumps({"symptoms": ["fever"], "onset_date": "2026-02-10",
                       "exposure_setting": "daycare center",
                       "recent_travel": False, "confidence": "high"})
    llm = ScriptedLLM(["not json at all", good])
    result = extract_note(NOTE, llm=llm)
    assert not result.extraction_failed
    assert result.attempts == 2


def test_hallucinated_onset_date_is_rejected():
    fabricated = json.dumps({"symptoms": ["fever"], "onset_date": "2025-12-25",
                             "exposure_setting": None, "recent_travel": False,
                             "confidence": "high"})
    llm = ScriptedLLM([fabricated] * 3)
    result = extract_note(NOTE, llm=llm)  # 2025-12-25 appears nowhere in NOTE
    assert result.extraction_failed
    assert "does not appear" in result.failure_reason


def test_exhausted_retries_fail_closed():
    llm = ScriptedLLM(["{bad", "{worse", "{worst"])
    result = extract_note(NOTE, llm=llm)
    assert isinstance(result, ExtractionResult)
    assert result.extraction_failed and result.attempts == 3
    assert result.symptoms_mapped == []  # null-filled, never guessed


def test_markdown_fences_are_tolerated():
    fenced = "```json\n" + json.dumps(
        {"symptoms": [], "onset_date": None, "exposure_setting": None,
         "recent_travel": False, "confidence": "low"}) + "\n```"
    result = extract_note(NOTE, llm=ScriptedLLM([fenced]))
    assert not result.extraction_failed
