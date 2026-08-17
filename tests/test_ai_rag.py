"""rag.py's contract: cite-or-refuse is enforced by code, not requested nicely."""

from vitalsignal.ai.llm import FakeLLM, LLMResponse
from vitalsignal.ai.rag import GUIDANCE_DOCS, Bm25Index, _tokenize, answer


class UncitedLLM:
    name = "uncited"

    def complete(self, system, user, json_mode=False):
        return LLMResponse("Exclude the worker until symptoms resolve.", model="uncited")


def _index():
    return Bm25Index.build(GUIDANCE_DOCS)


def test_tokenizer_strips_stopwords_and_plurals():
    assert _tokenize("the food handlers") == ["food", "handler"]
    assert "illness" in _tokenize("illness")  # -ss guard: not stripped to 'illnes'


def test_on_topic_question_cites_expected_doc():
    out = answer(
        "What is the exclusion guidance for a food handler with salmonellosis?",
        index=_index(), llm=FakeLLM(),
    )
    assert out["grounded"]
    assert "doc-salm-exclusion" in out["citations"]


def test_off_topic_question_refuses_before_llm():
    out = answer("What is the recommended rabies post-exposure schedule?",
                 index=_index(), llm=FakeLLM())
    assert out["answer"] == "INSUFFICIENT_CONTEXT"
    assert out["note"] == "retrieval below confidence floor"


def test_answer_without_citation_is_downgraded():
    out = answer(
        "What is the exclusion guidance for a food handler with salmonellosis?",
        index=_index(), llm=UncitedLLM(),
    )
    assert out["answer"] == "INSUFFICIENT_CONTEXT"
    assert not out["grounded"]
    assert "downgraded" in out["note"]
