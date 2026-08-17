"""Eval harness for the AI layer. This file is what makes the LLM code
engineering rather than vibes: it produces numbers, and CI fails if the
numbers drop below the gates.

Two suites:

  1. EXTRACTION -- runs extract_note() over the golden set the data generator
     wrote (data/golden/extraction_eval.jsonl: note text + the ground-truth
     fields the generator embedded in it). Because the generator *knows* what
     it put in each note, these labels are exact, not annotator guesses.
     Metrics: symptom micro-P/R/F1 after vocabulary mapping, onset-date exact
     match, exposure exact match, travel accuracy, schema-failure rate.

  2. RAG -- a hand-written set of question/expectation pairs, including
     questions the corpus genuinely cannot answer. For answerable questions we
     check the right document is cited; for unanswerable ones we check the
     system refuses. Refusal quality is measured, not assumed: a RAG system
     that never says INSUFFICIENT_CONTEXT is broken in the dangerous direction.

The report records which LLM backend produced it. Numbers from the `fake`
backend validate the harness and the vocabulary mapping; only numbers from
`azure_openai` say anything about a real model. The gates below are set for
the fake backend so CI is deterministic; tighten them per-model in Azure.
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from vitalsignal.ai.extract import extract_note, map_symptoms
from vitalsignal.ai.llm import get_llm
from vitalsignal.ai.rag import REFUSAL, answer, get_index

GOLDEN_PATH = Path("data/golden/extraction_eval.jsonl")

# CI gates. Chosen against the deterministic fake backend; a regression in the
# harness, the vocabulary, or the prompt contract trips them.
GATES = {
    "extraction.symptom_f1": 0.90,
    "extraction.onset_exact": 0.95,
    "extraction.schema_failure_rate_max": 0.02,
    "rag.citation_accuracy": 0.80,
    "rag.refusal_accuracy": 1.00,
}

RAG_CASES = [
    {
        "q": "What is the exclusion guidance for a food handler with salmonellosis?",
        "expect_doc": "doc-salm-exclusion",
        "answerable": True,
    },
    {
        "q": "When must hepatitis A post-exposure prophylaxis be given?",
        "expect_doc": "doc-hepa-ppx",
        "answerable": True,
    },
    {
        "q": "How long is a pertussis case excluded from school if untreated?",
        "expect_doc": "doc-pertussis-ppx",
        "answerable": True,
    },
    {
        "q": "When should a long-term care facility report an influenza outbreak?",
        "expect_doc": "doc-flu-surge",
        "answerable": True,
    },
    {
        "q": "What serial testing cadence is recommended for COVID-19 clusters?",
        "expect_doc": "doc-covid-cluster",
        "answerable": True,
    },
    {
        "q": "What is the recommended rabies post-exposure schedule?",
        "expect_doc": None,
        "answerable": False,  # nothing about rabies exists in the corpus
    },
    {
        "q": "What is the measles vaccination exclusion period for schools?",
        "expect_doc": None,
        "answerable": False,
    },
]


# ---------------------------------------------------------------------------
# Extraction suite
# ---------------------------------------------------------------------------
def eval_extraction(llm) -> dict:
    if not GOLDEN_PATH.exists():
        raise FileNotFoundError(f"{GOLDEN_PATH} missing -- run `make generate` first")

    cases = [json.loads(line) for line in GOLDEN_PATH.read_text().splitlines() if line]

    tp = fp = fn = 0
    onset_hits = exposure_hits = travel_hits = 0
    failures = 0
    worst: list[dict] = []

    for c in cases:
        result = extract_note(c["note"], llm=llm)
        if result.extraction_failed:
            failures += 1
            continue

        # The generator's labels are raw terms; map them through the same
        # vocabulary so we compare canon-to-canon, not string-to-string.
        expected, _ = map_symptoms(c["expected"]["symptoms"])
        got = set(result.symptoms_mapped)
        exp = set(expected)
        tp += len(got & exp)
        fp += len(got - exp)
        fn += len(exp - got)

        exp_onset = date.fromisoformat(c["expected"]["onset_date"])
        onset_ok = result.onset_date == exp_onset
        onset_hits += onset_ok

        exp_exposure = c["expected"]["exposure_setting"].lower()
        exposure_ok = (result.exposure_setting or "") == exp_exposure
        exposure_hits += exposure_ok

        travel_hits += result.recent_travel == c["expected"]["recent_travel"]

        if not (onset_ok and exposure_ok and got == exp) and len(worst) < 5:
            worst.append(
                {
                    "id": c["message_control_id"],
                    "missing_symptoms": sorted(exp - got),
                    "spurious_symptoms": sorted(got - exp),
                    "onset_ok": onset_ok,
                    "exposure_ok": exposure_ok,
                }
            )

    n = len(cases)
    scored = n - failures
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "n_cases": n,
        "schema_failure_rate": failures / n if n else 1.0,
        "symptom_precision": round(precision, 4),
        "symptom_recall": round(recall, 4),
        "symptom_f1": round(f1, 4),
        "onset_exact": round(onset_hits / scored, 4) if scored else 0.0,
        "exposure_exact": round(exposure_hits / scored, 4) if scored else 0.0,
        "travel_accuracy": round(travel_hits / scored, 4) if scored else 0.0,
        "worst_cases": worst,
    }


# ---------------------------------------------------------------------------
# RAG suite
# ---------------------------------------------------------------------------
def eval_rag(llm) -> dict:
    index = get_index()
    cite_hits = 0
    refusal_hits = 0
    n_answerable = sum(1 for c in RAG_CASES if c["answerable"])
    n_unanswerable = len(RAG_CASES) - n_answerable
    details = []

    for c in RAG_CASES:
        out = answer(c["q"], index=index, llm=llm)
        if c["answerable"]:
            ok = c["expect_doc"] in out["citations"]
            cite_hits += ok
        else:
            ok = out["answer"] == REFUSAL
            refusal_hits += ok
        details.append(
            {"q": c["q"], "ok": bool(ok), "citations": out["citations"],
             "refused": out["answer"] == REFUSAL}
        )

    return {
        "n_cases": len(RAG_CASES),
        "citation_accuracy": round(cite_hits / n_answerable, 4),
        "refusal_accuracy": round(refusal_hits / n_unanswerable, 4),
        "details": details,
    }


# ---------------------------------------------------------------------------
# Gate check + report
# ---------------------------------------------------------------------------
def check_gates(ext: dict, rag: dict) -> list[str]:
    breaches = []
    if ext["symptom_f1"] < GATES["extraction.symptom_f1"]:
        breaches.append(
            f"symptom_f1 {ext['symptom_f1']} < {GATES['extraction.symptom_f1']}"
        )
    if ext["onset_exact"] < GATES["extraction.onset_exact"]:
        breaches.append(f"onset_exact {ext['onset_exact']} < {GATES['extraction.onset_exact']}")
    if ext["schema_failure_rate"] > GATES["extraction.schema_failure_rate_max"]:
        breaches.append(
            f"schema_failure_rate {ext['schema_failure_rate']} > "
            f"{GATES['extraction.schema_failure_rate_max']}"
        )
    if rag["citation_accuracy"] < GATES["rag.citation_accuracy"]:
        breaches.append(
            f"citation_accuracy {rag['citation_accuracy']} < {GATES['rag.citation_accuracy']}"
        )
    if rag["refusal_accuracy"] < GATES["rag.refusal_accuracy"]:
        breaches.append(
            f"refusal_accuracy {rag['refusal_accuracy']} < {GATES['rag.refusal_accuracy']}"
        )
    return breaches


def run() -> int:
    llm = get_llm()
    ext = eval_extraction(llm)
    rag = eval_rag(llm)
    breaches = check_gates(ext, rag)

    report = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "llm_backend": getattr(llm, "name", "unknown"),
        "caveat_if_fake": (
            "Backend 'fake' is deterministic; these numbers validate the "
            "harness and vocabulary mapping, not a model."
        ),
        "gates": GATES,
        "gate_breaches": breaches,
        "extraction": ext,
        "rag": rag,
    }
    out = Path("_out")
    out.mkdir(exist_ok=True)
    (out / "ai_eval_report.json").write_text(json.dumps(report, indent=2))

    print(
        f"ai-eval [{report['llm_backend']}] over {ext['n_cases']} golden notes:\n"
        f"  symptoms   P {ext['symptom_precision']}  R {ext['symptom_recall']}  "
        f"F1 {ext['symptom_f1']}\n"
        f"  onset {ext['onset_exact']}  exposure {ext['exposure_exact']}  "
        f"travel {ext['travel_accuracy']}  schema-fail {ext['schema_failure_rate']}\n"
        f"  rag: citation {rag['citation_accuracy']}  refusal {rag['refusal_accuracy']}\n"
        f"  report -> _out/ai_eval_report.json"
    )
    if breaches:
        print("GATE FAILURES:")
        for b in breaches:
            print(f"  - {b}")
        return 1
    print("  all gates passed")
    return 0


if __name__ == "__main__":
    sys.exit(run())
