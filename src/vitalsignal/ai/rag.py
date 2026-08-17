"""Retrieval-augmented answering over public health guidance documents.

The use case: an epidemiologist gets an outbreak signal from the ML layer and
asks "what's the exclusion guidance for a food handler with Salmonella?" The
answer must come from the guidance corpus with a citation, or be an explicit
refusal -- an uncited answer in public health is a liability, not a feature.

Architecture notes:

  * Two retrieval backends behind one interface:
      local        -- BM25 implemented here in ~40 lines over a corpus of
                      guidance snippets shipped in-repo. Zero services, fully
                      testable in CI.
      azure_search -- Azure AI Search (hybrid keyword + vector in production;
                      this client uses the simple search API so it works on the
                      free tier).
  * The GUIDANCE CORPUS IS SYNTHETIC-BUT-REALISTIC. Each document paraphrases
    the kind of guidance a state health department publishes. They are written
    for this project and are not authoritative CDC/VDH text -- the README says
    so, and so does this docstring, because a repo that presents invented
    clinical guidance as real would be worse than no repo.
  * The answer contract is enforced, not requested: a response that lacks a
    [doc-id] citation matching a retrieved document is downgraded to
    INSUFFICIENT_CONTEXT by post-processing. Trust the check, not the model.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from vitalsignal.ai.llm import get_llm
from vitalsignal.config import get_settings

INDEX_DIR = "rag_index"

# ---------------------------------------------------------------------------
# Corpus. Written for this project; paraphrases real guidance *patterns*.
# ---------------------------------------------------------------------------
GUIDANCE_DOCS: list[dict] = [
    {
        "id": "doc-salm-exclusion",
        "title": "Salmonellosis: work and daycare exclusion",
        "condition": "Salmonellosis",
        "text": (
            "Food handlers diagnosed with salmonellosis should be excluded from work "
            "until diarrhea has resolved. Two consecutive negative stool cultures "
            "collected at least 24 hours apart are recommended before returning to "
            "sensitive occupations, including food handling and direct patient care. "
            "Children in daycare may return 24 hours after symptoms resolve unless "
            "local policy requires culture clearance."
        ),
    },
    {
        "id": "doc-salm-investigation",
        "title": "Salmonellosis: outbreak investigation steps",
        "condition": "Salmonellosis",
        "text": (
            "When two or more culture-confirmed cases share a common exposure, open "
            "an outbreak investigation. Collect a food history covering the 72 hours "
            "before symptom onset for each case. Request serotyping on all isolates "
            "and coordinate with the state lab for whole genome sequencing. Notify "
            "environmental health if a licensed food establishment is implicated."
        ),
    },
    {
        "id": "doc-shig-exclusion",
        "title": "Shigellosis: exclusion and clearance",
        "condition": "Shigellosis",
        "text": (
            "Shigella has a very low infectious dose. Exclude symptomatic staff and "
            "children from food handling, healthcare, and daycare settings. Return "
            "requires resolution of diarrhea and, for sensitive occupations, negative "
            "stool cultures per jurisdiction policy. Emphasize handwashing "
            "supervision for children under five."
        ),
    },
    {
        "id": "doc-hepa-ppx",
        "title": "Hepatitis A: post-exposure prophylaxis window",
        "condition": "Hepatitis A",
        "text": (
            "Post-exposure prophylaxis with hepatitis A vaccine or immune globulin "
            "should be administered within 14 days of last exposure. Healthy persons "
            "aged 1 to 40 years should receive single-antigen hepatitis A vaccine. "
            "Persons outside that range or immunocompromised may receive immune "
            "globulin. Identify close contacts and unvaccinated food handling "
            "coworkers for prophylaxis."
        ),
    },
    {
        "id": "doc-pertussis-ppx",
        "title": "Pertussis: contact management",
        "condition": "Pertussis",
        "text": (
            "Provide antibiotic post-exposure prophylaxis to household contacts and "
            "high-risk contacts, including infants, pregnant women in the third "
            "trimester, and those who will contact infants. Exclude the case from "
            "school or work until 5 days of appropriate antibiotic therapy are "
            "completed, or 21 days from cough onset if untreated."
        ),
    },
    {
        "id": "doc-flu-surge",
        "title": "Influenza: facility surge reporting",
        "condition": "Influenza",
        "text": (
            "Long-term care facilities should report influenza outbreaks when two or "
            "more residents develop influenza-like illness within 72 hours. Initiate "
            "antiviral treatment for cases and chemoprophylaxis for exposed residents "
            "regardless of vaccination status. Continue prophylaxis for at least 14 "
            "days and until 7 days after the last laboratory-confirmed case."
        ),
    },
    {
        "id": "doc-covid-cluster",
        "title": "COVID-19: cluster response in congregate settings",
        "condition": "COVID-19",
        "text": (
            "For clusters in congregate settings, recommend serial testing of "
            "residents and staff every 3 to 7 days until no new cases are identified "
            "over 14 days. Reinforce ventilation assessment and source control. "
            "Report clusters meeting jurisdictional thresholds to the health "
            "department within 24 hours."
        ),
    },
    {
        "id": "doc-general-reporting",
        "title": "General: reportable condition timelines",
        "condition": "General",
        "text": (
            "Rapidly communicable or high-consequence conditions must be reported "
            "immediately by phone. Most enteric and vaccine-preventable conditions "
            "are reportable within 3 business days of diagnosis. Electronic case "
            "reports satisfy the reporting obligation when the receiving system "
            "acknowledges the message."
        ),
    },
]

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Tiny English stopword list + naive plural stripping. Not Porter stemming --
# just enough that "handler" matches "handlers" and glue words stop swamping
# IDF. Azure AI Search's language analyzer does this properly in production.
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have if in into is it its of on or "
    "should that the this to was were what when where which who will with".split()
)


def _tokenize(text: str) -> list[str]:
    out = []
    for tok in _TOKEN_RE.findall(text.lower()):
        if tok in _STOPWORDS:
            continue
        if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        out.append(tok)
    return out


# ---------------------------------------------------------------------------
# Local BM25 backend
# ---------------------------------------------------------------------------
@dataclass
class Bm25Index:
    docs: list[dict]
    doc_freq: Counter
    doc_tokens: list[list[str]]
    avg_len: float
    k1: float = 1.5
    b: float = 0.75

    TITLE_BOOST = 3  # title terms count 3x -- titles are dense with intent

    @classmethod
    def build(cls, docs: list[dict]) -> Bm25Index:
        doc_tokens = [
            _tokenize(d["title"]) * cls.TITLE_BOOST + _tokenize(d["text"])
            for d in docs
        ]
        df: Counter = Counter()
        for toks in doc_tokens:
            df.update(set(toks))
        avg = sum(len(t) for t in doc_tokens) / max(len(doc_tokens), 1)
        return cls(docs=docs, doc_freq=df, doc_tokens=doc_tokens, avg_len=avg)

    def search(self, query: str, k: int = 3) -> list[tuple[float, dict]]:
        q = _tokenize(query)
        n = len(self.docs)
        scored = []
        for toks, doc in zip(self.doc_tokens, self.docs, strict=True):
            tf = Counter(toks)
            score = 0.0
            for term in q:
                if term not in tf:
                    continue
                idf = math.log(1 + (n - self.doc_freq[term] + 0.5) / (self.doc_freq[term] + 0.5))
                denom = tf[term] + self.k1 * (1 - self.b + self.b * len(toks) / self.avg_len)
                score += idf * tf[term] * (self.k1 + 1) / denom
            if score > 0:
                scored.append((score, doc))
        return sorted(scored, key=lambda t: -t[0])[:k]

    # Persistence keeps `--build-index` honest as a pipeline step even though
    # rebuilding from the in-repo corpus takes milliseconds.
    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "corpus.json").write_text(json.dumps(self.docs, indent=2))

    @classmethod
    def load(cls, path: Path) -> Bm25Index:
        docs = json.loads((path / "corpus.json").read_text())
        return cls.build(docs)


# ---------------------------------------------------------------------------
# Azure AI Search backend (same interface, real service)
# ---------------------------------------------------------------------------
class AzureSearchIndex:
    def __init__(self) -> None:
        import os

        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents import SearchClient

        self.client = SearchClient(
            endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
            index_name=os.environ.get("AZURE_SEARCH_INDEX", "phguidance"),
            credential=AzureKeyCredential(os.environ["AZURE_SEARCH_API_KEY"]),
        )

    def search(self, query: str, k: int = 3) -> list[tuple[float, dict]]:
        results = self.client.search(search_text=query, top=k)
        return [
            (r["@search.score"], {"id": r["id"], "title": r["title"], "text": r["text"]})
            for r in results
        ]


def get_index():
    s = get_settings()
    if s.search_backend == "azure_search":
        return AzureSearchIndex()
    idx_path = Path(s.uri("gold", INDEX_DIR))
    if (idx_path / "corpus.json").exists():
        return Bm25Index.load(idx_path)
    return Bm25Index.build(GUIDANCE_DOCS)


# ---------------------------------------------------------------------------
# Grounded answering
# ---------------------------------------------------------------------------
ANSWER_SYSTEM = """GROUNDED_ANSWER_TASK
You answer public health practice questions for an epidemiologist, using ONLY
the provided guidance passages.

Rules:
- Every factual sentence must end with a citation like [doc-id] naming the
  passage it came from.
- If the passages do not contain the answer, reply with exactly:
  INSUFFICIENT_CONTEXT
- Do not use outside knowledge. Do not soften a refusal with a partial guess.
- Keep the answer under 120 words."""

REFUSAL = "INSUFFICIENT_CONTEXT"
_CITE_RE = re.compile(r"\[(doc-[\w-]+)\]")

# Retrieval-confidence floor: if the best BM25 hit scores below this, refuse
# before the LLM is even called -- weakly-related passages are exactly the
# context that produces confident, wrong, cited-looking answers. The value is
# calibrated on this repo's eval set (on-topic questions score ~8+, off-topic
# ~3), so treat it as a smoke gate for THIS corpus, not a universal constant.
# Azure AI Search's hybrid semantic ranker replaces this heuristic in prod.
MIN_RETRIEVAL_SCORE = 5.0


def answer(question: str, k: int = 3, index=None, llm=None) -> dict:
    index = index or get_index()
    llm = llm or get_llm()

    hits = index.search(question, k=k)
    floor = getattr(index, "min_score", MIN_RETRIEVAL_SCORE)
    if not hits or hits[0][0] < floor:
        return {
            "answer": REFUSAL,
            "citations": [],
            "retrieved": sorted(d["id"] for _, d in hits),
            "grounded": False,
            "note": "retrieval below confidence floor",
        }

    context = "\n\n".join(f"[{d['id']}] {d['text']}" for _, d in hits)
    resp = llm.complete(ANSWER_SYSTEM, f"Passages:\n{context}\n\nQuestion: {question}")

    retrieved_ids = {d["id"] for _, d in hits}
    cited = _CITE_RE.findall(resp.text)
    valid_citations = [c for c in cited if c in retrieved_ids]

    # Enforcement, not politeness: an answer with no valid citation is treated
    # as ungrounded regardless of how confident it sounds.
    if resp.text.strip() != REFUSAL and not valid_citations:
        return {
            "answer": REFUSAL,
            "citations": [],
            "retrieved": sorted(retrieved_ids),
            "grounded": False,
            "note": "model answered without a valid citation; downgraded",
        }

    return {
        "answer": resp.text.strip(),
        "citations": valid_citations,
        "retrieved": sorted(retrieved_ids),
        "grounded": resp.text.strip() != REFUSAL,
        "model": resp.model,
    }


def build_index() -> None:
    s = get_settings()
    idx = Bm25Index.build(GUIDANCE_DOCS)
    out = Path(s.uri("gold", INDEX_DIR))
    idx.save(out)
    print(f"rag: indexed {len(GUIDANCE_DOCS)} guidance docs -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-index", action="store_true")
    ap.add_argument("--ask", default=None)
    args = ap.parse_args()
    if args.build_index:
        build_index()
    if args.ask:
        print(json.dumps(answer(args.ask), indent=2))
    if not args.build_index and not args.ask:
        ap.print_help()
