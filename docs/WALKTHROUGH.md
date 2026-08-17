# VitalSignal — file-by-file walkthrough

Read this top-to-bottom and you have read the project. Each section covers what
the file does, the design decisions inside it, and — where relevant — the bug
the build process actually hit there, because those are the best interview
stories in the repo.

---

## Foundation

### `src/vitalsignal/config.py`
The only file that knows whether we're local or on Azure. `Settings.uri(layer,
table)` returns either a local path under `_lake/` or an
`abfss://container@account.dfs.core.windows.net/layer/table` URI. Everything
downstream calls `uri()` and stays backend-agnostic — there is deliberately no
`if backend == "azure"` anywhere else in the codebase. `LAYERS` is a closed
tuple so a typo like `uri("slver", …)` raises instead of silently creating a
new directory. `get_settings()` is `lru_cache`d: one read of the environment
per process, consistent everywhere.

### `src/vitalsignal/spark.py`
SparkSession factory plus format-agnostic `read_table`/`write_table`. On
Databricks, `getOrCreate()` returns the cluster's existing session, so the
local-only settings (`master("local[*]")`, 2g driver) are guarded. Two configs
worth defending: `spark.sql.shuffle.partitions=4` (the default 200 produces 200
tiny files per write on a laptop) and `timeParserPolicy=CORRECTED` (reject
silently-wrong date parses; badness must surface as NULL + quarantine, not as
garbage data). `write_table` reads the format from settings so pipeline code
never hardcodes parquet vs delta — flipping `VS_TABLE_FORMAT=delta` on
Databricks changes the storage format without touching a single job.

---

## Data generation

### `src/vitalsignal/generate/synthetic_ecr.py`
Produces the entire universe: HL7 v2.5.1-*style* ORU^R01 messages
(MSH/PID/OBR/OBX/NTE) as NDJSON envelopes, partitioned by `ingest_date=`.

Decisions that make downstream work honest:

- **Poisson counts with structure.** Baseline rate per condition × facility
  volume × weekday factor (weekend reporting drops ~55%, like real feeds) ×
  outbreak multiplier. The weekday effect exists so the ML model has a
  seasonality confounder to learn around, not just a step function.
- **Injected outbreak windows are the ground truth.** Written to
  `_truth/outbreaks.json`. Every ML metric downstream is measured against a
  label this file invented — the README says so in three places because that is
  the ceiling on what the metrics mean.
- **Deliberate dirt.** ~4% retransmissions (same payload, later `received_at`)
  and ~2.5% corrupted messages (dropped OBX, garbage timestamp, truncation) so
  silver's dedupe and quarantine have real work.
- **The golden eval set is a free by-product.** The generator knows exactly
  which symptoms/onset/exposure it wrote into each NTE note, so it emits 120
  `(note, expected)` pairs to `data/golden/extraction_eval.jsonl`. Exact
  labels, no annotators.

---

## Data engineering

### `src/vitalsignal/ingest/land_to_bronze.py`
Landing → bronze. Three bronze rules enforced in code: never modify payloads;
add lineage only (`_batch_id`, `_ingest_ts`, `source_file` via
`_metadata.file_path`, `_payload_sha256`); be idempotent. Idempotency = a tiny
`_ingest_log` control table anti-joined against incoming file paths — the
zero-infrastructure equivalent of Databricks Auto Loader's checkpoint, and the
job comment says exactly that trade. An explicit `LANDING_SCHEMA` avoids
schema inference (an extra full data pass, plus silent type drift when a file's
shape changes). Verified behavior: the second run prints `bronze: nothing new
to ingest` and appends zero rows.

### `src/vitalsignal/transform/bronze_to_silver.py`
The biggest file, three jobs in order: **parse, quarantine, dedupe**.

- **Parsing is pure Spark SQL expressions** (`split`, `filter`,
  `try_element_at`), no Python UDFs — a UDF here would serialize every row to
  Python workers, ~10x cost on a real cluster. The helpers encode the HL7
  numbering trap: for MSH, field *n* is at split-index *n* (MSH-1 *is* the
  separator); for every other segment it's *n+1*. That off-by-one is asserted
  in `tests/test_hl7_parsing.py`.
- **Build story, part 1:** the first run crashed with `CANNOT_PARSE_TIMESTAMP`
  — Spark 4's ANSI mode makes `to_timestamp` *throw* on the generator's
  corrupted `00000000000000` timestamp, killing the whole job for one bad row.
  Fix: `try_to_timestamp`/`try_element_at`, which return NULL — and the DQ
  layer converts that NULL into a named quarantine reason. One malformed
  message must never take down a statewide feed.
- **Quarantine attaches reasons.** `apply_dq` builds an array of failed rule
  names per row (`missing_condition_code`, `report_date_in_future`, …). Clean
  rows have an empty array; failures go to `quarantine/ecr_rejects` *with the
  raw payload*, so they're replayable after a fix. Nothing is dropped.
- **Dedupe collapses on payload SHA-256**, not message-control-ID: a
  retransmission reuses both, a *correction* reuses neither, so hashing the
  payload collapses only true retransmissions. Window keeps first arrival and
  records `retransmission_count` — a data-quality signal per facility.

Run result: 27,798 bronze → 25,997 silver, 722 quarantined with reasons, 1,079
retransmissions collapsed.

---

## Analytics engineering (`dbt/vitalsignal/`)

### `dbt_project.yml` / `profiles.yml`
Two targets, one codebase: `dev` = DuckDB reading the silver Parquet in place;
`prod` = Databricks SQL Warehouse over Unity Catalog. Vars pin the surveillance
calendar (`spine_start`/`spine_end`) and the minimum baseline history.
Staging materializes as views, marts as tables.

### `macros/`
- `date_spine.sql`, `days_between.sql` — `adapter.dispatch` implementations for
  DuckDB (`generate_series`, `date_diff`) and Databricks (`sequence`/`explode`,
  `datediff`). This is the mechanism that lets one model file run on both
  engines; it also avoids a package-hub network dependency in CI.
- `surrogate_key.sql` — md5 over coalesced columns with a `_null_` sentinel so
  two rows differing only by NULL can't hash-collide.
- `test_accepted_range.sql` — hand-rolled generic range test (again: no
  dbt_utils network fetch in CI).

### `models/staging/stg_ecr_cases.sql`
Rename, cast, one shared filter (`where is_positive`). No joins, no
aggregation — staging that does more than this becomes a second silver layer
nobody governs.

### `models/marts/dim_facility.sql`
Facility dimension derived from the fact stream, with the operationally
important columns: `days_since_last_report` and `reporting_status`
(active/stale at 14 days). A facility gone quiet is a *reporting failure*, not
an absence of disease — surveillance teams page on this column.

### `models/marts/dim_condition.sql`
Condition dimension with `condition_group` (enteric/respiratory) defined once,
here, so every downstream report shares one grouping definition.

### `models/marts/fct_case_daily.sql`
The backbone. Grain: one row per (date × facility × condition). Two design
decisions carry the project:

1. **Zero-fill via cross join** of the date spine × both dimensions before
   left-joining counts. Without it, a quiet Tuesday has *no row*, and every
   rolling average downstream divides by the wrong denominator.
2. **Leakage-safe baseline:** the 28-day window is `rows between 28 preceding
   and 1 preceding` — the current day is excluded from its own baseline. If it
   weren't, a spike would damp its own z-score, and the ML model would train on
   a feature that has partially seen its label.

`baseline_z` is NULL (not 0) when history < 21 days or the series is flat:
"unknown" must never be encoded as "normal".

### `models/marts/agg_condition_daily_state.sql`
Statewide rollup at the grain a Power BI/Fabric report binds to, so the BI
layer never aggregates the fact table live.

### Tests (`_marts.yml`, `tests/*.sql`)
24 schema tests plus two singular tests. The two worth naming:
`unique` on `case_daily_sk` **is** the grain test (a duplicated key = broken
grain), and `assert_gold_reconciles_silver.sql` pins `sum(case_count)` in gold
to `count(*)` in silver over the spine — the exact failure mode of a botched
cross join is silent inflation, and this makes it loud. 29/29 pass.

---

## ML engineering

### `src/vitalsignal/ml/features.py`
Reads **the warehouse, not the lake** — the z-score and zero-fill are defined
once in dbt and tested there, so model and dashboard can't drift on what
"expected count" means. Leakage discipline: `lag_1`, `lag_7`,
`prior_mean_cases` (expanding mean of *shifted* values), `delta_1` — every
historical feature sees only days < t. `ratio_to_baseline` adds +0.5 to the
denominator to stay finite on true-zero baselines. Writes
`feature_spec.json` — the contract listing feature columns — which both
`train.py` and scoring import, so a feature added in one place can't silently
miss the other.

### `src/vitalsignal/ml/train.py`
Four defensible decisions:

1. **Temporal split** (last 75 days held out). Random splits on rolling
   features are leaks: Tuesday in train, Wednesday in test, and the 7-day sum
   straddles both.
2. **PR-AUC as the headline.** Positives are ~1.4%; ROC-AUC flatters at that
   base rate (ours is 0.993 — pretty, but PR-AUC 0.930 is the meaningful one).
3. **The baseline is the status-quo rule, not a dummy classifier.** Public
   health already runs "alert when z ≥ 3". The model only matters if it beats
   that *at equal alert volume*. It does: P 0.83/R 0.88 vs P 0.19/R 0.23 at ~8
   alerts/week — and the audit explains why (see below).
4. **Threshold chosen on train under an alert budget** (epidemiologists can
   investigate ~8 signals/week). Choosing it on test would leak and would
   produce an operationally meaningless number.

Also: `class_weight="balanced"` for the imbalance; NaNs (series with <28 days
history) flow into `HistGradientBoostingClassifier`, which learns a split
direction for "missing" — more honest than imputing a baseline that doesn't
exist. Permutation importance is computed **on the held-out set** (impurity
importances are computed on train and flatter noisy features) and logged to
MLflow. Everything — params, metrics, both baselines, the report JSON, the
model with an input-example signature — goes to MLflow, and the model is
registered by name.

**The audit story:** the first training run on low-volume data produced PR-AUC
0.024 with 7 test positives — a number that means nothing. Rather than tune
hyperparameters, the fix was the data (realistic surveillance volumes, 20
outbreak windows). And when the retrained model beat the z-rule by a suspicious
margin, the response was permutation importance + checking flagged rows against
truth — which surfaced the *real* mechanism: **a sustained outbreak
contaminates its own 28-day baseline**, so the z-rule goes blind mid-outbreak
while a trailing-7-day count doesn't. That is a genuine, citable weakness of
naive aberration detection, discovered by auditing a too-good result instead
of shipping it.

### `src/vitalsignal/ml/score_batch.py`
Loads the model **from the registry by name** (`models:/outbreak_signal_clf/N`)
— training and scoring are decoupled; promotion is a registry operation, not a
deploy. Reads the threshold from the training report so the alert budget the
model was tuned for is the one it serves. Writes `gold.fct_outbreak_signal`
with score, flag, threshold, model version, and scoring timestamp — full
provenance per row.

---

## AI engineering

### `src/vitalsignal/ai/llm.py`
One interface, two backends. `AzureOpenAILLM` uses **Entra ID by default**
(`DefaultAzureCredential` → bearer-token provider; a managed identity beats a
key in an app setting), key auth only as a fallback — and the Bicep sets
`disableLocalAuth: true` so infra and code agree. Temperature 0: extraction is
not a creative task, and determinism makes prompt versions comparable in the
eval harness. `FakeLLM` is a regex-driven stub whose docstring says exactly
what it is; it exists so CI runs with no network, no key, no cost — and any
number it produces is labeled as harness-validation, not model performance.

### `src/vitalsignal/ai/extract.py`
Free-text NTE note → structured surveillance fields. The engineering around
the model is the point:

1. **Schema-first:** output parses into a Pydantic model or it's a failure.
2. **Bounded retries with feedback:** up to 3 attempts; each retry shows the
   model its own validation error. Exhaustion returns a null-filled record
   with `extraction_failed=True` — fail closed, never guess.
3. **Controlled vocabulary:** free text maps to canonical surveillance terms
   via a synonym index with longest-match containment ("severe bloody
   diarrhea" → `bloody diarrhea`, not `diarrhea`). Unmapped terms are
   *preserved* in `unmapped_terms` — that column is how you discover the
   vocabulary needs extending.
4. **Groundedness check on dates:** an onset date not literally present in the
   note text is rejected. A plausible invented date is worse than a null.

**Build story, part 2:** the module's own `__main__` demo caught a bug on
first run — the demo note used synonyms ("loose stools", "febrile") the fake
backend's lexicon didn't contain, so extraction returned zero symptoms. The
fix (synonyms in the fake's lexicon) also made the offline harness exercise
the vocabulary-mapping path instead of only the happy path.

### `src/vitalsignal/ai/rag.py`
Grounded Q&A over an in-repo corpus of guidance snippets (written for this
project; explicitly *not* authoritative CDC/VDH text). Local backend is BM25
in ~40 lines; `AzureSearchIndex` is the same interface over Azure AI Search.

The contract is **cite-or-refuse, enforced in code**: a response without a
citation matching a *retrieved* doc ID is downgraded to
`INSUFFICIENT_CONTEXT` by post-processing — trust the check, not the model.
A retrieval-confidence floor refuses *before* the LLM is called when the best
BM25 score is weak, because weakly-related passages are exactly the context
that produces confident, wrong, cited-looking answers.

**Build story, part 3:** the eval harness failed its first run and drove three
retrieval fixes in sequence: (1) refusal accuracy was 0.0 — BM25 always
returns *something*, so the floor was added; (2) an exclusion question ranked
the investigation doc first — title terms now count 3×; (3) the ranking was
still wrong because "handler" ≠ "handlers" and stopwords swamped IDF — a
20-line tokenizer (stopword list + naive plural strip, with an `-ss` guard so
"illness" survives) fixed it. Each fix is a standard IR technique, and each is
commented with the production replacement (Azure AI Search's analyzer +
semantic ranker).

### `src/vitalsignal/ai/evals.py`
The file that makes the LLM code engineering instead of vibes. Two suites:
**extraction** over the 120 golden notes (symptom micro-P/R/F1 *after*
vocabulary mapping — canon compared to canon — plus onset/exposure/travel
exact-match and schema-failure rate) and **RAG** over hand-written Q&A pairs
that include questions the corpus genuinely cannot answer, because a system
that never refuses is broken in the dangerous direction. `GATES` defines
thresholds; any breach → exit 1 → CI fails. The report records which backend
produced it, with the caveat inline in the JSON.

---

## Tests (`tests/`)
21 tests, each pinning a promise a docstring makes:

- `test_hl7_parsing.py` — the MSH/PID off-by-one; missing OBX → *named*
  quarantine reasons; garbage timestamp → NULL not crash; MSH-only truncation
  survives parsing.
- `test_dedupe.py` — first arrival wins; retransmission count correct.
- `test_features_leakage.py` — a 100-case spike on the last day must not
  contaminate `prior_mean_cases`; first-row lag is NaN (*unknown*), not 0
  (*no cases*) — those are different facts.
- `test_train_utils.py` — the temporal split is a clean cut; the alert-budget
  threshold yields exactly budget × weeks alerts.
- `test_ai_extract.py` — a `ScriptedLLM` scripts failure→recovery (2
  attempts), hallucinated-date rejection, fail-closed exhaustion, and
  markdown-fence tolerance.
- `test_ai_rag.py` — tokenizer behavior (including the `-ss` guard), correct
  citation on-topic, refusal-before-LLM off-topic, and the uncited-answer
  downgrade via an `UncitedLLM` that answers confidently without citing.

## CI (`.github/workflows/ci.yml`)
Lint (ruff) → unit tests → **the entire pipeline end-to-end on a fresh
runner** with a smaller dataset, ending at the eval gates whose exit code fails
the build. `VS_LLM_BACKEND=fake` so CI needs no secrets and spends no tokens.
Reports upload as artifacts. A repo whose README promises a pipeline should
have CI that executes the pipeline.

## Infra & orchestration
- `infra/main.bicep` — ADLS Gen2 (HNS on), Databricks premium + **Access
  Connector** with Storage Blob Data Contributor (identity-based lake access;
  no storage keys anywhere), Azure ML workspace (+ Key Vault, App Insights),
  Azure OpenAI with `disableLocalAuth: true` **and the gpt-4o-mini deployment**
  (capacity is the part people forget), AI Search basic. Written and reviewed;
  not deployed from this repo — the README's verification table says so.
- `orchestration/databricks_job.json` — nightly Jobs 2.1 spec from Git source:
  bronze → silver → dbt (native `dbt_task` against a SQL Warehouse) → features
  → score, on a spot-with-fallback job cluster with `VS_BACKEND=azure` in the
  environment. The task graph mirrors the Makefile because the code is
  backend-agnostic.
- `azureml/train_job.yml` — weekly retraining as an Azure ML command job;
  `VS_MLFLOW_TRACKING_URI=azureml://` re-points tracking with zero code
  changes. Training lives in Azure ML (not Databricks) so the model is born
  where registry, experiments, and endpoints are governed; the seam between
  the platforms is the feature table in the lake.

## Known gaps (also in the README)
Local writes are Parquet, not Delta (config-switched, unverified here). The
BM25 floor is corpus-calibrated. All metrics rest on synthetic labels. The
fake backend's eval numbers validate the harness, not a model.
