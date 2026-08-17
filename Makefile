PY ?= python3
export VS_LOCAL_ROOT ?= $(CURDIR)/_lake
export PYTHONPATH := src

.PHONY: all clean generate bronze silver dbt features train score ai-eval test pipeline

all: pipeline

clean:
	rm -rf _lake _mlruns _out dbt/vitalsignal/target dbt/vitalsignal/logs

generate:      ## 1. synthetic eCR feed -> landing zone
	$(PY) -m vitalsignal.generate.synthetic_ecr --days 240 --facilities 12 --out _lake/landing

bronze:        ## 2. landing -> bronze (raw, append-only, with lineage columns)
	$(PY) -m vitalsignal.ingest.land_to_bronze

silver:        ## 3. bronze -> silver (HL7 parse, conform, dedupe, quarantine)
	$(PY) -m vitalsignal.transform.bronze_to_silver

dbt:           ## 4. silver -> gold (analytics engineering)
	mkdir -p $(VS_LOCAL_ROOT)/gold   # DuckDB will not create the parent directory
	cd dbt/vitalsignal && dbt build --target dev

features:      ## 5. gold -> ML feature table
	$(PY) -m vitalsignal.ml.features

train:         ## 6. train + log to MLflow / Azure ML
	$(PY) -m vitalsignal.ml.train

score:         ## 7. batch inference -> gold signal table
	$(PY) -m vitalsignal.ml.score_batch

ai-index:      ## 8. build the retrieval index for RAG
	$(PY) -m vitalsignal.ai.rag --build-index

ai-eval:       ## 9. run the LLM eval harness (gates CI)
	$(PY) -m vitalsignal.ai.evals

test:
	$(PY) -m pytest -q

pipeline: generate bronze silver dbt features train score ai-index ai-eval
	@echo "pipeline complete"
