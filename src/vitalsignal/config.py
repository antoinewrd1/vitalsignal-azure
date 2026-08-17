"""Single source of truth for where data lives and which backend we talk to.

The whole point of this module is that *no other file* contains an
`if backend == "azure"` branch. Paths, table format and service clients are
resolved here, so the transformation logic is identical whether it runs on a
laptop or on an Azure Databricks job cluster.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# Medallion layers, in order. Used for path building and for validation.
LAYERS = ("landing", "bronze", "silver", "quarantine", "gold", "feature", "ml")


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _env_path(key: str, default: str) -> Path:
    return Path(_env(key, default)).expanduser().resolve()


@dataclass(frozen=True)
class Settings:
    backend: str            # "local" | "azure"
    local_root: Path        # lakehouse root when backend == "local"
    storage_account: str    # ADLS Gen2 account when backend == "azure"
    container: str          # ADLS Gen2 filesystem (container)
    table_format: str       # "parquet" | "delta"
    mlflow_tracking_uri: str
    model_name: str
    llm_backend: str        # "fake" | "azure_openai"
    search_backend: str     # "local" | "azure_search"

    # ---- path resolution -------------------------------------------------
    def uri(self, layer: str, table: str) -> str:
        """Return the storage URI for a table in a medallion layer.

        local  -> /abs/path/_lake/silver/ecr_cases
        azure  -> abfss://lakehouse@stvitalsignal.dfs.core.windows.net/silver/ecr_cases
        """
        if layer not in LAYERS:
            raise ValueError(f"unknown layer {layer!r}; expected one of {LAYERS}")
        if self.backend == "azure":
            return (
                f"abfss://{self.container}@{self.storage_account}"
                f".dfs.core.windows.net/{layer}/{table}"
            )
        p = self.local_root / layer / table
        p.parent.mkdir(parents=True, exist_ok=True)
        return str(p)

    @property
    def is_azure(self) -> bool:
        return self.backend == "azure"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    backend = _env("VS_BACKEND", "local").lower()
    if backend not in ("local", "azure"):
        raise ValueError(f"VS_BACKEND must be 'local' or 'azure', got {backend!r}")

    # Delta is the format on Databricks. Locally we fall back to Parquet so the
    # repo runs with zero extra JARs -- see docs/WALKTHROUGH.md ("Known gaps").
    default_fmt = "delta" if backend == "azure" else "parquet"

    return Settings(
        backend=backend,
        local_root=_env_path("VS_LOCAL_ROOT", "./_lake"),
        storage_account=_env("VS_STORAGE_ACCOUNT", "stvitalsignal"),
        container=_env("VS_CONTAINER", "lakehouse"),
        table_format=_env("VS_TABLE_FORMAT", default_fmt).lower(),
        mlflow_tracking_uri=_env("VS_MLFLOW_TRACKING_URI", "./_mlruns"),
        model_name=_env("VS_MODEL_NAME", "outbreak_signal_clf"),
        llm_backend=_env("VS_LLM_BACKEND", "fake").lower(),
        search_backend=_env("VS_SEARCH_BACKEND", "local").lower(),
    )
