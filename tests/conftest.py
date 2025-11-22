import pytest
from copy import deepcopy
from core.loaders.dataset_config import DATASET_REGISTRY, DatasetConfig
from core.loaders.dataset_ids import DatasetId


@pytest.fixture
def patched_registry(monkeypatch):
    # create deep-clone of the frozen configs
    cloned = {}

    for k, v in DATASET_REGISTRY.items():
        cloned[k] = DatasetConfig(
            dataset_id=v.dataset_id,
            relative_path=v.relative_path,
            parse_dates=list(v.parse_dates),
            rename_columns=dict(v.rename_columns),
            required_columns=list(v.required_columns),
            dtypes=None if v.dtypes is None else dict(v.dtypes),
            delimiter=v.delimiter,
            has_header=v.has_header,
            encoding=v.encoding,
            null_values=None if v.null_values is None else list(v.null_values),
        )

    monkeypatch.setattr("core.loaders.dataset_config.DATASET_REGISTRY", cloned)
    return cloned
