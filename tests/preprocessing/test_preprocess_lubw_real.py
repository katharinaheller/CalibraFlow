import polars as pl
from core.preprocessing.preprocessing_orchestrator import PreprocessingOrchestrator
from core.preprocessing.preprocessing_config import PREPROCESSOR_REGISTRY
from core.loaders.dataset_ids import DatasetId

def test_lubw_real_timestamp_parsing():
    # minimal repräsentativer Ausschnitt
    df = pl.DataFrame({
        "datetime": ["2024-11-14 00:01:00"],
        "NO2": [24.445],
        "O3": [3.691],
        "PM10": [23.545],
        "PM2p5": [22.218],
        "TEMP": [4.951],
        "RLF": [92.67],
        "p-Luft": [1012.0],
        "NSCH": [0.0],
        "WIR": [176.868],
        "WIV": [0.9],
    })

    orch = PreprocessingOrchestrator(PREPROCESSOR_REGISTRY)
    out = orch.preprocess(DatasetId.LUBW_MINUTE, df)

    assert "timestamp" in out.columns          # timestamp existiert
    assert out["timestamp"].dtype == pl.Datetime  # Typ korrekt
    assert out.height == 1
