import pytest
import polars as pl
from core.interfaces.IAnomalyDetector import IAnomalyDetector

def test_interface_is_abstract():
    with pytest.raises(TypeError):
        IAnomalyDetector()  # cannot instantiate abstract class

def test_interface_methods_exist():
    class Dummy(IAnomalyDetector):
        def fit(self, df, feature_columns):
            return None
        def score(self, df):
            return df
        def detect(self, df, threshold):
            return df

    dummy = Dummy()
    df = pl.DataFrame({"x":[1]})
    assert isinstance(dummy.fit(df, ["x"]), type(None))
    assert isinstance(dummy.score(df), pl.DataFrame)
    assert isinstance(dummy.detect(df, 1.0), pl.DataFrame)
