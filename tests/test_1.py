import pytest
import pandas as pd
from definition_e6ef71ea7f554c459deff91f37596af5 import generate_synthetic_ild

@pytest.fixture
def mock_frequency_params():
    return {'distribution': 'poisson', 'lambda': 5}

@pytest.fixture
def mock_severity_params():
    return {'distribution': 'lognorm', 'mean': 8, 'sigma': 1.5}


def test_generate_synthetic_ild_positive_observations(mock_frequency_params, mock_severity_params):
    num_observations = 100
    reporting_threshold = 0
    df = generate_synthetic_ild(mock_frequency_params, mock_severity_params, num_observations, reporting_threshold)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert 'Loss_ID' in df.columns
    assert 'Amount' in df.columns

def test_generate_synthetic_ild_reporting_threshold(mock_frequency_params, mock_severity_params):
    num_observations = 100
    reporting_threshold = 1000
    df = generate_synthetic_ild(mock_frequency_params, mock_severity_params, num_observations, reporting_threshold)
    assert all(df['Amount'] >= reporting_threshold)

def test_generate_synthetic_ild_zero_observations(mock_frequency_params, mock_severity_params):
    num_observations = 0
    reporting_threshold = 0
    df = generate_synthetic_ild(mock_frequency_params, mock_severity_params, num_observations, reporting_threshold)
    assert isinstance(df, pd.DataFrame)
    assert df.empty

def test_generate_synthetic_ild_invalid_frequency_params():
    severity_params = {'distribution': 'lognorm', 'mean': 8, 'sigma': 1.5}
    num_observations = 100
    reporting_threshold = 0
    with pytest.raises(TypeError):
        generate_synthetic_ild("invalid", severity_params, num_observations, reporting_threshold)

def test_generate_synthetic_ild_invalid_severity_params():
    frequency_params = {'distribution': 'poisson', 'lambda': 5}
    num_observations = 100
    reporting_threshold = 0
    with pytest.raises(TypeError):
        generate_synthetic_ild(frequency_params, "invalid", num_observations, reporting_threshold)
