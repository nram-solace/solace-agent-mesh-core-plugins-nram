"""Basic tests for ML Datasets plugin."""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path

# Test data creation
def create_test_dataset():
    """Create test dataset for testing."""
    np.random.seed(42)
    n_samples = 50
    
    data = {
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(5, 2, n_samples),
        'feature3': np.random.choice(['A', 'B', 'C'], n_samples),
        'feature4': np.random.uniform(0, 100, n_samples),
        'target': np.random.choice([0, 1], n_samples)
    }
    
    return pd.DataFrame(data)

def test_plugin_structure():
    """Test that the plugin has the correct structure."""
    plugin_dir = Path(__file__).parent.parent
    
    # Check required files exist
    assert (plugin_dir / "solace-agent-mesh-plugin.yaml").exists()
    assert (plugin_dir / "pyproject.toml").exists()
    assert (plugin_dir / "src" / "__init__.py").exists()
    assert (plugin_dir / "src" / "agents" / "ml_datasets" / "ml_datasets_agent_component.py").exists()
    assert (plugin_dir / "configs" / "agents" / "ml_datasets.yaml").exists()

def test_dataset_service():
    """Test the dataset service functionality."""
    from src.agents.ml_datasets.services.dataset_service import DatasetService
    
    # Create dataset service
    service = DatasetService(default_max_records=50)
    
    # Test listing available datasets
    datasets = service.list_available_datasets()
    
    # Check structure
    assert isinstance(datasets, dict)
    assert 'sklearn' in datasets
    assert 'seaborn' in datasets
    assert 'synthetic' in datasets
    
    # Check sklearn datasets
    sklearn_datasets = datasets['sklearn']
    assert 'iris' in sklearn_datasets
    assert 'wine' in sklearn_datasets
    
    # Check seaborn datasets
    seaborn_datasets = datasets['seaborn']
    assert 'tips' in seaborn_datasets
    assert 'flights' in seaborn_datasets
    
    # Check synthetic datasets
    synthetic_datasets = datasets['synthetic']
    assert 'classification' in synthetic_datasets
    assert 'regression' in synthetic_datasets

def test_get_sklearn_dataset():
    """Test getting sklearn datasets."""
    from src.agents.ml_datasets.services.dataset_service import DatasetService
    
    service = DatasetService(default_max_records=50)
    df, metadata = service.get_sklearn_dataset('iris')
    
    # Check DataFrame
    assert df is not None
    assert len(df) > 0
    assert len(df) <= 50  # Respects max_records
    
    # Check metadata
    assert isinstance(metadata, dict)
    assert metadata['dataset_type'] == 'sklearn'
    assert metadata['dataset_name'] == 'iris'
    assert 'description' in metadata
    assert 'n_samples' in metadata
    assert 'n_features' in metadata

def test_get_seaborn_dataset():
    """Test getting seaborn datasets."""
    from src.agents.ml_datasets.services.dataset_service import DatasetService
    
    service = DatasetService(default_max_records=50)
    df, metadata = service.get_seaborn_dataset('tips')
    
    # Check DataFrame
    assert df is not None
    assert len(df) > 0
    assert len(df) <= 50  # Respects max_records
    
    # Check metadata
    assert isinstance(metadata, dict)
    assert metadata['dataset_type'] == 'seaborn'
    assert metadata['dataset_name'] == 'tips'
    assert 'n_samples' in metadata
    assert 'n_features' in metadata

def test_generate_synthetic_dataset():
    """Test generating synthetic datasets."""
    from src.agents.ml_datasets.services.dataset_service import DatasetService
    
    service = DatasetService(default_max_records=50)
    df, metadata = service.generate_synthetic_dataset('classification', n_samples=30)
    
    # Check DataFrame
    assert df is not None
    assert len(df) == 30
    assert 'target' in df.columns
    
    # Check metadata
    assert isinstance(metadata, dict)
    assert metadata['dataset_type'] == 'synthetic'
    assert metadata['dataset_name'] == 'classification'
    assert 'n_samples' in metadata
    assert 'n_features' in metadata

def test_invalid_dataset_name():
    """Test handling invalid dataset names."""
    from src.agents.ml_datasets.services.dataset_service import DatasetService
    
    service = DatasetService(default_max_records=50)
    
    with pytest.raises(ValueError):
        service.get_sklearn_dataset('invalid_dataset')
        
    with pytest.raises(ValueError):
        service.get_seaborn_dataset('invalid_dataset')
        
    with pytest.raises(ValueError):
        service.generate_synthetic_dataset('invalid_type')

def test_agent_component_creation():
    """Test that the agent component can be created."""
    from src.agents.ml_datasets.ml_datasets_agent_component import MLDatasetsAgentComponent
    
    # Test agent creation with minimal config
    config = {
        "agent_name": "test_datasets_agent",
        "default_max_records": 50,
        "enable_sklearn_datasets": True,
        "enable_seaborn_datasets": True,
        "enable_synthetic_datasets": True
    }
    
    # Mock the base class methods that we don't need for this test
    class MockConfig:
        def get_config(self, key, default=None):
            return config.get(key, default)
    
    # This test just ensures the imports work correctly
    # Full agent testing would require more setup
    assert MLDatasetsAgentComponent is not None

def test_actions_import():
    """Test that actions can be imported."""
    from src.agents.ml_datasets.actions.get_dataset import GetDataset
    from src.agents.ml_datasets.actions.list_datasets import ListDatasets
    
    # Test that actions have required attributes
    get_action = GetDataset()
    list_action = ListDatasets()
    
    assert hasattr(get_action, 'invoke')
    assert hasattr(list_action, 'invoke')
    
    # Test action parameters
    assert get_action.name == "get_dataset"
    assert list_action.name == "list_datasets"

def test_info_structure():
    """Test the agent info structure."""
    from src.agents.ml_datasets.ml_datasets_agent_component import info
    
    assert isinstance(info, dict)
    assert info['agent_name'] == 'ml_datasets'
    assert info['class_name'] == 'MLDatasetsAgentComponent'
    assert 'description' in info
    assert 'config_parameters' in info
    
    # Check config parameters
    config_params = info['config_parameters']
    param_names = [p['name'] for p in config_params]
    assert 'agent_name' in param_names
    assert 'default_max_records' in param_names

if __name__ == "__main__":
    pytest.main([__file__])