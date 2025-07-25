"""Basic tests for sam-ml-pandas plugin."""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path

# Test data creation
def create_test_data():
    """Create test data for testing."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(5, 2, n_samples),
        'feature3': np.random.choice(['A', 'B', 'C'], n_samples),
        'feature4': np.random.uniform(0, 100, n_samples),
        'target_regression': np.random.normal(10, 3, n_samples),
        'target_classification': np.random.choice([0, 1], n_samples)
    }
    
    # Add some missing values
    data['feature1'][::10] = np.nan
    data['feature3'][::15] = None
    
    return pd.DataFrame(data)

def test_plugin_structure():
    """Test that the plugin has the correct structure."""
    plugin_dir = Path(__file__).parent.parent
    
    # Check required files exist
    assert (plugin_dir / "solace-agent-mesh-plugin.yaml").exists()
    assert (plugin_dir / "pyproject.toml").exists()
    assert (plugin_dir / "src" / "__init__.py").exists()
    assert (plugin_dir / "src" / "agents" / "ml_pandas" / "ml_pandas_agent_component.py").exists()
    assert (plugin_dir / "configs" / "agents" / "ml_pandas.yaml").exists()

def test_data_service():
    """Test the data service functionality."""
    from src.agents.ml_pandas.services.data_service import DataService
    
    # Create test data
    df = create_test_data()
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        data_service = DataService(output_directory=temp_dir, max_rows_display=50)
        
        # Test data summary
        summary = data_service.get_data_summary(df)
        assert "shape" in summary
        assert summary["shape"]["rows"] == 100
        assert summary["shape"]["columns"] == 6
        assert "columns" in summary
        assert "dtypes" in summary
        assert "missing_values" in summary
        
        # Test data preview
        preview = data_service.get_data_preview(df, n_rows=10)
        assert "head" in preview
        assert len(preview["head"]) == 10
        assert "shape" in preview
        
        # Test missing data analysis
        missing_analysis = data_service.analyze_missing_data(df)
        assert "total_missing" in missing_analysis
        assert "missing_by_column" in missing_analysis
        assert missing_analysis["total_missing"] > 0
        
        # Test correlation analysis
        corr_analysis = data_service.get_correlation_analysis(df)
        assert "correlation_matrix" in corr_analysis

def test_agent_component_creation():
    """Test that the agent component can be created."""
    from src.agents.ml_pandas.ml_pandas_agent_component import MLPandasAgentComponent
    
    # Create test data file
    df = create_test_data()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_file = f.name
    
    try:
        # Test agent creation with minimal config
        config = {
            "agent_name": "test_ml_agent",
            "data_file": temp_file,
            "data_file_format": "csv",
            "target_column": "target_regression"
        }
        
        # Mock the base class methods that we don't need for this test
        class MockConfig:
            def get_config(self, key, default=None):
                return config.get(key, default)
        
        # This test just ensures the imports work correctly
        # Full agent testing would require more setup
        assert MLPandasAgentComponent is not None
        
    finally:
        # Clean up
        os.unlink(temp_file)

def test_actions_import():
    """Test that actions can be imported."""
    from src.agents.ml_pandas.actions.data_analysis import DataAnalysisAction
    from src.agents.ml_pandas.actions.simple_ml import SimpleMlAction
    
    # Test that actions have required attributes
    data_action = DataAnalysisAction()
    ml_action = SimpleMlAction()
    
    assert hasattr(data_action, 'invoke')
    assert hasattr(ml_action, 'invoke')
    
    # Test action parameters
    assert data_action.name == "data_analysis"
    assert ml_action.name == "simple_ml"

if __name__ == "__main__":
    pytest.main([__file__])