"""Basic tests for ML Datasets plugin."""

import unittest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.ml_datasets.services.dataset_service import DatasetService
from agents.ml_datasets.ml_datasets_agent_component import MLDatasetsAgentComponent


class TestDatasetService(unittest.TestCase):
    """Test the dataset service."""

    def setUp(self):
        """Set up test fixtures."""
        self.service = DatasetService(default_max_records=50)

    def test_list_available_datasets(self):
        """Test listing available datasets."""
        datasets = self.service.list_available_datasets()
        
        # Check structure
        self.assertIsInstance(datasets, dict)
        self.assertIn('sklearn', datasets)
        self.assertIn('seaborn', datasets)
        self.assertIn('synthetic', datasets)
        
        # Check sklearn datasets
        sklearn_datasets = datasets['sklearn']
        self.assertIn('iris', sklearn_datasets)
        self.assertIn('wine', sklearn_datasets)
        
        # Check seaborn datasets
        seaborn_datasets = datasets['seaborn']
        self.assertIn('tips', seaborn_datasets)
        self.assertIn('flights', seaborn_datasets)
        
        # Check synthetic datasets
        synthetic_datasets = datasets['synthetic']
        self.assertIn('classification', synthetic_datasets)
        self.assertIn('regression', synthetic_datasets)

    def test_get_sklearn_dataset(self):
        """Test getting sklearn datasets."""
        df, metadata = self.service.get_sklearn_dataset('iris')
        
        # Check DataFrame
        self.assertIsNotNone(df)
        self.assertGreater(len(df), 0)
        self.assertLessEqual(len(df), 50)  # Respects max_records
        
        # Check metadata
        self.assertIsInstance(metadata, dict)
        self.assertEqual(metadata['dataset_type'], 'sklearn')
        self.assertEqual(metadata['dataset_name'], 'iris')
        self.assertIn('description', metadata)
        self.assertIn('n_samples', metadata)
        self.assertIn('n_features', metadata)

    def test_get_seaborn_dataset(self):
        """Test getting seaborn datasets."""
        df, metadata = self.service.get_seaborn_dataset('tips')
        
        # Check DataFrame
        self.assertIsNotNone(df)
        self.assertGreater(len(df), 0)
        self.assertLessEqual(len(df), 50)  # Respects max_records
        
        # Check metadata
        self.assertIsInstance(metadata, dict)
        self.assertEqual(metadata['dataset_type'], 'seaborn')
        self.assertEqual(metadata['dataset_name'], 'tips')
        self.assertIn('n_samples', metadata)
        self.assertIn('n_features', metadata)

    def test_generate_synthetic_dataset(self):
        """Test generating synthetic datasets."""
        df, metadata = self.service.generate_synthetic_dataset('classification', n_samples=30)
        
        # Check DataFrame
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 30)
        self.assertIn('target', df.columns)
        
        # Check metadata
        self.assertIsInstance(metadata, dict)
        self.assertEqual(metadata['dataset_type'], 'synthetic')
        self.assertEqual(metadata['dataset_name'], 'classification')
        self.assertIn('n_samples', metadata)
        self.assertIn('n_features', metadata)

    def test_invalid_dataset_name(self):
        """Test handling invalid dataset names."""
        with self.assertRaises(ValueError):
            self.service.get_sklearn_dataset('invalid_dataset')
            
        with self.assertRaises(ValueError):
            self.service.get_seaborn_dataset('invalid_dataset')
            
        with self.assertRaises(ValueError):
            self.service.generate_synthetic_dataset('invalid_type')


class TestMLDatasetsAgentComponent(unittest.TestCase):
    """Test the ML Datasets agent component."""

    def test_info_structure(self):
        """Test the agent info structure."""
        from agents.ml_datasets.ml_datasets_agent_component import info
        
        self.assertIsInstance(info, dict)
        self.assertEqual(info['agent_name'], 'ml_datasets')
        self.assertEqual(info['class_name'], 'MLDatasetsAgentComponent')
        self.assertIn('description', info)
        self.assertIn('config_parameters', info)
        
        # Check config parameters
        config_params = info['config_parameters']
        param_names = [p['name'] for p in config_params]
        self.assertIn('agent_name', param_names)
        self.assertIn('default_max_records', param_names)

    def test_actions_list(self):
        """Test that actions are properly defined."""
        from agents.ml_datasets.actions.get_dataset import GetDataset
        from agents.ml_datasets.actions.list_datasets import ListDatasets
        
        self.assertTrue(hasattr(GetDataset, 'invoke'))
        self.assertTrue(hasattr(ListDatasets, 'invoke'))


if __name__ == '__main__':
    unittest.main()