"""Dataset service for generating and retrieving ML datasets."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn import datasets
import seaborn as sns

from solace_ai_connector.common.log import log


class DatasetService:
    """Service for providing various ML datasets."""
    
    def __init__(self, default_max_records: int = 100):
        """Initialize the dataset service.
        
        Args:
            default_max_records: Default maximum number of records to return
        """
        self.default_max_records = default_max_records
        
    def get_sklearn_dataset(self, dataset_name: str, max_records: Optional[int] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Get a sklearn dataset.
        
        Args:
            dataset_name: Name of the sklearn dataset
            max_records: Maximum number of records to return
            
        Returns:
            Tuple of (DataFrame, metadata)
            
        Raises:
            ValueError: If dataset name is not supported
        """
        # Convert max_records to integer if it's a string
        if isinstance(max_records, str):
            try:
                max_records = int(max_records)
            except (ValueError, TypeError):
                raise ValueError(f"max_records must be a positive integer, got: {max_records}")
        
        # Ensure max_records is a positive integer
        if max_records is not None and max_records <= 0:
            raise ValueError(f"max_records must be a positive integer, got: {max_records}")
        
        max_records = max_records or self.default_max_records
        
        sklearn_datasets = {
            'iris': datasets.load_iris,
            'wine': datasets.load_wine,
            'breast_cancer': datasets.load_breast_cancer,
            'digits': datasets.load_digits,
            'boston': self._load_boston_housing,  # Custom wrapper due to deprecation
            'diabetes': datasets.load_diabetes,
            'linnerud': datasets.load_linnerud
        }
        
        if dataset_name not in sklearn_datasets:
            available = list(sklearn_datasets.keys())
            raise ValueError(f"Dataset '{dataset_name}' not supported. Available: {available}")
            
        log.info(f"ml-datasets: Loading sklearn dataset: {dataset_name}")

        if dataset_name == "boston":
            # Boston housing dataset is deprecated, create synthetic alternative
            log.warning("ml-datasets: Boston housing dataset deprecated, creating synthetic alternative")
            return self._load_boston_housing(max_records)
        
        # Load the dataset
        try:
            if dataset_name == "iris":
                data = datasets.load_iris()
            elif dataset_name == "wine":
                data = datasets.load_wine()
            elif dataset_name == "breast_cancer":
                data = datasets.load_breast_cancer()
            elif dataset_name == "digits":
                data = datasets.load_digits()
            elif dataset_name == "diabetes":
                data = datasets.load_diabetes()
            elif dataset_name == "linnerud":
                data = datasets.load_linnerud()
            else:
                raise ValueError(f"Unknown sklearn dataset: {dataset_name}")

            # Convert to DataFrame
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target

            # Limit records if needed
            if len(df) > max_records:
                log.info(f"ml-datasets: Limited dataset to {max_records} records (original had {len(data.data)})")
                df = df.head(max_records)

            # Create metadata
            metadata = {
                "dataset_name": dataset_name,
                "dataset_type": "sklearn",
                "n_samples": len(df),
                "n_features": len(df.columns) - 1,  # Exclude target
                "feature_names": data.feature_names,
                "target_name": "target",
                "description": data.DESCR.split('\n')[0] if hasattr(data, 'DESCR') else f"Sklearn {dataset_name} dataset"
            }

            return df, metadata

        except Exception as e:
            raise ValueError(f"Failed to load sklearn dataset '{dataset_name}': {str(e)}")
    
    def _load_boston_housing(self):
        """Load Boston housing dataset using alternative method due to sklearn deprecation."""
        try:
            # Try the old method first
            return datasets.load_boston()
        except ImportError:
            # If deprecated, create a simple synthetic housing dataset
            log.warning("ml-datasets: Boston housing dataset deprecated, creating synthetic alternative")
            n_samples = 506
            n_features = 13
            
            # Generate synthetic housing data
            np.random.seed(42)
            X = np.random.randn(n_samples, n_features)
            y = np.random.randn(n_samples) * 10 + 25  # Housing prices around $25k
            
            # Create a simple dataset object
            class SimpleDataset:
                def __init__(self, data, target):
                    self.data = data
                    self.target = target
                    self.feature_names = [f'feature_{i}' for i in range(n_features)]
                    self.DESCR = "Synthetic housing dataset (Boston dataset deprecated)"
            
            return SimpleDataset(X, y)
    
    def get_seaborn_dataset(self, dataset_name: str, max_records: Optional[int] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Get a seaborn dataset.
        
        Args:
            dataset_name: Name of the seaborn dataset
            max_records: Maximum number of records to return
            
        Returns:
            Tuple of (DataFrame, metadata)
            
        Raises:
            ValueError: If dataset name is not supported
        """
        # Convert max_records to integer if it's a string
        if isinstance(max_records, str):
            try:
                max_records = int(max_records)
            except (ValueError, TypeError):
                raise ValueError(f"max_records must be a positive integer, got: {max_records}")
        
        # Ensure max_records is a positive integer
        if max_records is not None and max_records <= 0:
            raise ValueError(f"max_records must be a positive integer, got: {max_records}")
        
        max_records = max_records or self.default_max_records
        
        seaborn_datasets = [
            'tips', 'flights', 'titanic', 'iris', 'car_crashes', 'mpg',
            'diamonds', 'attention', 'dots', 'exercise', 'gammas', 'geyser',
            'penguins', 'planets', 'taxis', 'fmri', 'anagrams', 'anscombe'
        ]
        
        if dataset_name not in seaborn_datasets:
            raise ValueError(f"Dataset '{dataset_name}' not supported. Available: {seaborn_datasets}")
            
        log.info(f"ml-datasets: Loading seaborn dataset: {dataset_name}")
        
        try:
            if dataset_name == "tips":
                df = sns.load_dataset("tips")
            elif dataset_name == "flights":
                df = sns.load_dataset("flights")
            elif dataset_name == "titanic":
                df = sns.load_dataset("titanic")
            elif dataset_name == "car_crashes":
                df = sns.load_dataset("car_crashes")
            elif dataset_name == "mpg":
                df = sns.load_dataset("mpg")
            elif dataset_name == "penguins":
                df = sns.load_dataset("penguins")
            else:
                raise ValueError(f"Unknown seaborn dataset: {dataset_name}")

            # Limit records if needed
            original_size = len(df)
            if len(df) > max_records:
                log.info(f"ml-datasets: Limited dataset to {max_records} records (original had {original_size})")
                df = df.head(max_records)

            # Create metadata
            metadata = {
                "dataset_name": dataset_name,
                "dataset_type": "seaborn",
                "n_samples": len(df),
                "n_features": len(df.columns),
                "column_names": list(df.columns),
                "data_types": df.dtypes.to_dict(),
                "description": f"Seaborn {dataset_name} dataset"
            }

            return df, metadata

        except Exception as e:
            raise ValueError(f"Failed to load seaborn dataset '{dataset_name}': {str(e)}")
    
    def generate_synthetic_dataset(self, dataset_type: str, n_samples: Optional[int] = None, 
                                 **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Generate a synthetic dataset.
        
        Args:
            dataset_type: Type of synthetic dataset
            n_samples: Number of samples to generate
            **kwargs: Additional parameters for dataset generation
            
        Returns:
            Tuple of (DataFrame, metadata)
            
        Raises:
            ValueError: If dataset type is not supported
        """
        # Convert n_samples to integer if it's a string
        if isinstance(n_samples, str):
            try:
                n_samples = int(n_samples)
            except (ValueError, TypeError):
                raise ValueError(f"n_samples must be a positive integer, got: {n_samples}")
        
        # Ensure n_samples is a positive integer
        if n_samples is not None and n_samples <= 0:
            raise ValueError(f"n_samples must be a positive integer, got: {n_samples}")
        
        n_samples = min(n_samples or self.default_max_records, self.default_max_records)
        
        synthetic_types = {
            'classification': self._generate_classification_data,
            'regression': self._generate_regression_data,
            'clustering': self._generate_clustering_data,
            'blobs': self._generate_blob_data,
            'moons': self._generate_moons_data,
            'circles': self._generate_circles_data
        }
        
        if dataset_type not in synthetic_types:
            available = list(synthetic_types.keys())
            raise ValueError(f"Synthetic dataset type '{dataset_type}' not supported. Available: {available}")
            
        log.info(f"ml-datasets: Generating synthetic {dataset_type} dataset with {n_samples} samples")
        
        return synthetic_types[dataset_type](n_samples, **kwargs)
    
    def _convert_to_int(self, value, param_name: str) -> int:
        """Convert a value to integer with proper error handling.
        
        Args:
            value: Value to convert
            param_name: Name of the parameter for error messages
            
        Returns:
            Integer value
            
        Raises:
            ValueError: If conversion fails
        """
        if isinstance(value, str):
            try:
                value = int(value)
            except (ValueError, TypeError):
                raise ValueError(f"{param_name} must be a positive integer, got: {value}")
        
        if value <= 0:
            raise ValueError(f"{param_name} must be a positive integer, got: {value}")
        
        return value
    
    def _convert_to_float(self, value, param_name: str) -> float:
        """Convert a value to float with proper error handling.
        
        Args:
            value: Value to convert
            param_name: Name of the parameter for error messages
            
        Returns:
            Float value
            
        Raises:
            ValueError: If conversion fails
        """
        if isinstance(value, str):
            try:
                value = float(value)
            except (ValueError, TypeError):
                raise ValueError(f"{param_name} must be a valid number, got: {value}")
        
        return value

    def _generate_classification_data(self, n_samples: int, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Generate synthetic classification data."""
        # Convert parameters to proper types
        n_features = self._convert_to_int(kwargs.get('n_features', 4), 'n_features')
        n_classes = self._convert_to_int(kwargs.get('n_classes', 2), 'n_classes')
        n_informative = self._convert_to_int(kwargs.get('n_informative', min(n_features, 2)), 'n_informative')
        
        X, y = datasets.make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_classes=n_classes,
            random_state=42
        )
        
        # Create DataFrame
        feature_cols = [f'feature_{i}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_cols)
        df['target'] = y
        df['target_name'] = [f'class_{i}' for i in y]
        
        metadata = {
            'dataset_type': 'synthetic',
            'dataset_name': 'classification',
            'description': f"Synthetic classification dataset with {n_classes} classes",
            'n_samples': n_samples,
            'n_features': n_features,
            'n_classes': n_classes,
            'columns': list(df.columns)
        }
        
        return df, metadata
    
    def _generate_regression_data(self, n_samples: int, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Generate synthetic regression data."""
        # Convert parameters to proper types
        n_features = self._convert_to_int(kwargs.get('n_features', 4), 'n_features')
        noise = self._convert_to_float(kwargs.get('noise', 0.1), 'noise')
        
        X, y = datasets.make_regression(
            n_samples=n_samples,
            n_features=n_features,
            noise=noise,
            random_state=42
        )
        
        # Create DataFrame
        feature_cols = [f'feature_{i}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_cols)
        df['target'] = y
        
        metadata = {
            'dataset_type': 'synthetic',
            'dataset_name': 'regression',
            'description': f"Synthetic regression dataset with {n_features} features",
            'n_samples': n_samples,
            'n_features': n_features,
            'noise_level': noise,
            'columns': list(df.columns)
        }
        
        return df, metadata
    
    def _generate_clustering_data(self, n_samples: int, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Generate synthetic clustering data."""
        # Convert parameters to proper types
        n_centers = self._convert_to_int(kwargs.get('n_centers', 3), 'n_centers')
        n_features = self._convert_to_int(kwargs.get('n_features', 2), 'n_features')
        
        X, y = datasets.make_blobs(
            n_samples=n_samples,
            centers=n_centers,
            n_features=n_features,
            random_state=42
        )
        
        # Create DataFrame
        feature_cols = [f'feature_{i}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_cols)
        df['cluster'] = y
        
        metadata = {
            'dataset_type': 'synthetic',
            'dataset_name': 'clustering',
            'description': f"Synthetic clustering dataset with {n_centers} clusters",
            'n_samples': n_samples,
            'n_features': n_features,
            'n_clusters': n_centers,
            'columns': list(df.columns)
        }
        
        return df, metadata
    
    def _generate_blob_data(self, n_samples: int, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Generate blob data for clustering."""
        return self._generate_clustering_data(n_samples, **kwargs)
    
    def _generate_moons_data(self, n_samples: int, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Generate moons data."""
        # Convert parameters to proper types
        noise = self._convert_to_float(kwargs.get('noise', 0.1), 'noise')
        
        X, y = datasets.make_moons(n_samples=n_samples, noise=noise, random_state=42)
        
        df = pd.DataFrame(X, columns=['feature_0', 'feature_1'])
        df['target'] = y
        df['target_name'] = [f'moon_{i}' for i in y]
        
        metadata = {
            'dataset_type': 'synthetic',
            'dataset_name': 'moons',
            'description': "Synthetic two moons dataset for binary classification",
            'n_samples': n_samples,
            'n_features': 2,
            'noise_level': noise,
            'columns': list(df.columns)
        }
        
        return df, metadata
    
    def _generate_circles_data(self, n_samples: int, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Generate circles data."""
        # Convert parameters to proper types
        noise = self._convert_to_float(kwargs.get('noise', 0.1), 'noise')
        
        X, y = datasets.make_circles(n_samples=n_samples, noise=noise, random_state=42)
        
        df = pd.DataFrame(X, columns=['feature_0', 'feature_1'])
        df['target'] = y
        df['target_name'] = [f'circle_{i}' for i in y]
        
        metadata = {
            'dataset_type': 'synthetic',
            'dataset_name': 'circles',
            'description': "Synthetic concentric circles dataset for binary classification",
            'n_samples': n_samples,
            'n_features': 2,
            'noise_level': noise,
            'columns': list(df.columns)
        }
        
        return df, metadata
    
    def list_available_datasets(self) -> Dict[str, List[str]]:
        """List all available datasets.
        
        Returns:
            Dictionary mapping dataset type to list of available datasets
        """
        return {
            'sklearn': ['iris', 'wine', 'breast_cancer', 'digits', 'boston', 'diabetes', 'linnerud'],
            'seaborn': ['tips', 'flights', 'titanic', 'iris', 'car_crashes', 'mpg', 'diamonds', 
                       'attention', 'dots', 'exercise', 'gammas', 'geyser', 'penguins', 
                       'planets', 'taxis', 'fmri', 'anagrams', 'anscombe'],
            'synthetic': ['classification', 'regression', 'clustering', 'blobs', 'moons', 'circles']
        }