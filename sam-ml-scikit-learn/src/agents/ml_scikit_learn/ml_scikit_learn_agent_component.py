"""ML Scikit-Learn agent component for handling machine learning operations."""

import copy
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from pathlib import Path

from solace_ai_connector.common.log import log
from solace_agent_mesh.agents.base_agent_component import (
    agent_info,
    BaseAgentComponent,
)

from .actions.exploratory_data_analysis import ExploratoryDataAnalysis
from .actions.regression_analysis import RegressionAnalysis
from .actions.classification_analysis import ClassificationAnalysis
from .actions.outlier_detection import OutlierDetection
from .actions.model_persistence import ModelPersistence
from .services.ml_service import MLService


info = copy.deepcopy(agent_info)
info.update(
    {
        "agent_name": "ml_scikit_learn",
        "class_name": "MLScikitLearnAgentComponent",
        "description": "Machine Learning agent for regression, classification, EDA, and outlier detection using scikit-learn",
        "config_parameters": [
            {
                "name": "agent_name",
                "required": True,
                "description": "Name of this ML agent instance",
                "type": "string",
            },
            {
                "name": "data_source",
                "required": False,
                "description": "Default data source type (csv, database, api)",
                "type": "string",
                "default": "csv",
            },
            {
                "name": "data_path",
                "required": False,
                "description": "Default path to data file or database connection string",
                "type": "string",
            },
            {
                "name": "target_column",
                "required": False,
                "description": "Default target column for ML tasks",
                "type": "string",
            },
            {
                "name": "feature_columns",
                "required": False,
                "description": "Default feature columns for ML tasks",
                "type": "list",
            },
            {
                "name": "test_size",
                "required": False,
                "description": "Default test size for train-test split",
                "type": "float",
                "default": 0.2,
            },
            {
                "name": "random_state",
                "required": False,
                "description": "Random state for reproducibility",
                "type": "integer",
                "default": 42,
            },
            {
                "name": "model_storage_path",
                "required": False,
                "description": "Path to store trained models",
                "type": "string",
                "default": "./models",
            },
            {
                "name": "visualization_output_path",
                "required": False,
                "description": "Path to store generated visualizations",
                "type": "string",
                "default": "./visualizations",
            },
            {
                "name": "max_memory_usage",
                "required": False,
                "description": "Maximum memory usage in MB",
                "type": "integer",
                "default": 1024,
            },
            {
                "name": "parallel_jobs",
                "required": False,
                "description": "Number of parallel jobs for computations",
                "type": "integer",
                "default": -1,
            },
            {
                "name": "default_model_type",
                "required": False,
                "description": "Default model type for regression/classification",
                "type": "string",
                "default": "random_forest",
            },
            {
                "name": "enable_caching",
                "required": False,
                "description": "Enable model and computation caching",
                "type": "boolean",
                "default": True,
            },
            {
                "name": "response_guidelines",
                "required": False,
                "description": "Guidelines to be attached to action responses",
                "type": "string",
            }
        ],
    }
)


class MLScikitLearnAgentComponent(BaseAgentComponent):
    """Component for handling machine learning operations using scikit-learn."""

    info = info
    actions = [
        ExploratoryDataAnalysis,
        RegressionAnalysis,
        ClassificationAnalysis,
        OutlierDetection,
        ModelPersistence,
    ]

    def __init__(self, module_info: Dict[str, Any] = None, **kwargs):
        """Initialize the ML Scikit-Learn agent component.

        Args:
            module_info: Optional module configuration.
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If required configuration is missing.
        """
        module_info = module_info or info
        super().__init__(module_info, **kwargs)

        self.agent_name = self.get_config("agent_name")
        self.data_source = self.get_config("data_source", "csv")
        self.data_path = self.get_config("data_path")
        self.target_column = self.get_config("target_column")
        self.feature_columns = self.get_config("feature_columns", [])
        self.test_size = self.get_config("test_size", 0.2)
        self.random_state = self.get_config("random_state", 42)
        self.model_storage_path = self.get_config("model_storage_path", "./models")
        self.visualization_output_path = self.get_config("visualization_output_path", "./visualizations")
        self.max_memory_usage = self.get_config("max_memory_usage", 1024)
        self.parallel_jobs = self.get_config("parallel_jobs", -1)
        self.default_model_type = self.get_config("default_model_type", "random_forest")
        self.enable_caching = self.get_config("enable_caching", True)
        self.response_guidelines = self.get_config("response_guidelines", "")

        self.action_list.fix_scopes("<agent_name>", self.agent_name)
        module_info["agent_name"] = self.agent_name

        # Initialize ML service
        self.ml_service = self._create_ml_service()

        # Create storage directories
        self._create_storage_directories()

        # Generate agent description
        self._generate_agent_description()

        # Log prominent startup message
        log.info("=" * 80)
        log.info("🚀 ML SCIKIT-LEARN AGENT STARTED SUCCESSFULLY")
        log.info("=" * 80)
        log.info("Agent Name: %s", self.agent_name)
        log.info("Version: %s", "0.0.0+local.nram")
        log.info("Available Actions: %s", [action.__name__ for action in self.actions])
        log.info("Data Source: %s", self.data_source)
        if self.data_path:
            log.info("Data Path: %s", self.data_path)
        else:
            log.warning("⚠️  ML_DATA_PATH environment variable not set - agent may not function properly")
        log.info("Model Storage: %s", self.model_storage_path)
        log.info("Visualization Output: %s", self.visualization_output_path)
        log.info("Default Model Type: %s", self.default_model_type)
        log.info("Parallel Jobs: %s", self.parallel_jobs)
        log.info("Caching Enabled: %s", self.enable_caching)
        log.info("=" * 80)
        log.info("✅ ML Scikit-Learn Agent is ready for machine learning tasks!")
        log.info("🔍 Agent should be available in SAM as 'ml_scikit_learn'")
        log.info("=" * 80)
        
        # Also print to stdout for immediate visibility
        print("=" * 80)
        print("🚀 ML SCIKIT-LEARN AGENT STARTED SUCCESSFULLY")
        print("=" * 80)
        print(f"Agent Name: {self.agent_name}")
        print(f"Version: 0.0.0+local.nram")
        print(f"Available Actions: {[action.__name__ for action in self.actions]}")
        print("=" * 80)
        print("✅ ML Scikit-Learn Agent is ready for machine learning tasks!")
        print("🔍 Agent should be available in SAM as 'ml_scikit_learn'")
        print("=" * 80)

    def _create_ml_service(self) -> MLService:
        """Create and configure the ML service.

        Returns:
            Configured MLService instance.
        """
        return MLService(
            random_state=self.random_state,
            parallel_jobs=self.parallel_jobs,
            max_memory_usage=self.max_memory_usage,
            enable_caching=self.enable_caching,
        )

    def _create_storage_directories(self):
        """Create necessary storage directories."""
        try:
            Path(self.model_storage_path).mkdir(parents=True, exist_ok=True)
            Path(self.visualization_output_path).mkdir(parents=True, exist_ok=True)
            log.info("Created storage directories: %s, %s", 
                    self.model_storage_path, self.visualization_output_path)
        except Exception as e:
            log.warning("Failed to create storage directories: %s", str(e))

    def _generate_agent_description(self):
        """Generate the agent description based on configuration."""
        description_parts = [
            f"ML Scikit-Learn agent '{self.agent_name}'",
            f"Default data source: {self.data_source}",
        ]
        
        if self.data_path:
            description_parts.append(f"Default data path: {self.data_path}")
        
        if self.target_column:
            description_parts.append(f"Default target column: {self.target_column}")
        
        if self.feature_columns:
            description_parts.append(f"Default feature columns: {', '.join(self.feature_columns)}")
        
        description_parts.extend([
            f"Test size: {self.test_size}",
            f"Random state: {self.random_state}",
            f"Default model type: {self.default_model_type}",
            f"Parallel jobs: {self.parallel_jobs}",
            f"Caching enabled: {self.enable_caching}",
        ])

        self.agent_description = " | ".join(description_parts)

    def get_agent_summary(self):
        """Get a summary of the agent configuration.

        Returns:
            Dictionary containing agent summary.
        """
        return {
            "agent_name": self.agent_name,
            "description": self.agent_description,
            "data_source": self.data_source,
            "data_path": self.data_path,
            "target_column": self.target_column,
            "feature_columns": self.feature_columns,
            "test_size": self.test_size,
            "random_state": self.random_state,
            "default_model_type": self.default_model_type,
            "model_storage_path": self.model_storage_path,
            "visualization_output_path": self.visualization_output_path,
            "parallel_jobs": self.parallel_jobs,
            "enable_caching": self.enable_caching,
        }

    def get_ml_service(self) -> MLService:
        """Get the ML service instance.

        Returns:
            MLService instance.
        """
        return self.ml_service

    def load_data(self, data_source: str = None, data_path: str = None) -> pd.DataFrame:
        """Load data from the specified source.

        Args:
            data_source: Type of data source (csv, database, api)
            data_path: Path to data or connection string

        Returns:
            Loaded pandas DataFrame.

        Raises:
            ValueError: If data source is not supported or data cannot be loaded.
        """
        data_source = data_source or self.data_source
        data_path = data_path or self.data_path

        if not data_path:
            raise ValueError("Data path is required")

        try:
            if data_source.lower() == "csv":
                return pd.read_csv(data_path)
            elif data_source.lower() == "excel":
                return pd.read_excel(data_path)
            elif data_source.lower() == "json":
                return pd.read_json(data_path)
            elif data_source.lower() == "parquet":
                return pd.read_parquet(data_path)
            else:
                raise ValueError(f"Unsupported data source: {data_source}")
        except Exception as e:
            raise ValueError(f"Failed to load data from {data_path}: {str(e)}")

    def validate_data(self, df: pd.DataFrame, target_column: str = None, 
                     feature_columns: List[str] = None) -> Dict[str, Any]:
        """Validate data for ML operations.

        Args:
            df: Input DataFrame
            target_column: Target column name
            feature_columns: List of feature column names

        Returns:
            Validation results dictionary.

        Raises:
            ValueError: If validation fails.
        """
        target_column = target_column or self.target_column
        feature_columns = feature_columns or self.feature_columns

        validation_results = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
            "data_info": {}
        }

        # Check if DataFrame is empty
        if df.empty:
            validation_results["is_valid"] = False
            validation_results["errors"].append("DataFrame is empty")
            return validation_results

        # Check for required columns
        if target_column and target_column not in df.columns:
            validation_results["is_valid"] = False
            validation_results["errors"].append(f"Target column '{target_column}' not found")

        if feature_columns:
            missing_features = [col for col in feature_columns if col not in df.columns]
            if missing_features:
                validation_results["is_valid"] = False
                validation_results["errors"].append(f"Feature columns not found: {missing_features}")

        # Check for missing values
        if target_column and df[target_column].isnull().any():
            validation_results["warnings"].append(f"Target column '{target_column}' contains missing values")

        if feature_columns:
            missing_in_features = df[feature_columns].isnull().sum()
            if missing_in_features.any():
                validation_results["warnings"].append(f"Feature columns contain missing values: {missing_in_features[missing_in_features > 0].to_dict()}")

        # Check data types
        if target_column:
            validation_results["data_info"]["target_type"] = str(df[target_column].dtype)

        if feature_columns:
            validation_results["data_info"]["feature_types"] = df[feature_columns].dtypes.to_dict()

        # Check for sufficient data
        min_samples = 10
        if len(df) < min_samples:
            validation_results["warnings"].append(f"Dataset has only {len(df)} samples, minimum recommended is {min_samples}")

        if validation_results["errors"]:
            raise ValueError(f"Data validation failed: {'; '.join(validation_results['errors'])}")

        return validation_results 