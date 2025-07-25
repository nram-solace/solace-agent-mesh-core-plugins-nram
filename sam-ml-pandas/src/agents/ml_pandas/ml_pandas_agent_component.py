"""ML Pandas agent component for handling simple ML and EDA operations."""

import copy
import os
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from pathlib import Path

from solace_ai_connector.common.log import log
from solace_agent_mesh.agents.base_agent_component import (
    agent_info,
    BaseAgentComponent,
)

from .actions.data_analysis import DataAnalysisAction
from .actions.simple_ml import SimpleMlAction
from .services.data_service import DataService

# Import version
try:
    from ... import __version__
except ImportError:
    try:
        from sam_ml_pandas import __version__
    except ImportError:
        __version__ = "0.1.0+local.nram"  # Fallback version


info = copy.deepcopy(agent_info)
info.update(
    {
        "agent_name": "ml_pandas",
        "class_name": "MLPandasAgentComponent",
        "description": "Simple ML and EDA agent using pandas for data analysis and basic machine learning",
        "config_parameters": [
            {
                "name": "agent_name",
                "required": True,
                "description": "Name of this ML pandas agent instance",
                "type": "string",
            },
            {
                "name": "data_file",
                "required": True,
                "description": "Path to the data file",
                "type": "string",
            },
            {
                "name": "data_file_format",
                "required": False,
                "description": "Format of the data file (csv, json, excel, parquet)",
                "type": "string",
                "default": "csv",
            },
            {
                "name": "data_file_columns",
                "required": False,
                "description": "Comma-separated list of columns to use (empty for all columns)",
                "type": "string",
            },
            {
                "name": "target_column",
                "required": False,
                "description": "Target column for ML tasks",
                "type": "string",
            },
            {
                "name": "categorical_columns",
                "required": False,
                "description": "Comma-separated list of categorical columns",
                "type": "string",
            },
            {
                "name": "numerical_columns",
                "required": False,
                "description": "Comma-separated list of numerical columns",
                "type": "string",
            },
            {
                "name": "output_directory",
                "required": False,
                "description": "Directory to save analysis results and plots",
                "type": "string",
                "default": "./ml_pandas_output",
            },
            {
                "name": "max_rows_display",
                "required": False,
                "description": "Maximum number of rows to display in results",
                "type": "integer",
                "default": 100,
            },
        ],
    }
)


class MLPandasAgentComponent(BaseAgentComponent):
    """Component for handling simple ML and EDA operations using pandas."""

    info = info
    actions = [DataAnalysisAction, SimpleMlAction]

    def __init__(self, module_info: Dict[str, Any] = None, **kwargs):
        """Initialize the ML Pandas agent component.

        Args:
            module_info: Optional module configuration.
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If required configuration is missing.
        """
        module_info = module_info or info
        super().__init__(module_info, **kwargs)

        self.agent_name = self.get_config("agent_name")
        self.data_file = self.get_config("data_file")
        self.data_file_format = self.get_config("data_file_format", "csv")
        self.data_file_columns = self.get_config("data_file_columns", "")
        self.target_column = self.get_config("target_column", "")
        self.categorical_columns = self.get_config("categorical_columns", "")
        self.numerical_columns = self.get_config("numerical_columns", "")
        self.output_directory = self.get_config("output_directory", "./ml_pandas_output")
        self.max_rows_display = self.get_config("max_rows_display", 100)

        self.action_list.fix_scopes("<agent_name>", self.agent_name)
        module_info["agent_name"] = self.agent_name

        # Validate data file exists
        if not os.path.exists(self.data_file):
            raise ValueError(f"Data file not found: {self.data_file}")

        # Initialize data service
        self.data_service = self._create_data_service()

        # Create output directory
        self._create_output_directory()

        # Load and validate data
        self.data = self._load_data()
        
        # Parse column configurations
        self.selected_columns = self._parse_columns(self.data_file_columns)
        self.target_col = self.target_column.strip() if self.target_column else None
        self.categorical_cols = self._parse_columns(self.categorical_columns)
        self.numerical_cols = self._parse_columns(self.numerical_columns)

        # Generate and store the agent description
        self._generate_agent_description()

        # Log prominent startup message
        log.info("=" * 80)
        log.info("📊 ML PANDAS AGENT (v%s) STARTED SUCCESSFULLY", __version__)
        log.info("=" * 80)
        log.info("Agent Name: %s", self.agent_name)
        log.info("Data File: %s", self.data_file)
        log.info("Data Format: %s", self.data_file_format)
        log.info("Data Shape: %s", self.data.shape)
        log.info("Available Actions: %s", [action.__name__ for action in self.actions])
        if self.target_col:
            log.info("Target Column: %s", self.target_col)
        if self.selected_columns:
            log.info("Selected Columns: %s", len(self.selected_columns))
        log.info("Output Directory: %s", self.output_directory)
        log.info("=" * 80)
        log.info("✅ ML Pandas Agent is ready for data analysis!")
        log.info("🔍 Agent should be available in SAM as 'ml_pandas'")
        log.info("=" * 80)
        
        # Also print to stdout for immediate visibility
        print("=" * 80)
        print(f"📊 ML PANDAS AGENT (v{__version__}) STARTED SUCCESSFULLY")
        print("=" * 80)
        print(f"Agent Name: {self.agent_name}")
        print(f"Data File: {self.data_file}")
        print(f"Data Shape: {self.data.shape}")
        print(f"Available Actions: {[action.__name__ for action in self.actions]}")
        print("=" * 80)
        print("✅ ML Pandas Agent is ready for data analysis!")
        print("🔍 Agent should be available in SAM as 'ml_pandas'")
        print("=" * 80)

    def _create_data_service(self) -> DataService:
        """Create the data service instance."""
        return DataService(
            output_directory=self.output_directory,
            max_rows_display=self.max_rows_display
        )

    def _create_output_directory(self):
        """Create the output directory if it doesn't exist."""
        try:
            Path(self.output_directory).mkdir(parents=True, exist_ok=True)
            log.info("Created output directory: %s", self.output_directory)
        except Exception as e:
            log.warning("Failed to create output directory: %s", str(e))

    def _load_data(self) -> pd.DataFrame:
        """Load data from the configured file."""
        try:
            if self.data_file_format.lower() == "csv":
                data = pd.read_csv(self.data_file)
            elif self.data_file_format.lower() == "json":
                data = pd.read_json(self.data_file)
            elif self.data_file_format.lower() in ["excel", "xlsx"]:
                data = pd.read_excel(self.data_file)
            elif self.data_file_format.lower() == "parquet":
                data = pd.read_parquet(self.data_file)
            else:
                raise ValueError(f"Unsupported data format: {self.data_file_format}")
            
            log.info("Successfully loaded data with shape: %s", data.shape)
            return data
        except Exception as e:
            raise ValueError(f"Failed to load data from {self.data_file}: {str(e)}")

    def _parse_columns(self, columns_str: str) -> List[str]:
        """Parse comma-separated column string into list."""
        if not columns_str or not columns_str.strip():
            return []
        return [col.strip() for col in columns_str.split(",") if col.strip()]

    def _generate_agent_description(self):
        """Generate and store the agent description."""
        description = f"ML Pandas agent for simple data analysis and machine learning.\n\n"
        description += f"Data file: {self.data_file}\n"
        description += f"Data shape: {self.data.shape[0]} rows, {self.data.shape[1]} columns\n"
        description += f"Format: {self.data_file_format}\n"
        
        if self.target_col:
            description += f"Target column: {self.target_col}\n"
        
        if self.selected_columns:
            description += f"Selected columns: {len(self.selected_columns)} columns\n"
        else:
            description += "Using all columns\n"

        # Add column info
        column_list = self.data.columns.tolist()[:10]
        description += f"\nAvailable columns: {', '.join(column_list)}"
        if len(self.data.columns) > 10:
            description += f" (and {len(self.data.columns) - 10} more)"

        self._agent_description = {
            "agent_name": self.agent_name,
            "description": description.strip(),
            "always_open": self.info.get("always_open", False),
            "actions": self.get_actions_summary(),
        }

    def get_agent_summary(self):
        """Get a summary of the agent's capabilities."""
        return self._agent_description
    
    def get_data_service(self) -> DataService:
        """Get the data service instance."""
        return self.data_service

    def get_data(self) -> pd.DataFrame:
        """Get the loaded data."""
        return self.data

    def get_working_data(self) -> pd.DataFrame:
        """Get the working dataset (selected columns if specified)."""
        if self.selected_columns:
            available_cols = [col for col in self.selected_columns if col in self.data.columns]
            if available_cols:
                return self.data[available_cols]
        return self.data