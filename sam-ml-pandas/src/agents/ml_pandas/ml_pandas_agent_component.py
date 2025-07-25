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
from .actions.data_loader import DataLoaderAction
from .actions.data_query import DataQueryAction
from .actions.data_summarizer import DataSummarizerAction
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
        "description": "Collaborative ML and EDA agent using pandas for data analysis and basic machine learning with multi-agent workflow support",
        "config_parameters": [
            {
                "name": "agent_name",
                "required": True,
                "description": "Name of this ML pandas agent instance",
                "type": "string",
            },
            {
                "name": "data_file",
                "required": False,
                "description": "Path to the default data file (optional - can receive data from other agents)",
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
            {
                "name": "collaborative_mode",
                "required": False,
                "description": "Enable collaborative mode for multi-agent workflows",
                "type": "boolean",
                "default": True,
            },
        ],
    }
)


class MLPandasAgentComponent(BaseAgentComponent):
    """Component for handling simple ML and EDA operations using pandas."""

    info = info
    actions = [DataLoaderAction, DataQueryAction, DataAnalysisAction, SimpleMlAction, DataSummarizerAction]

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
        self.data_file = self.get_config("data_file", "")  # Make optional
        self.data_file_format = self.get_config("data_file_format", "csv")
        self.data_file_columns = self.get_config("data_file_columns", "")
        self.target_column = self.get_config("target_column", "")
        self.categorical_columns = self.get_config("categorical_columns", "")
        self.numerical_columns = self.get_config("numerical_columns", "")
        self.output_directory = self.get_config("output_directory", "./ml_pandas_output")
        self.max_rows_display = self.get_config("max_rows_display", 100)
        self.collaborative_mode = self.get_config("collaborative_mode", True)

        self.action_list.fix_scopes("<agent_name>", self.agent_name)
        self.action_list.set_agent(self)
        module_info["agent_name"] = self.agent_name

        # Initialize data service
        self.data_service = self._create_data_service()

        # Create output directory
        self._create_output_directory()

        # Initialize data storage
        self.data = None
        self.current_data_source = None
        self.data_history = []  # Track data sources for collaborative workflows
        
        # Load default data if specified
        if self.data_file and os.path.exists(self.data_file):
            try:
                self.data = self._load_data()
                self.current_data_source = f"Default file: {self.data_file}"
                self.data_history.append({"source": "file", "path": self.data_file, "shape": self.data.shape})
                log.info("ml-pandas: Loaded default data file: %s", self.data_file)
            except Exception as e:
                log.warning("ml-pandas: Failed to load default data file: %s", str(e))
                self.data = None
        
        # Parse column configurations
        self.selected_columns = self._parse_columns(self.data_file_columns)
        self.target_col = self.target_column.strip() if self.target_column else None
        self.categorical_cols = self._parse_columns(self.categorical_columns)
        self.numerical_cols = self._parse_columns(self.numerical_columns)

        # Generate and store the agent description
        self._generate_agent_description()

        # Log prominent startup message
        log.info("=" * 80)
        log.info("ml-pandas: 📊 ML PANDAS AGENT (v%s) STARTED SUCCESSFULLY", __version__)
        log.info("=" * 80)
        log.info("ml-pandas: Agent Name: %s", self.agent_name)
        if self.data_file:
            log.info("ml-pandas: Default Data File: %s", self.data_file)
        else:
            log.info("ml-pandas: Default Data File: None (collaborative mode enabled)")
        log.info("ml-pandas: Data Format: %s", self.data_file_format)
        if self.data is not None:
            log.info("ml-pandas: Data Shape: %s", self.data.shape)
        else:
            log.info("ml-pandas: Data Shape: No data loaded yet")
        log.info("ml-pandas: Collaborative Mode: %s", "Enabled" if self.collaborative_mode else "Disabled")
        log.info("ml-pandas: Available Actions: %s", [action.__name__ for action in self.actions])
        if self.target_col:
            log.info("ml-pandas: Target Column: %s", self.target_col)
        if self.selected_columns:
            log.info("ml-pandas: Selected Columns: %s", len(self.selected_columns))
        log.info("ml-pandas: Output Directory: %s", self.output_directory)
        log.info("=" * 80)
        log.info("ml-pandas: ✅ ML Pandas Agent is ready for collaborative data analysis!")
        log.info("ml-pandas: 🔍 Agent should be available in SAM as 'ml_pandas'")
        log.info("ml-pandas: 🤝 Use 'load_data' to receive data from other agents")
        log.info("ml-pandas: 📊 Use 'summarize_data' for quick data summaries")
        log.info("=" * 80)
        
        # Also print to stdout for immediate visibility
        print("=" * 80)
        print(f"📊 ML PANDAS AGENT (v{__version__}) STARTED SUCCESSFULLY")
        print("=" * 80)
        print(f"Agent Name: {self.agent_name}")
        if self.data_file:
            print(f"Default Data File: {self.data_file}")
        else:
            print("Default Data File: None (collaborative mode enabled)")
        if self.data is not None:
            print(f"Data Shape: {self.data.shape}")
        else:
            print("Data Shape: No data loaded yet")
        print(f"Collaborative Mode: {'Enabled' if self.collaborative_mode else 'Disabled'}")
        print(f"Available Actions: {[action.__name__ for action in self.actions]}")
        print("=" * 80)
        print("✅ ML Pandas Agent is ready for collaborative data analysis!")
        print("🔍 Agent should be available in SAM as 'ml_pandas'")
        print("🤝 Use 'load_data' to receive data from other agents")
        print("📊 Use 'summarize_data' for quick data summaries")
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
            log.info("ml-pandas: Created output directory: %s", self.output_directory)
        except Exception as e:
            log.warning("ml-pandas: Failed to create output directory: %s", str(e))

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
            
            log.info("ml-pandas: Successfully loaded data with shape: %s", data.shape)
            return data
        except Exception as e:
            raise ValueError(f"ml-pandas: Failed to load data from {self.data_file}: {str(e)}")

    def _parse_columns(self, columns_str: str) -> List[str]:
        """Parse comma-separated column string into list."""
        if not columns_str or not columns_str.strip():
            return []
        return [col.strip() for col in columns_str.split(",") if col.strip()]

    def _generate_agent_description(self):
        """Generate and store the agent description."""
        description = f"ML Pandas agent for flexible data analysis and machine learning.\n\n"
        
        if self.data is not None:
            description += f"Current data source: {self.current_data_source}\n"
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
        else:
            description += "No data currently loaded.\n"
            description += "Use the 'load_data' action to receive data from other agents or load from files.\n"
            description += "Use the 'summarize_data' action for quick summaries once data is loaded.\n\n"
            description += "**Collaborative Workflow:**\n"
            description += "- Use `load_type: 'json_data'` to receive data from SQL agents\n"
            description += "- Avoid `amfs://` file references - use direct JSON data transfer\n"
            description += "- Example: SQL agent → JSON data → ML pandas agent for analysis"

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
        """Get the currently loaded working data.
        
        Returns:
            DataFrame containing the current data
            
        Raises:
            ValueError: If no data is loaded
        """
        if self.data is None:
            raise ValueError("No data is currently loaded. Use the 'load_data' action to receive data from other agents or load from files.")
        return self.data

    def get_data_summary(self) -> Dict[str, Any]:
        """Get a summary of the currently loaded data.
        
        Returns:
            Dictionary containing data summary information
        """
        if self.data is None:
            return {
                "status": "no_data",
                "message": "No data is currently loaded. Use the 'load_data' action to receive data from other agents or load from files."
            }
        
        return {
            "status": "loaded",
            "shape": self.data.shape,
            "columns": self.data.columns.tolist(),
            "data_types": self.data.dtypes.to_dict(),
            "source": self.current_data_source,
            "history": self.data_history
        }

    def get_collaborative_workflow_help(self) -> str:
        """Get help information for collaborative workflows.
        
        Returns:
            String containing help information
        """
        help_text = """
**Collaborative Workflow Help**

To work with data from other agents (like SQL Database agent):

1. **Receive Data from SQL Agent (Recommended):**
   - Use `load_type: "sql_agent_data"`
   - Pass the SQL agent JSON response as `json_data` parameter
   - SQL agent should use `return_json_data: true` parameter
   - Example: SQL agent returns structured JSON → ML pandas agent loads it

2. **Receive Generic JSON Data:**
   - Use `load_type: "json_data"`
   - Pass any JSON data as `json_data` parameter
   - Example: Any agent returns JSON data → ML pandas agent loads it

3. **Avoid File References:**
   - Don't use `amfs://` URLs or file references
   - Use direct data transfer via JSON

4. **Example Workflow:**
   ```
   SQL Agent: "Get sales data for last 3 months" with return_json_data=true
   → Returns structured JSON with query info and data
   ML Pandas Agent: load_type="sql_agent_data", json_data="[SQL_AGENT_JSON_RESPONSE]"
   → Loads and analyzes the data with full context
   ```

5. **Available Actions:**
   - `load_data`: Receive data from other agents
   - `summarize_data`: Quick data summaries
   - `query_data`: Filter and query data
   - `data_analysis`: Detailed analysis
   - `simple_ml`: Machine learning tasks
        """
        return help_text.strip()

    def load_data_from_file(self, file_path: str, file_format: str = None) -> pd.DataFrame:
        """Load data from a file dynamically."""
        try:
            if not os.path.exists(file_path):
                raise ValueError(f"Data file not found: {file_path}")
            
            format_to_use = file_format or self.data_file_format
            
            if format_to_use.lower() == "csv":
                data = pd.read_csv(file_path)
            elif format_to_use.lower() == "json":
                data = pd.read_json(file_path)
            elif format_to_use.lower() in ["excel", "xlsx"]:
                data = pd.read_excel(file_path)
            elif format_to_use.lower() == "parquet":
                data = pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported data format: {format_to_use}")
            
            self.data = data
            self.current_data_source = f"File: {file_path}"
            self.data_history.append({"source": "file", "path": file_path, "shape": data.shape})
            log.info("ml-pandas: Successfully loaded data from %s with shape: %s", file_path, data.shape)
            return data
        except Exception as e:
            raise ValueError(f"ml-pandas: Failed to load data from {file_path}: {str(e)}")

    def receive_data_from_agent(self, data: pd.DataFrame, source_agent: str, description: str = None) -> pd.DataFrame:
        """Receive data from another agent in collaborative workflows."""
        try:
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Data must be a pandas DataFrame")
            
            self.data = data
            self.current_data_source = f"Agent: {source_agent}"
            if description:
                self.current_data_source += f" - {description}"
            
            self.data_history.append({
                "source": "agent", 
                "agent": source_agent, 
                "description": description,
                "shape": data.shape
            })
            
            log.info("ml-pandas: Successfully received data from agent %s with shape: %s", source_agent, data.shape)
            return data
        except Exception as e:
            raise ValueError(f"ml-pandas: Failed to receive data from agent {source_agent}: {str(e)}")

    def receive_data_from_json(self, json_data: str, source_agent: str = None, description: str = None) -> pd.DataFrame:
        """Receive data as JSON string from another agent or source."""
        try:
            import json
            
            # Validate input
            if not json_data or not json_data.strip():
                raise ValueError("JSON data is empty or not provided")
            
            # Try to parse JSON
            try:
                data_dict = json.loads(json_data)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON format: {str(e)}")
            
            # Handle different JSON formats
            if isinstance(data_dict, list):
                # List of records
                if not data_dict:
                    raise ValueError("JSON data contains an empty list")
                data = pd.DataFrame(data_dict)
            elif isinstance(data_dict, dict):
                if "data" in data_dict and isinstance(data_dict["data"], list):
                    # {"data": [...]} format
                    if not data_dict["data"]:
                        raise ValueError("JSON data contains an empty data array")
                    data = pd.DataFrame(data_dict["data"])
                elif "records" in data_dict and isinstance(data_dict["records"], list):
                    # {"records": [...]} format
                    if not data_dict["records"]:
                        raise ValueError("JSON data contains an empty records array")
                    data = pd.DataFrame(data_dict["records"])
                elif "results" in data_dict and isinstance(data_dict["results"], list):
                    # {"results": [...]} format
                    if not data_dict["results"]:
                        raise ValueError("JSON data contains an empty results array")
                    data = pd.DataFrame(data_dict["results"])
                else:
                    # Single record or other format
                    data = pd.DataFrame([data_dict])
            else:
                raise ValueError(f"Invalid JSON data format. Expected list or dict, got {type(data_dict)}")
            
            # Validate DataFrame
            if data.empty:
                raise ValueError("JSON data resulted in an empty DataFrame")
            
            self.data = data
            self.current_data_source = f"JSON from {source_agent or 'external source'}"
            if description:
                self.current_data_source += f" - {description}"
            
            self.data_history.append({
                "source": "json", 
                "agent": source_agent, 
                "description": description,
                "shape": data.shape
            })
            
            log.info("ml-pandas: Successfully loaded data from JSON with shape: %s", data.shape)
            return data
        except Exception as e:
            raise ValueError(f"ml-pandas: Failed to load data from JSON: {str(e)}")

    def receive_data_from_sql_agent(self, sql_json_data: str, source_agent: str = None, description: str = None) -> pd.DataFrame:
        """Receive data from SQL agent in the new JSON format.
        
        Args:
            sql_json_data: JSON string from SQL agent with return_json_data=True
            source_agent: Name of the SQL agent
            description: Description of the data
            
        Returns:
            DataFrame containing the extracted data
        """
        try:
            import json
            
            # Parse the SQL agent JSON response
            sql_response = json.loads(sql_json_data)
            
            # Extract data from all successful queries
            all_data = []
            for result in sql_response.get("results", []):
                data_records = result.get("data", [])
                all_data.extend(data_records)
            
            if not all_data:
                raise ValueError("No data found in SQL agent response")
            
            # Convert to DataFrame
            data = pd.DataFrame(all_data)
            
            # Store metadata
            self.data = data
            self.current_data_source = f"SQL Agent: {source_agent or 'unknown'}"
            if description:
                self.current_data_source += f" - {description}"
            
            # Add to history
            self.data_history.append({
                "source": "sql_agent", 
                "agent": source_agent, 
                "description": description,
                "shape": data.shape,
                "query": sql_response.get("query", ""),
                "total_queries": sql_response.get("total_queries", 0),
                "total_records": len(all_data)
            })
            
            log.info("ml-pandas: Successfully loaded data from SQL agent with shape: %s, total records: %d", 
                    data.shape, len(all_data))
            return data
            
        except Exception as e:
            raise ValueError(f"ml-pandas: Failed to load data from SQL agent: {str(e)}")

    def get_data_history(self) -> List[Dict[str, Any]]:
        """Get the history of data sources used in this session."""
        return self.data_history.copy()

    def clear_data(self) -> None:
        """Clear the current data and reset data source."""
        self.data = None
        self.current_data_source = None
        log.info("ml-pandas: Data cleared")

    def filter_data(self, query: str) -> pd.DataFrame:
        """Filter data based on a query string."""
        if self.data is None:
            raise ValueError("No data loaded. Please use the 'load_data' action to load data first.")
        
        try:
            # Use pandas query method for filtering
            filtered_data = self.data.query(query)
            log.info("ml-pandas: Filtered data from %d to %d rows using query: %s", 
                    len(self.data), len(filtered_data), query)
            return filtered_data
        except Exception as e:
            raise ValueError(f"ml-pandas: Failed to filter data with query '{query}': {str(e)}")