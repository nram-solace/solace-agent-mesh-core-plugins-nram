"""SQL Database agent component for handling database operations."""

import copy
from typing import Dict, Any, Optional, List
import yaml

from solace_ai_connector.common.log import log
from solace_agent_mesh.agents.base_agent_component import (
    agent_info,
    BaseAgentComponent,
)

from .services.database_service import (
    DatabaseService,
    MySQLService,
    PostgresService,
    SQLiteService,
    MSSQLService,
)
from .actions.search_query import SearchQuery

# Import version
try:
    from ... import __version__
except ImportError:
    __version__ = "0.0.0+local.unknown"  # Fallback version


info = copy.deepcopy(agent_info)
info.update(
    {
        "agent_name": "sql_database",
        "class_name": "SQLDatabaseAgentComponent",
        "description": "SQL Database agent for executing natural language queries against SQL databases",
        "config_parameters": [
            {
                "name": "agent_name",
                "required": True,
                "description": "Name of this SQL database agent instance",
                "type": "string",
            },
            {
                "name": "db_type",
                "required": True,
                "description": "Database type (mysql, mssql, postgres, or sqlite )",
                "type": "string",
            },
            {
                "name": "host",
                "required": False,
                "description": "Database host (for MySQL and PostgreSQL)",
                "type": "string",
            },
            {
                "name": "port",
                "required": False,
                "description": "Database port (for MySQL and PostgreSQL)",
                "type": "integer",
            },
            {
                "name": "user",
                "required": False,
                "description": "Database user (for MySQL and PostgreSQL)",
                "type": "string",
            },
            {
                "name": "password",
                "required": False,
                "description": "Database password (for MySQL and PostgreSQL)",
                "type": "string",
            },
            {
                "name": "database",
                "required": True,
                "description": "Database name (or file path for SQLite)",
                "type": "string",
            },
            {
                "name": "query_timeout",
                "required": False,
                "description": "Query timeout in seconds",
                "type": "integer",
                "default": 30,
            },
            {
                "name": "database_purpose",
                "required": True,
                "description": "Purpose of the database",
                "type": "string",
            },
            {
                "name": "data_description",
                "required": False,
                "description": "Detailed description of the data held in the database. Will be auto-detected if not provided.",
                "type": "string",
            },
            {
                "name": "auto_detect_schema",
                "required": False,
                "description": "Automatically create a schema based on the database structure",
                "type": "boolean",
                "default": True,
            },
            {
                "name": "database_schema",
                "required": False,
                "description": "Database schema if auto_detect_schema is False",
                "type": "string",
            },
            {
                "name": "schema_summary",
                "required": False,
                "description": "Summary of the database schema if auto_detect_schema is False. Will be used in agent description.",
                "type": "string",
            },
            {
                "name": "query_examples",
                "required": False,
                "description": "Natural language to SQL query examples to help the agent understand how to query the database. Format: List of objects with 'natural_language' and 'sql_query' keys. Will be attached to the schema when auto_detect_schema is False.",
                "type": "list",
            },
            {
                "name": "csv_files",
                "required": False,
                "description": "List of CSV files to import as tables on startup",
                "type": "list",
            },
            {
                "name": "csv_directories",
                "required": False,
                "description": "List of directories to scan for CSV files to import as tables on startup",
                "type": "list",
            },
            {
                "name": "response_guidelines",
                "required": False,
                "description": "Guidelines to be attached to action responses. These will be included in the response message.",
                "type": "string",
            }
        ],
    }
)


class SQLDatabaseAgentComponent(BaseAgentComponent):
    """Component for handling SQL database operations."""

    info = info
    actions = [SearchQuery]

    def __init__(self, module_info: Dict[str, Any] = None, **kwargs):
        """Initialize the SQL Database agent component.

        Args:
            module_info: Optional module configuration.
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If required database configuration is missing.
        """
        module_info = module_info or info
        
        # Debug broker request/response configuration
        log.info("sql-db: Initializing SQL agent component")
        log.debug("sql-db: Available kwargs keys: %s", list(kwargs.keys()))
        
        # Log component_config if present
        if 'component_config' in kwargs:
            log.debug("sql-db: component_config keys: %s", list(kwargs['component_config'].keys()))
        
        super().__init__(module_info, **kwargs)

        # Debug broker request/response after initialization
        if hasattr(self, 'broker_request_response') and self.broker_request_response:
            log.info("sql-db: Broker request/response is properly initialized")
        else:
            log.warning("sql-db: Broker request/response is NOT initialized after super().__init__")
            
        # Test the is_broker_request_response_enabled method
        if self.is_broker_request_response_enabled():
            log.info("sql-db: Broker request/response is enabled according to is_broker_request_response_enabled()")
        else:
            log.warning("sql-db: Broker request/response is NOT enabled according to is_broker_request_response_enabled()")

        self.agent_name = self.get_config("agent_name")
        self.db_type = self.get_config("db_type")
        self.database_purpose = self.get_config("database_purpose")
        self.data_description = self.get_config("data_description")
        self.auto_detect_schema = self.get_config("auto_detect_schema", True)
        self.query_timeout = self.get_config("query_timeout", 30)
        self.response_guidelines = self.get_config("response_guidelines", "")

        self.action_list.fix_scopes("<agent_name>", self.agent_name)
        module_info["agent_name"] = self.agent_name

        # Initialize database handler
        self.db_handler = self._create_db_handler()
        log.info("sql-db: Database handler created successfully for %s database", self.db_type)

        # Import any configured CSV files
        csv_files = self.get_config("csv_files", [])
        csv_directories = self.get_config("csv_directories", [])
        if csv_files or csv_directories:
            log.info("sql-db: Importing CSV files - Files: %s, Directories: %s", 
                    len(csv_files) if csv_files else 0, len(csv_directories) if csv_directories else 0)
            try:
                self.db_handler.import_csv_files(csv_files, csv_directories)
                log.info("sql-db: CSV import completed successfully")
            except Exception as e:
                log.error("sql-db: Error importing CSV files: %s", str(e))
        else:
            log.info("sql-db: No CSV files configured for import")

        # Get schema information
        if self.auto_detect_schema:
            try:
                log.info("sql-db: Auto-detecting database schema...")
                self.detailed_schema = self._detect_schema()
                log.info("sql-db: Schema detection completed - Found %d tables", len(self.detailed_schema))
                for table_name, table_info in self.detailed_schema.items():
                    column_count = len(table_info.get("columns", {}))
                    log.debug("sql-db: Table '%s' has %d columns", table_name, column_count)
            except Exception as e:
                log.error("sql-db: Failed to auto-detect schema: %s", str(e))
                self.detailed_schema = {}
        else:
            log.info("sql-db: Schema auto-detection disabled")
            self.detailed_schema = {}

        # Clean the schema before converting to YAML
        schema_dict_cleaned = self._clean_schema(self.detailed_schema)
        # Convert dictionary to YAML string
        self.detailed_schema = yaml.dump(schema_dict_cleaned, default_flow_style=False, allow_unicode=True)
        # Generate schema prompt from detected schema
        self.schema_summary = self._get_schema_summary()
        if not self.schema_summary:
            raise ValueError("Failed to generate schema summary from auto-detected schema")
        
        # Update the search_query action with schema information (simplified for prompt length)
        for action in self.action_list.actions:
            if action.name == "search_query":
                current_directive = action._prompt_directive
                # Use a more concise schema summary to avoid prompt length issues
                schema_info = f"\n\nDatabase Tables: {self.schema_summary}"
                action._prompt_directive = current_directive + schema_info
                break

        # Generate and store the agent description
        self._generate_agent_description()

        # Log prominent startup message
        log.info("=" * 80)
        log.info("sql-db: 🗄️  SQL DATABASE AGENT (v%s) STARTED SUCCESSFULLY", __version__)
        log.info("=" * 80)
        log.info("sql-db: Agent Name: %s", self.agent_name)
        log.info("sql-db: Database Type: %s", self.db_type)
        log.info("sql-db: Database Name: %s", self.get_config("database"))
        if self.get_config("host"):
            log.info("sql-db: Database Host: %s", self.get_config("host"))
        if self.get_config("port"):
            log.info("sql-db: Database Port: %s", self.get_config("port"))
        log.info("sql-db: Available Actions: %s", [action.__name__ for action in self.actions])
        log.info("sql-db: Auto Detect Schema: %s", self.auto_detect_schema)
        log.info("sql-db: Query Timeout: %s seconds", self.query_timeout)
        if self.database_purpose:
            log.info("sql-db: Database Purpose: %s", self.database_purpose)
        log.info("=" * 80)
        log.info("sql-db: ✅ SQL Database Agent is ready for database operations!")
        log.info("sql-db: 🔍 Agent should be available in SAM as 'sql_database'")
        log.info("=" * 80)
        
        # Also print to stdout for immediate visibility
        print("=" * 80)
        print(f"🗄️  SQL DATABASE AGENT (v{__version__}) STARTED SUCCESSFULLY")
        print("=" * 80)
        print(f"Agent Name: {self.agent_name}")
        print(f"Database Type: {self.db_type}")
        print(f"Database Name: {self.get_config('database')}")
        print(f"Available Actions: {[action.__name__ for action in self.actions]}")
        print("=" * 80)
        print("✅ SQL Database Agent is ready for database operations!")
        print("🔍 Agent should be available in SAM as 'sql_database'")
        print("=" * 80)

    def _create_db_handler(self) -> DatabaseService:
        """Create appropriate database handler based on configuration.
        
        Returns:
            Database service instance
            
        Raises:
            ValueError: If database configuration is invalid
        """
        connection_params = {
            "database": self.get_config("database"),
        }

        if self.db_type in ("mysql", "mssql", "postgres"):
            # Add connection parameters needed for MySQL/PostgreSQL
            connection_params.update({
                "host": self.get_config("host"),
                "port": self.get_config("port"),
                "user": self.get_config("user"),
                "password": self.get_config("password"),
            })

        if self.db_type == "mysql":
            return MySQLService(connection_params, query_timeout=self.query_timeout)
        if self.db_type == "mssql":
            return MSSQLService(connection_params, query_timeout=self.query_timeout)
        elif self.db_type == "postgres":
            return PostgresService(connection_params, query_timeout=self.query_timeout)
        elif self.db_type in ("sqlite", "sqlite3"):
            return SQLiteService(connection_params, query_timeout=self.query_timeout)
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")

    def _detect_schema(self) -> Dict[str, Any]:
        """Detect database schema including tables, columns, relationships and sample data.
        
        Returns:
            Dictionary containing detailed schema information
        """
        schema = {}
        tables = self.db_handler.get_tables()
        
        for table in tables:
            table_info = {
                "columns": {},
                "primary_keys": self.db_handler.get_primary_keys(table),
                "foreign_keys": self.db_handler.get_foreign_keys(table),
                "indexes": self.db_handler.get_indexes(table)
            }
            
            # Get detailed column information
            columns = self.db_handler.get_columns(table)
            for col in columns:
                col_name = col["name"]
                table_info["columns"][col_name] = {
                    "type": str(col["type"]),
                    "nullable": col.get("nullable", True),
                }
                
                # Get sample values and statistics for the column
                try:
                    unique_values = self.db_handler.get_unique_values(table, col_name)
                    if unique_values:
                        # Convert any decimal.Decimal objects to strings for YAML serialization
                        cleaned_values = []
                        for value in unique_values:
                            if hasattr(value, 'as_tuple'):  # Check if it's a Decimal
                                cleaned_values.append(str(value))
                            else:
                                cleaned_values.append(value)
                        table_info["columns"][col_name]["sample_values"] = cleaned_values

                    stats = self.db_handler.get_column_stats(table, col_name)
                    if stats:
                        # Convert any decimal.Decimal objects to strings for YAML serialization
                        cleaned_stats = {}
                        for key, value in stats.items():
                            if hasattr(value, 'as_tuple'):  # Check if it's a Decimal
                                cleaned_stats[key] = str(value)
                            else:
                                cleaned_stats[key] = value
                        table_info["columns"][col_name]["statistics"] = cleaned_stats
                except Exception:
                    # Skip sample data if there's an error
                    pass

            schema[table] = table_info

        # Clean the schema to remove any problematic fields
        return self._clean_schema(schema)

    def _clean_schema(self, schema_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Clean the schema dictionary by removing problematic fields and converting unsupported types.
        
        Args:
            schema_dict: The schema dictionary to clean
            
        Returns:
            Cleaned schema dictionary
        """
        def clean_value(value):
            """Recursively clean values to ensure YAML serialization compatibility."""
            if hasattr(value, 'as_tuple'):  # decimal.Decimal
                return str(value)
            elif hasattr(value, '__dict__') and not isinstance(value, (str, int, float, bool, list, dict)):
                # Convert custom objects to string representation
                return str(value)
            elif isinstance(value, (list, tuple)):
                return [clean_value(item) for item in value]
            elif isinstance(value, dict):
                return {k: clean_value(v) for k, v in value.items()}
            else:
                return value
        
        cleaned_schema = {}
        for table, table_data in schema_dict.items():
            cleaned_table_data = {}
            for key, value in table_data.items():
                if key == "columns":
                    cleaned_columns = {}
                    for column, column_data in value.items():
                        cleaned_column_data = {}
                        for col_key, col_value in column_data.items():
                            if col_key in ["statistics", "sample_values"]:
                                cleaned_column_data[col_key] = clean_value(col_value)
                            else:
                                cleaned_column_data[col_key] = col_value
                        cleaned_columns[column] = cleaned_column_data
                    cleaned_table_data[key] = cleaned_columns
                else:
                    cleaned_table_data[key] = clean_value(value)
            cleaned_schema[table] = cleaned_table_data
        
        return cleaned_schema

    def _get_schema_summary(self) -> str:
        """Gets a terse formatted summary of the database schema.

        Returns:
            A string with a one-line summary of each table and its columns.
        """
        if not self.detailed_schema:
            return "Schema information not available."

        try:
            schema_dict = yaml.safe_load(self.detailed_schema)  # Convert YAML to dictionary
            if not isinstance(schema_dict, dict):
                raise ValueError("Error: Parsed schema is not a valid dictionary.")

        except yaml.YAMLError as exc:
            raise ValueError(f"Error: Failed to parse schema. Invalid YAML format. Details: {exc}") from exc

        # Construct summary lines (limited to first 10 columns per table to keep prompt short)
        summary_lines = []
        for table_name, table_info in schema_dict.items():
            columns = table_info.get("columns")
            if isinstance(columns, dict):
                column_names = list(columns.keys())
                # Limit to first 10 columns to keep prompt length manageable
                if len(column_names) > 10:
                    column_summary = ', '.join(column_names[:10]) + f" (+{len(column_names)-10} more)"
                else:
                    column_summary = ', '.join(column_names)
                summary_lines.append(f"{table_name}: {column_summary}")

        return "\n".join(summary_lines)

    def _generate_agent_description(self):
        """Generate and store the agent description."""
        description = f"This agent provides read-only access to a {self.db_type} database.\n\n"

        if self.database_purpose:
            description += f"Purpose:\n{self.database_purpose}\n\n"

        if self.data_description:
            description += f"Data Description:\n{self.data_description}\n"
        
        # Extract table information if schema exists
        try:
            schema_dict = yaml.safe_load(self.detailed_schema)
            if isinstance(schema_dict, dict) and schema_dict:
                tables = list(schema_dict.keys())
                description += f"Contains {len(tables)} tables: {', '.join(tables)}\n"
        except yaml.YAMLError:
            pass  # Silently fail if YAML parsing fails

        self._agent_description = {
            "agent_name": self.agent_name,
            "description": description.strip(),
            "always_open": self.info.get("always_open", False),
            "actions": self.get_actions_summary(),
        }

    def get_agent_summary(self):
        """Get a summary of the agent's capabilities."""
        return self._agent_description
    
    def get_db_handler(self) -> DatabaseService:
        """Get the database handler instance."""
        return self.db_handler

    def is_broker_request_response_enabled(self) -> bool:
        """Check if broker request/response is enabled for this agent.
        
        Returns:
            True if broker request/response is enabled, False otherwise.
        """
        # Check if broker_request_response attribute exists and is not None
        if hasattr(self, 'broker_request_response') and self.broker_request_response is not None:
            return True
        
        # Check if it's enabled in the configuration
        try:
            broker_config = self.get_config("broker_request_response")
            if broker_config and isinstance(broker_config, dict):
                return broker_config.get("enabled", False)
        except:
            pass
        
        # Check if it's in the component_config
        try:
            component_config = getattr(self, 'component_config', {})
            if isinstance(component_config, dict) and 'broker_request_response' in component_config:
                broker_config = component_config['broker_request_response']
                if isinstance(broker_config, dict):
                    return broker_config.get("enabled", False)
        except:
            pass
        
        # Check for new app-level request_reply_enabled in broker config
        try:
            broker_config = self.get_config("broker_config")
            if broker_config and isinstance(broker_config, dict):
                return broker_config.get("request_reply_enabled", False)
        except:
            pass
        
        # Check for request_reply_enabled in shared broker config
        try:
            # This might be available through the framework's broker configuration
            if hasattr(self, 'broker_config') and self.broker_config:
                return self.broker_config.get("request_reply_enabled", False)
        except:
            pass
        
        return False


