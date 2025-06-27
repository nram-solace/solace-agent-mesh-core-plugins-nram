"""Service for handling SQL database operations."""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import List, Dict, Any, Generator, Optional
import sqlalchemy as sa
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import inspect, text

from solace_ai_connector.common.log import log

from .csv_import_service import CsvImportService


class DatabaseService(ABC):
    """Abstract base class for database services."""

    def __init__(self, connection_params: Dict[str, Any], query_timeout: int = 30):
        """Initialize the database service.
        
        Args:
            connection_params: Database connection parameters
            query_timeout: Query timeout in seconds
        """
        self.connection_params = connection_params
        self.query_timeout = query_timeout
        self.engine = self._create_engine()
        self.csv_import_service = CsvImportService(self.engine)

    def import_csv_files(self, files: Optional[List[str]] = None,
                        directories: Optional[List[str]] = None) -> None:
        """Import CSV files into database tables.
        
        Args:
            files: List of CSV file paths
            directories: List of directory paths containing CSV files
        """
        self.csv_import_service.import_csv_files(files, directories)

    @abstractmethod
    def _create_engine(self) -> Engine:
        """Create SQLAlchemy engine for database connection.
        
        Returns:
            SQLAlchemy Engine instance
        """
        pass

    @contextmanager
    def get_connection(self) -> Generator[sa.Connection, None, None]:
        """Get a database connection from the pool.
        
        Yields:
            Active database connection
            
        Raises:
            SQLAlchemyError: If connection fails
        """
        try:
            connection = self.engine.connect()
            yield connection
        except SQLAlchemyError as e:
            log.error("Database connection error: %s", str(e), exc_info=True)
            raise
        finally:
            connection.close()

    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute a SQL query.
        
        Args:
            query: SQL query to execute
            
        Returns:
            List of dictionaries containing query results
            
        Raises:
            SQLAlchemyError: If query execution fails
        """
        try:
            with self.get_connection() as conn:
                result = conn.execute(text(query))
                return list(result.mappings())
        except SQLAlchemyError as e:
            log.error("Query execution error: %s", str(e), exc_info=True)
            raise

    def get_tables(self) -> List[str]:
        """Get all table names in the database.
        
        Returns:
            List of table names
        """
        inspector = inspect(self.engine)
        return inspector.get_table_names()

    def get_columns(self, table: str) -> List[Dict[str, Any]]:
        """Get detailed column information for a table.
        
        Args:
            table: Table name
            
        Returns:
            List of column details including name, type, nullable, etc.
        """
        inspector = inspect(self.engine)
        all_columns = inspector.get_columns(table)
        
        # Filter out unsupported column types
        supported_columns = []
        for column in all_columns:
            if self._is_column_supported_for_distinct(column["type"], column["name"]):
                supported_columns.append(column)
            else:
                # Log that we're skipping unsupported column types
                log.debug(f"Skipping column '{column['name']}' with unsupported type '{column['type']}' for schema detection")
        
        return supported_columns

    def get_primary_keys(self, table: str) -> List[str]:
        """Get primary key columns for a table.
        
        Args:
            table: Table name
            
        Returns:
            List of primary key column names
        """
        inspector = inspect(self.engine)
        pk_constraint = inspector.get_pk_constraint(table)
        return pk_constraint['constrained_columns'] if pk_constraint else []

    def get_foreign_keys(self, table: str) -> List[Dict[str, Any]]:
        """Get foreign key relationships for a table.
        
        Args:
            table: Table name
            
        Returns:
            List of foreign key details
        """
        inspector = inspect(self.engine)
        return inspector.get_foreign_keys(table)

    def get_indexes(self, table: str) -> List[Dict[str, Any]]:
        """Get indexes for a table.
        
        Args:
            table: Table name
            
        Returns:
            List of index details
        """
        inspector = inspect(self.engine)
        return inspector.get_indexes(table)

    def _is_column_supported_for_distinct(self, column_type: str, column_name: str = None) -> bool:
        """Check if a column type supports DISTINCT operations.
        
        Args:
            column_type: SQLAlchemy column type as string
            column_name: Optional column name for additional checks
            
        Returns:
            True if the column type supports DISTINCT, False otherwise
        """
        # MSSQL geography and geometry types don't support DISTINCT
        unsupported_types = [
            'geography', 'geometry', 'hierarchyid', 'sql_variant',
            'timestamp', 'rowversion'
        ]
        
        # Handle both string and SQLAlchemy type objects
        if hasattr(column_type, 'name'):
            # SQLAlchemy type object
            type_name = column_type.name.lower()
        else:
            # String representation
            type_name = str(column_type).lower()
        
        # Check if any unsupported type is in the type name
        for unsupported_type in unsupported_types:
            if unsupported_type in type_name:
                return False
        
        # Additional check for SQLAlchemy generic types that might be unrecognized
        if hasattr(column_type, '__class__') and 'Type' in str(column_type.__class__):
            # This might be a generic/unrecognized type, be more cautious
            type_str = str(column_type).lower()
            if any(unsupported_type in type_str for unsupported_type in unsupported_types):
                return False
        
        # Additional check based on column name (for cases where SQLAlchemy doesn't recognize the type)
        if column_name:
            column_name_lower = column_name.lower()
            # Common column names that might be geography/geometry types
            spatial_indicators = ['location', 'geom', 'geometry', 'geography', 'shape', 'spatial', 'coord']
            if any(indicator in column_name_lower for indicator in spatial_indicators):
                # If the type is unrecognized and column name suggests spatial data, be cautious
                if 'Type' in str(column_type.__class__) or 'UNKNOWN' in str(column_type).upper():
                    return False
        
        return True

    def get_unique_values(self, table: str, column: str, limit: int = 3) -> List[Any]:
        """Get sample of unique values from a column.
        
        Args:
            table: Table name
            column: Column name
            limit: Maximum number of values to return
            
        Returns:
            List of unique values
        """
        # Check if the column type supports DISTINCT operations
        columns = self.get_columns(table)
        column_info = next((col for col in columns if col["name"] == column), None)
        
        if column_info and not self._is_column_supported_for_distinct(column_info["type"], column_info["name"]):
            # For unsupported types, just get a few sample values without DISTINCT
            if self.engine.name == 'mysql':
                query = f"SELECT {column} FROM {table} WHERE {column} IS NOT NULL ORDER BY RAND() LIMIT {limit}"
            elif self.engine.name == 'postgresql':
                query = f"SELECT {column} FROM {table} WHERE {column} IS NOT NULL ORDER BY RANDOM() LIMIT {limit}"
            elif self.engine.name == 'mssql':
                query = f"SELECT TOP {limit} {column} FROM {table} WHERE {column} IS NOT NULL ORDER BY NEWID()"
            else:
                query = f"SELECT {column} FROM {table} WHERE {column} IS NOT NULL ORDER BY RANDOM() LIMIT {limit}"
        else:
            # Use DISTINCT for supported types
            if self.engine.name == 'mysql':
                # MySQL uses RAND() instead of RANDOM()
                query = f"SELECT DISTINCT {column} FROM {table} WHERE {column} IS NOT NULL ORDER BY RAND() LIMIT {limit}"
            elif self.engine.name == 'postgresql':
                # PostgreSQL requires DISTINCT ON when using ORDER BY
                query = f"SELECT DISTINCT ON ({column}) {column} FROM {table} WHERE {column} IS NOT NULL ORDER BY {column}, RANDOM() LIMIT {limit}"
            elif self.engine.name == 'mssql':
                # MSSQL: Use TOP with subquery to get random distinct values
                query = f"""
                SELECT TOP {limit} {column} 
                FROM (
                    SELECT DISTINCT {column} 
                    FROM {table} 
                    WHERE {column} IS NOT NULL
                ) AS distinct_values 
                ORDER BY NEWID()
                """
            else:
                # SQLite uses RANDOM()
                query = f"SELECT DISTINCT {column} FROM {table} WHERE {column} IS NOT NULL ORDER BY RANDOM() LIMIT {limit}"
        
        results = self.execute_query(query)
        return [row[column] for row in results]

    def get_column_stats(self, table: str, column: str) -> Dict[str, Any]:
        """Get basic statistics for a column.
        
        Args:
            table: Table name
            column: Column name
            
        Returns:
            Dictionary of statistics (min, max, avg, etc.)
        """
        query = f"""
            SELECT 
                COUNT(*) as count,
                COUNT(DISTINCT {column}) as unique_count,
                MIN({column}) as min_value,
                MAX({column}) as max_value
            FROM {table}
            WHERE {column} IS NOT NULL
        """
        results = self.execute_query(query)
        return results[0] if results else {}


class MySQLService(DatabaseService):
    """MySQL database service implementation."""

    def _create_engine(self) -> Engine:
        """Create MySQL database engine."""
        connection_url = sa.URL.create(
            "mysql+mysqlconnector",
            username=self.connection_params.get("user"),
            password=self.connection_params.get("password"),
            host=self.connection_params.get("host"),
            port=self.connection_params.get("port"),
            database=self.connection_params.get("database"),
        )
        
        return sa.create_engine(
            connection_url,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800,
            pool_pre_ping=True,
            connect_args={"connect_timeout": self.query_timeout}
        )


class PostgresService(DatabaseService):
    """PostgreSQL database service implementation."""
    
    def _create_engine(self) -> Engine:
        """Create PostgreSQL database engine."""
        connection_url = sa.URL.create(
            "postgresql+psycopg2",
            username=self.connection_params.get("user"),
            password=self.connection_params.get("password"),
            host=self.connection_params.get("host"),
            port=self.connection_params.get("port"),
            database=self.connection_params.get("database"),
        )
        
        return sa.create_engine(
            connection_url,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800,
            pool_pre_ping=True,
            connect_args={"connect_timeout": self.query_timeout}
        )


class SQLiteService(DatabaseService):
    """SQLite database service implementation."""
    
    def _create_engine(self) -> Engine:
        """Create SQLite database engine."""
        connection_url = sa.URL.create(
            "sqlite",
            database=self.connection_params.get("database")
        )
        
        return sa.create_engine(
            connection_url,
            pool_size=1,  # SQLite doesn't support concurrent connections
            pool_recycle=1800,
            pool_pre_ping=True,
            connect_args={"timeout": self.query_timeout}
        )


class MSSQLService(DatabaseService):
    """Microsoft SQL Server database service implementation."""
    
    def _create_engine(self) -> Engine:
        """Create MSSQL database engine."""
        connection_url = sa.URL.create(
            "mssql+pyodbc",
            username=self.connection_params.get("user"),
            password=self.connection_params.get("password"),
            host=self.connection_params.get("host"),
            port=self.connection_params.get("port"),
            database=self.connection_params.get("database"),
            query={
                "driver": "ODBC Driver 17 for SQL Server",
                "TrustServerCertificate": "yes"
            }
        )
        
        return sa.create_engine(
            connection_url,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800,
            pool_pre_ping=True,
            connect_args={"timeout": self.query_timeout}
        )
