# SAM SQL Database Plugin

A Solace Agent Mesh plugin that provides natural language to SQL query conversion and execution against multiple database types.

## Supported Database Types

- **MySQL** - Uses `mysql+mysqlconnector` SQLAlchemy dialect
- **PostgreSQL** - Uses `postgresql+psycopg2` SQLAlchemy dialect  
- **Microsoft SQL Server (MSSQL)** - Uses `mssql+pyodbc` SQLAlchemy dialect
- **SQLite** - Uses `sqlite` SQLAlchemy dialect

## Installation

### Prerequisites

Install the required database drivers based on your database type:

```bash
# For MySQL
pip install mysql-connector-python

# For PostgreSQL  
pip install psycopg2-binary

# For Microsoft SQL Server
pip install pyodbc

# For SQLite (usually included with Python)
# No additional installation needed
```

### MSSQL Setup

For Microsoft SQL Server, you may also need to install the Microsoft ODBC Driver for SQL Server on your system:

- **Windows**: Download from Microsoft's website
- **Linux**: Follow Microsoft's installation guide for your distribution
- **macOS**: Use Homebrew: `brew install microsoft/mssql-release/mssql-tools`

The plugin will automatically try different ODBC drivers in this order:
1. ODBC Driver 18 for SQL Server
2. ODBC Driver 17 for SQL Server  
3. ODBC Driver 13 for SQL Server
4. SQL Server (generic)

## Configuration

The plugin supports the following configuration parameters:

- `db_type`: Database type (`mysql`, `postgres`, `mssql`, `sqlite`)
- `host`: Database host (required for MySQL, PostgreSQL, MSSQL)
- `port`: Database port (optional, defaults to standard ports)
- `user`: Database username (required for MySQL, PostgreSQL, MSSQL)
- `password`: Database password (required for MySQL, PostgreSQL, MSSQL)
- `database`: Database name or file path (required for all)
- `query_timeout`: Query timeout in seconds (default: 30)

## Features

- Natural language to SQL query conversion
- Automatic schema detection
- CSV file import capabilities
- Query result formatting in multiple formats (YAML, JSON, CSV, Markdown)
- Support for complex queries with joins, aggregations, and subqueries
- Database statistics and sample data generation

## Usage

The plugin creates an agent that can:
- Execute natural language queries against your database
- Provide schema information and table descriptions
- Import CSV files as database tables
- Generate reports and data summaries

## Example Configuration

```yaml
agent_name: my_sql_agent
db_type: mssql
host: localhost
port: 1433
user: myuser
password: mypassword
database: mydatabase
query_timeout: 30
database_purpose: "Customer relationship management system"
auto_detect_schema: true
``` 