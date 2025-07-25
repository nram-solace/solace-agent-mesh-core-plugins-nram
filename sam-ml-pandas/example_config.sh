#!/bin/bash

# Example configuration for sam-ml-pandas agent
# This shows how to configure environment variables for your data

# Replace with your agent instance name (should match what you create with `sam add agent`)
AGENT_NAME="my_ml_pandas_agent"

# Convert agent name to uppercase with underscores for environment variables
AGENT_PREFIX=$(echo "${AGENT_NAME}" | tr '[:lower:]' '[:upper:]' | tr '-' '_')

# Required: Path to your data file
export ${AGENT_PREFIX}_DATA_FILE="/path/to/your/data.csv"

# Optional: Data file format (default: csv)
export ${AGENT_PREFIX}_DATA_FILE_FORMAT="csv"  # csv, json, excel, parquet

# Optional: Specific columns to use (comma-separated, empty for all)
export ${AGENT_PREFIX}_DATA_FILE_COLUMNS="feature1,feature2,feature3,target"

# Optional: Target column for ML tasks
export ${AGENT_PREFIX}_TARGET_COLUMN="target"

# Optional: Specify categorical columns
export ${AGENT_PREFIX}_CATEGORICAL_COLUMNS="category1,category2"

# Optional: Specify numerical columns  
export ${AGENT_PREFIX}_NUMERICAL_COLUMNS="num1,num2,num3"

# Optional: Output directory for results (default: ./ml_pandas_output)
export ${AGENT_PREFIX}_OUTPUT_DIRECTORY="./ml_results"

# Optional: Maximum rows to display in results (default: 100)
export ${AGENT_PREFIX}_MAX_ROWS_DISPLAY="50"

echo "Environment configured for agent: ${AGENT_NAME}"
echo "Environment prefix: ${AGENT_PREFIX}"
echo "Data file: ${!AGENT_PREFIX}_DATA_FILE"

# Example for your specific use case (ccgold data):
echo ""
echo "Example for ccgold data:"
echo "export CCGOLD_ML_PANDAS_DATA_FILE=\"/opt/sam/ccgoldminer/data/final_data.csv\""
echo "export CCGOLD_ML_PANDAS_TARGET_COLUMN=\"target\""
echo "export CCGOLD_ML_PANDAS_DATA_FILE_FORMAT=\"csv\""
echo "export CCGOLD_ML_PANDAS_OUTPUT_DIRECTORY=\"./ccgold_ml_analysis\""