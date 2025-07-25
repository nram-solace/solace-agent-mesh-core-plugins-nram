# SAM ML Pandas Plugin

A simple machine learning and exploratory data analysis (EDA) agent for Solace Agent Mesh using pandas and scikit-learn.

## Features

- **Exploratory Data Analysis (EDA)**
  - Data summary statistics
  - Missing value analysis  
  - Correlation analysis
  - Data visualization (histograms, heatmaps, boxplots, scatter plots)
  - Data preview

- **Simple Machine Learning**
  - Classification and regression tasks
  - Automatic task type detection
  - Multiple model types (Random Forest, Linear/Logistic Regression)
  - Automatic preprocessing (missing values, categorical encoding, scaling)
  - Model performance metrics
  - Feature importance analysis

## Installation

```bash
# Install the plugin
solace-agent-mesh plugin add sam-ml-pandas --pip -u git+https://github.com/SolaceLabs/solace-agent-mesh-core-plugins#subdirectory=sam-ml-pandas

# Create an agent instance
solace-agent-mesh add agent my_ml_agent --copy-from sam_ml_pandas:ml_pandas
```

## Configuration

Configure the agent using environment variables. For an agent named `my_ml_agent`:

```bash
# Required: Path to your data file
export MY_ML_AGENT_DATA_FILE="/path/to/your/data.csv"

# Optional: Data file format (default: csv)
export MY_ML_AGENT_DATA_FILE_FORMAT="csv"  # csv, json, excel, parquet

# Optional: Specific columns to use (comma-separated, empty for all)
export MY_ML_AGENT_DATA_FILE_COLUMNS="col1,col2,col3"

# Optional: Target column for ML tasks
export MY_ML_AGENT_TARGET_COLUMN="target"

# Optional: Specify categorical columns
export MY_ML_AGENT_CATEGORICAL_COLUMNS="category1,category2"

# Optional: Specify numerical columns  
export MY_ML_AGENT_NUMERICAL_COLUMNS="num1,num2,num3"

# Optional: Output directory for results
export MY_ML_AGENT_OUTPUT_DIRECTORY="./ml_results"

# Optional: Maximum rows to display in results (default: 100)
export MY_ML_AGENT_MAX_ROWS_DISPLAY="50"
```

## Usage

### Data Analysis Actions

1. **Complete Analysis** - Get full EDA report:
   ```
   "Please perform a complete data analysis"
   ```

2. **Data Summary** - Basic statistics and info:
   ```
   "Show me a summary of the data"
   ```

3. **Missing Data Analysis**:
   ```
   "Analyze missing values in the dataset"  
   ```

4. **Correlation Analysis**:
   ```
   "Show correlations between numerical columns"
   ```

5. **Data Visualization**:
   ```
   "Create a histogram visualization"
   "Generate a correlation heatmap"
   "Show boxplots for numerical columns"
   ```

### Machine Learning Actions

1. **Auto ML** - Automatic task detection:
   ```
   "Run machine learning on the target column"
   ```

2. **Classification**:
   ```
   "Perform classification using random forest"
   "Train a logistic regression model"
   ```

3. **Regression**:
   ```
   "Build a regression model to predict the target"
   "Use linear regression for prediction"
   ```

4. **Custom Features**:
   ```
   "Train a model using columns: feature1,feature2,feature3"
   ```

## Supported File Formats

- **CSV** (`.csv`) - Default format
- **JSON** (`.json`) - JSON format
- **Excel** (`.xlsx`, `.xls`) - Excel spreadsheets  
- **Parquet** (`.parquet`) - Parquet format

## Model Types

### Classification
- `random_forest` - Random Forest Classifier (default)
- `logistic` - Logistic Regression

### Regression  
- `random_forest` - Random Forest Regressor (default)
- `linear` - Linear Regression

## Output

All results are saved to the configured output directory with timestamped filenames:

- `data_analysis_*.json` - EDA results
- `ml_*.json` - ML model results
- `*.png` - Generated visualizations

## Example Environment Configuration

```bash
# Example for analyzing sales data
export SALES_ML_AGENT_DATA_FILE="/data/sales_data.csv"
export SALES_ML_AGENT_DATA_FILE_FORMAT="csv"
export SALES_ML_AGENT_TARGET_COLUMN="total_sales"
export SALES_ML_AGENT_CATEGORICAL_COLUMNS="region,product_category,season"
export SALES_ML_AGENT_NUMERICAL_COLUMNS="price,quantity,discount,profit"
export SALES_ML_AGENT_OUTPUT_DIRECTORY="./sales_analysis"
```

## Limitations

- Designed for simple ML tasks and basic EDA
- Automatic preprocessing may not be optimal for all datasets
- Limited to scikit-learn models
- No deep learning capabilities
- No advanced feature engineering

## Requirements

- **Python**: 3.11+ (optimized for Python 3.12)
- **Dependencies**:
  - pandas >= 2.0.0
  - numpy >= 1.24.0
  - scikit-learn >= 1.3.0
  - matplotlib >= 3.7.0
  - seaborn >= 0.12.0
  - plotly >= 5.15.0