# ML Datasets Agent Plugin (sam-ml-datasets)

A Solace Agent Mesh plugin that provides access to various machine learning datasets for training, analysis, and experimentation. **Designed for collaborative workflows with other ML agents!**

## Features

### Dataset Collections

- **Sklearn Datasets**: Classic machine learning datasets from scikit-learn
  - `iris`, `wine`, `breast_cancer`, `digits`, `diabetes`, `linnerud`
  - Includes both regression and classification datasets
  - Comes with detailed metadata and feature descriptions

- **Seaborn Datasets**: Real-world datasets commonly used in statistical analysis
  - `tips`, `flights`, `titanic`, `car_crashes`, `mpg`, `diamonds`
  - `penguins`, `planets`, `taxis`, `fmri`, `anagrams`, `anscombe`
  - Great for exploratory data analysis and visualization

- **Synthetic Datasets**: Programmatically generated datasets for testing
  - `classification`: Multi-class classification problems
  - `regression`: Linear and non-linear regression problems  
  - `clustering`: Blob, moons, and circles datasets for clustering
  - Customizable parameters (number of features, samples, noise level, etc.)

### Key Capabilities

- **Configurable Record Limits**: Default 100 records per dataset for efficiency
- **Multiple Output Formats**: JSON, YAML, and CSV support
- **Rich Metadata**: Includes dataset descriptions, feature names, and statistics
- **Integration Ready**: Designed to work with `sam-ml-pandas` for analysis
- **Efficient Delivery**: Includes data previews and file attachments
- **Collaborative Workflows**: Seamlessly provides data to other agents

## Installation

```bash
# Install the plugin
solace-agent-mesh plugin add sam-ml-datasets --pip -u git+https://github.com/SolaceLabs/solace-agent-mesh-core-plugins#subdirectory=sam-ml-datasets

# Create an agent instance
solace-agent-mesh add agent my_datasets --copy-from sam_ml_datasets:ml_datasets
```

## Configuration

Configure the agent using environment variables. For an agent named `my_datasets`:

```bash
# Optional: Configure dataset limits and features
export MY_DATASETS_DEFAULT_MAX_RECORDS=100    # Default maximum records per dataset
export MY_DATASETS_ENABLE_SKLEARN=true        # Enable sklearn datasets
export MY_DATASETS_ENABLE_SEABORN=true        # Enable seaborn datasets  
export MY_DATASETS_ENABLE_SYNTHETIC=true      # Enable synthetic dataset generation
```

## Collaborative Workflows

This agent is designed to work with other agents in collaborative workflows:

### Example: ML Datasets Agent + ML Pandas Agent

1. **ML Datasets Agent provides data:**
   ```
   "Get the iris dataset for analysis"
   ```

2. **ML Pandas Agent receives and analyzes:**
   ```
   "Load the iris dataset and perform EDA"
   ```

3. **ML Pandas Agent creates visualizations:**
   ```
   "Create correlation heatmap and feature distributions"
   ```

### Example: ML Datasets Agent + SQL Agent + ML Pandas Agent

1. **ML Datasets Agent provides synthetic data:**
   ```
   "Generate synthetic classification data with 6 features"
   ```

2. **SQL Agent queries the data:**
   ```
   "Analyze the synthetic dataset for patterns"
   ```

3. **ML Pandas Agent performs ML:**
   ```
   "Train a classification model on the synthetic data"
   ```

## Usage

### Dataset Retrieval Actions

1. **Get Specific Datasets:**
   ```
   "Get sklearn iris dataset"
   "Retrieve seaborn tips dataset with 200 records"
   "Generate synthetic classification data"
   ```

2. **Get Datasets with Custom Parameters:**
   ```
   "Generate synthetic data with 6 features and 3 classes"
   "Create regression dataset with noise level 0.2"
   ```

3. **Get Datasets in Different Formats:**
   ```
   "Get iris dataset in CSV format"
   "Export tips dataset as YAML with metadata"
   ```

4. **Get Data Directly for Agent-to-Agent Communication:**
   ```
   "Get seaborn flights dataset with data included in response"
   "Get sklearn iris dataset in JSON format with full data in message"
   ```

### Dataset Discovery Actions

1. **List All Available Datasets:**
   ```
   "Show me all available datasets"
   "List sklearn datasets only"
   "What synthetic datasets are available?"
   ```

2. **Get Dataset Information:**
   ```
   "What datasets are available for classification?"
   "Show me real-world datasets for analysis"
   ```

## Dataset Details

### Sklearn Datasets

| Dataset | Type | Samples | Features | Description |
|---------|------|---------|----------|-------------|
| iris | Classification | 150 | 4 | Classic flower classification |
| wine | Classification | 178 | 13 | Wine quality classification |
| breast_cancer | Classification | 569 | 30 | Cancer diagnosis |
| digits | Classification | 1797 | 64 | Handwritten digit recognition |
| diabetes | Regression | 442 | 10 | Diabetes progression |
| linnerud | Multivariate | 20 | 3+3 | Physical exercise data |

### Seaborn Datasets

| Dataset | Samples | Features | Description |
|---------|---------|----------|-------------|
| tips | 244 | 7 | Restaurant tip data |
| flights | 144 | 3 | Airline passenger numbers |
| titanic | 891 | 15 | Titanic passenger data |
| car_crashes | 51 | 5 | US state car crash data |
| mpg | 398 | 9 | Car fuel efficiency |
| penguins | 344 | 8 | Palmer penguins data |

### Synthetic Dataset Options

For synthetic datasets, you can customize:

- `n_samples`: Number of samples to generate
- `n_features`: Number of features
- `n_classes`: Number of classes (classification)
- `n_informative`: Number of informative features
- `n_centers`: Number of cluster centers
- `noise`: Noise level (0.0 to 1.0)

## Output Format

All datasets are returned with:

1. **Dataset file**: The actual data in requested format (JSON/YAML/CSV)
2. **Metadata file**: Dataset information and statistics (YAML)
3. **Preview**: First 5 rows shown in the response message
4. **Summary**: Record count, feature count, and dataset description

## Integration with sam-ml-pandas

The ML Datasets agent is designed to work seamlessly with `sam-ml-pandas` for analysis:

```bash
# In a multi-agent conversation:
# 1. Get a dataset with data included in response (recommended for agent-to-agent)
get_dataset dataset_type=sklearn dataset_name=iris include_data_in_message=true response_format=json

# 2. Ask sam-ml-pandas to analyze it
@ml_pandas analyze the iris dataset I just retrieved, show summary statistics and create visualizations

# Alternative: Get dataset with file attachment (may have file system issues)
get_dataset dataset_type=seaborn dataset_name=flights
```

## Configuration Options

The agent supports these configuration parameters:

- `agent_name` (required): Name of the agent instance
- `default_max_records` (optional): Default record limit (default: 100)
- `enable_sklearn_datasets` (optional): Enable sklearn collection (default: true)
- `enable_seaborn_datasets` (optional): Enable seaborn collection (default: true)  
- `enable_synthetic_datasets` (optional): Enable synthetic generation (default: true)

## Error Handling

The agent provides clear error messages for:
- Invalid dataset types or names
- Unsupported parameters for synthetic datasets
- Missing required parameters
- Data generation failures

## Development

### Building the Plugin

```bash
cd sam-ml-datasets
hatch build
```

### Running Tests

```bash
python -m pytest tests/ -v
```

## Requirements

- **Python**: 3.11+ (optimized for Python 3.12)
- **Dependencies**:
  - scikit-learn >= 1.3.0
  - seaborn >= 0.12.0
  - pandas >= 2.0.0
  - numpy >= 1.24.0

## License

This plugin is part of the Solace Agent Mesh Core Plugins collection and is licensed under the Apache 2.0 License.