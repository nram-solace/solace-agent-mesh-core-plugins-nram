# ML Datasets Agent Plugin (sam-ml-datasets)

A Solace Agent Mesh plugin that provides access to various machine learning datasets for training, analysis, and experimentation. This agent serves as a data source for ML workloads and integrates seamlessly with other ML agents like `sam-ml-pandas` for analysis and EDA.

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

## Installation

Install the plugin in your Solace Agent Mesh project:

```bash
solace-agent-mesh plugin add sam-ml-datasets --pip -u git+https://github.com/SolaceLabs/solace-agent-mesh-core-plugins#subdirectory=sam-ml-datasets
```

## Configuration

### Environment Variables

Set these environment variables for your agent instance:

```bash
# Optional: Configure dataset limits and features
MY_DATASETS_DEFAULT_MAX_RECORDS=100    # Default maximum records per dataset
MY_DATASETS_ENABLE_SKLEARN=true        # Enable sklearn datasets
MY_DATASETS_ENABLE_SEABORN=true        # Enable seaborn datasets  
MY_DATASETS_ENABLE_SYNTHETIC=true      # Enable synthetic dataset generation
```

### Create Agent Instance

Create an agent from the plugin template:

```bash
solace-agent-mesh add agent my_datasets --copy-from sam_ml_datasets:ml_datasets
```

## Usage

### Available Actions

#### 1. `get_dataset` - Retrieve a specific dataset

Get a dataset by type and name:

```bash
# Get sklearn iris dataset
get_dataset dataset_type=sklearn dataset_name=iris

# Get seaborn tips dataset with custom record limit
get_dataset dataset_type=seaborn dataset_name=tips max_records=200

# Generate synthetic classification data
get_dataset dataset_type=synthetic dataset_name=classification

# Generate synthetic data with custom parameters
get_dataset dataset_type=synthetic dataset_name=classification synthetic_params='{"n_features": 6, "n_classes": 3, "n_informative": 4}'
```

**Parameters:**
- `dataset_type` (required): "sklearn", "seaborn", or "synthetic"
- `dataset_name` (required): Name of the specific dataset
- `max_records` (optional): Maximum records to return (default: 100)
- `response_format` (optional): "json", "yaml", or "csv" (default: "json")
- `include_metadata` (optional): Include dataset metadata (default: true)
- `synthetic_params` (optional): JSON string with synthetic dataset parameters

#### 2. `list_datasets` - List available datasets

List all available datasets or filter by type:

```bash
# List all datasets
list_datasets

# List only sklearn datasets
list_datasets dataset_type=sklearn

# Get results in JSON format
list_datasets response_format=json
```

**Parameters:**
- `dataset_type` (optional): "sklearn", "seaborn", "synthetic", or "all" (default: "all")
- `response_format` (optional): "json" or "yaml" (default: "yaml")

### Integration with sam-ml-pandas

The ML Datasets agent is designed to work seamlessly with `sam-ml-pandas` for analysis:

```bash
# In a multi-agent conversation:
# 1. Get a dataset
get_dataset dataset_type=sklearn dataset_name=iris

# 2. Ask sam-ml-pandas to analyze it
@ml_pandas analyze the iris dataset I just retrieved, show summary statistics and create visualizations
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

## Configuration Options

The agent supports these configuration parameters:

- `agent_name` (required): Name of the agent instance
- `default_max_records` (optional): Default record limit (default: 100)
- `enable_sklearn_datasets` (optional): Enable sklearn collection (default: true)
- `enable_seaborn_datasets` (optional): Enable seaborn collection (default: true)  
- `enable_synthetic_datasets` (optional): Enable synthetic generation (default: true)

## Output Format

All datasets are returned with:

1. **Dataset file**: The actual data in requested format (JSON/YAML/CSV)
2. **Metadata file**: Dataset information and statistics (YAML)
3. **Preview**: First 5 rows shown in the response message
4. **Summary**: Record count, feature count, and dataset description

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

## Dependencies

- `scikit-learn>=1.3.0` - For sklearn datasets and synthetic generation
- `seaborn>=0.11.0` - For seaborn dataset collection
- `pandas>=1.5.0` - For data manipulation
- `numpy>=1.21.0` - For numerical operations

## License

This plugin is part of the Solace Agent Mesh Core Plugins collection and is licensed under the same terms as the main project.