# SAM ML Scikit-Learn Plugin

A comprehensive machine learning plugin for Solace Agent Mesh that provides advanced analytics capabilities using scikit-learn, pandas, and other popular ML libraries.

## Features

### 🔍 Exploratory Data Analysis (EDA)
- **Statistical Analysis**: Descriptive statistics, correlation analysis, distribution plots
- **Data Visualization**: Histograms, box plots, scatter plots, correlation matrices
- **Data Quality Assessment**: Missing value analysis, data type detection, outlier identification
- **Feature Analysis**: Feature importance, feature correlation, dimensionality analysis

### 📊 Regression Analysis
- **Linear Regression**: Simple and multiple linear regression
- **Polynomial Regression**: Non-linear regression with polynomial features
- **Ridge/Lasso Regression**: Regularized regression with feature selection
- **Random Forest Regression**: Ensemble method for regression
- **Support Vector Regression**: SVR with different kernels
- **Model Evaluation**: R² score, MSE, MAE, cross-validation

### 🎯 Classification
- **Logistic Regression**: Binary and multiclass classification
- **Random Forest Classification**: Ensemble classification
- **Support Vector Classification**: SVC with different kernels
- **K-Nearest Neighbors**: KNN classification
- **Naive Bayes**: Gaussian and Multinomial Naive Bayes
- **Model Evaluation**: Accuracy, precision, recall, F1-score, confusion matrix

### 🔍 Outlier Detection
- **Statistical Methods**: Z-score, IQR-based detection
- **Isolation Forest**: Unsupervised outlier detection
- **Local Outlier Factor**: Density-based outlier detection
- **One-Class SVM**: Novelty detection
- **Elliptic Envelope**: Robust covariance-based detection

### 📈 Advanced Analytics
- **Feature Engineering**: Feature scaling, encoding, selection
- **Cross-Validation**: K-fold, stratified, time series split
- **Hyperparameter Tuning**: Grid search, random search
- **Model Persistence**: Save and load trained models
- **Performance Metrics**: Comprehensive evaluation metrics

## Installation

```bash
pip install sam-ml-scikit-learn
```

## Configuration

The plugin can be configured through environment variables or configuration files:

```yaml
# Example configuration
agent_name: ml_analytics
data_source: csv_file
data_path: /path/to/your/data.csv
target_column: target
feature_columns: [feature1, feature2, feature3]
test_size: 0.2
random_state: 42
```

## Usage Examples

### Exploratory Data Analysis
```python
# Perform comprehensive EDA
{
    "action": "exploratory_data_analysis",
    "params": {
        "data_source": "csv",
        "file_path": "data.csv",
        "analysis_type": "comprehensive",
        "include_visualizations": true
    }
}
```

### Regression Analysis
```python
# Train a regression model
{
    "action": "train_regression_model",
    "params": {
        "data_source": "csv",
        "file_path": "data.csv",
        "target_column": "price",
        "feature_columns": ["area", "bedrooms", "bathrooms"],
        "model_type": "random_forest",
        "test_size": 0.2
    }
}
```

### Classification
```python
# Train a classification model
{
    "action": "train_classification_model",
    "params": {
        "data_source": "csv",
        "file_path": "data.csv",
        "target_column": "species",
        "feature_columns": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "model_type": "random_forest",
        "test_size": 0.2
    }
}
```

### Outlier Detection
```python
# Detect outliers
{
    "action": "detect_outliers",
    "params": {
        "data_source": "csv",
        "file_path": "data.csv",
        "method": "isolation_forest",
        "contamination": 0.1
    }
}
```

## Supported Data Sources

- **CSV Files**: Direct file path or uploaded files
- **Pandas DataFrames**: In-memory data structures
- **Database Connections**: SQL queries via pandas
- **API Data**: JSON/XML data from REST APIs

## Model Persistence

Trained models can be saved and loaded for later use:

```python
# Save model
{
    "action": "save_model",
    "params": {
        "model_name": "my_regression_model",
        "file_path": "/path/to/save/model.pkl"
    }
}

# Load model
{
    "action": "load_model",
    "params": {
        "model_name": "my_regression_model",
        "file_path": "/path/to/load/model.pkl"
    }
}
```

## Visualization

The plugin generates various types of visualizations:

- **Distribution Plots**: Histograms, density plots, box plots
- **Correlation Plots**: Heatmaps, scatter matrices
- **Model Performance**: Learning curves, ROC curves, precision-recall curves
- **Feature Importance**: Bar charts, permutation importance plots

## Error Handling

The plugin includes comprehensive error handling for:

- Invalid data formats
- Missing required columns
- Insufficient data for training
- Model convergence issues
- Memory constraints

## Performance Optimization

- **Parallel Processing**: Utilizes joblib for parallel computations
- **Memory Management**: Efficient data handling for large datasets
- **Caching**: Model and computation caching for repeated operations
- **Batch Processing**: Support for large datasets through batch processing

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is licensed under the same license as the Solace Agent Mesh framework. 