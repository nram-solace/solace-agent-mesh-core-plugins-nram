"""Data service for ML Pandas agent."""

import os
import json
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import base64
import io

from solace_ai_connector.common.log import log


class DataService:
    """Service for data operations and utilities."""

    def __init__(self, output_directory: str = "./ml_pandas_output", max_rows_display: int = 100):
        """Initialize the data service.
        
        Args:
            output_directory: Directory to save outputs
            max_rows_display: Maximum rows to display in results
        """
        self.output_directory = output_directory
        self.max_rows_display = max_rows_display
        
        # Ensure output directory exists
        Path(output_directory).mkdir(parents=True, exist_ok=True)

    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get a comprehensive summary of the dataset."""
        summary = {
            "shape": {
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1])
            },
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        }
        
        # Add basic statistics for numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            # Convert describe() output to a clean dictionary format
            desc_stats = df[numerical_cols].describe()
            numerical_stats = {}
            for col in desc_stats.columns:
                numerical_stats[col] = {}
                for stat in desc_stats.index:
                    numerical_stats[col][stat] = float(desc_stats.loc[stat, col])
            summary["numerical_stats"] = numerical_stats
        
        # Add categorical column info
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            summary["categorical_info"] = {}
            for col in categorical_cols:
                unique_count = df[col].nunique()
                summary["categorical_info"][col] = {
                    "unique_values": int(unique_count),
                    "most_frequent": df[col].mode().iloc[0] if not df[col].empty else None
                }
                # Add sample values for small unique counts
                if unique_count <= 10:
                    summary["categorical_info"][col]["unique_list"] = df[col].unique().tolist()
        
        return summary

    def get_data_preview(self, df: pd.DataFrame, n_rows: int = None) -> Dict[str, Any]:
        """Get a preview of the data."""
        if n_rows is None:
            n_rows = min(self.max_rows_display, len(df))
        
        preview = {
            "head": df.head(n_rows).to_dict('records'),
            "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
            "columns": df.columns.tolist()
        }
        
        return preview

    def analyze_missing_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing data patterns."""
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        analysis = {
            "total_missing": int(missing_counts.sum()),
            "missing_by_column": {
                col: {
                    "count": int(count),
                    "percentage": float(missing_percentages[col])
                }
                for col, count in missing_counts.items()
                if count > 0
            },
            "complete_rows": int(df.dropna().shape[0]),
            "rows_with_missing": int(df.shape[0] - df.dropna().shape[0])
        }
        
        return analysis

    def get_correlation_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get correlation analysis for numerical columns."""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) < 2:
            return {"error": "Need at least 2 numerical columns for correlation analysis"}
        
        corr_matrix = df[numerical_cols].corr()
        
        # Convert correlation matrix to a clean dictionary format
        corr_dict = {}
        for col1 in corr_matrix.columns:
            corr_dict[col1] = {}
            for col2 in corr_matrix.columns:
                corr_dict[col1][col2] = float(corr_matrix.loc[col1, col2])
        
        analysis = {
            "correlation_matrix": corr_dict,
            "high_correlations": []
        }
        
        # Find high correlations (> 0.7 or < -0.7)
        for i, col1 in enumerate(numerical_cols):
            for j, col2 in enumerate(numerical_cols):
                if i < j:  # Avoid duplicates and self-correlation
                    corr_val = corr_matrix.loc[col1, col2]
                    if abs(corr_val) > 0.7:
                        analysis["high_correlations"].append({
                            "column1": col1,
                            "column2": col2,
                            "correlation": float(corr_val)
                        })
        
        return analysis

    def create_visualization(self, df: pd.DataFrame, viz_type: str, columns: List[str] = None, 
                           save_plot: bool = True) -> str:
        """Create a visualization and return the plot as base64 string."""
        plt.figure(figsize=(10, 6))
        
        try:
            if viz_type == "histogram":
                if columns:
                    for col in columns:
                        if col in df.columns and df[col].dtype in ['int64', 'float64']:
                            plt.hist(df[col].dropna(), alpha=0.7, label=col, bins=30)
                    plt.legend()
                else:
                    # Plot histogram for all numerical columns
                    numerical_cols = df.select_dtypes(include=[np.number]).columns[:4]  # Limit to 4
                    for col in numerical_cols:
                        plt.hist(df[col].dropna(), alpha=0.7, label=col, bins=30)
                    plt.legend()
                plt.title("Distribution of Numerical Columns")
                plt.xlabel("Values")
                plt.ylabel("Frequency")
            
            elif viz_type == "correlation_heatmap":
                numerical_cols = df.select_dtypes(include=[np.number]).columns
                if len(numerical_cols) < 2:
                    raise ValueError("Need at least 2 numerical columns for correlation heatmap")
                
                corr_matrix = df[numerical_cols].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                           square=True, linewidths=0.5)
                plt.title("Correlation Heatmap")
            
            elif viz_type == "boxplot":
                if columns:
                    df[columns].boxplot()
                else:
                    numerical_cols = df.select_dtypes(include=[np.number]).columns[:6]  # Limit to 6
                    if len(numerical_cols) > 0:
                        df[numerical_cols].boxplot()
                plt.title("Box Plot Distribution")
                plt.xticks(rotation=45)
            
            elif viz_type == "scatter":
                if columns and len(columns) >= 2:
                    plt.scatter(df[columns[0]], df[columns[1]], alpha=0.6)
                    plt.xlabel(columns[0])
                    plt.ylabel(columns[1])
                    plt.title(f"Scatter Plot: {columns[0]} vs {columns[1]}")
                else:
                    numerical_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numerical_cols) >= 2:
                        plt.scatter(df[numerical_cols[0]], df[numerical_cols[1]], alpha=0.6)
                        plt.xlabel(numerical_cols[0])
                        plt.ylabel(numerical_cols[1])
                        plt.title(f"Scatter Plot: {numerical_cols[0]} vs {numerical_cols[1]}")
            
            plt.tight_layout()
            
            # Save plot if requested
            if save_plot:
                plot_path = os.path.join(self.output_directory, f"{viz_type}_plot.png")
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                log.info("Plot saved to: %s", plot_path)
            
            # Convert plot to base64 string
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            plot_data = buffer.getvalue()
            buffer.close()
            plt.close()
            
            return base64.b64encode(plot_data).decode('utf-8')
            
        except Exception as e:
            plt.close()
            log.error("Error creating visualization: %s", str(e))
            return f"Error creating {viz_type} visualization: {str(e)}"

    def save_results(self, results: Dict[str, Any], filename: str) -> str:
        """Save results to a JSON file."""
        try:
            # Ensure results are properly cleaned for JSON serialization
            cleaned_results = self.clean_data_for_json(results)
            
            filepath = os.path.join(self.output_directory, filename)
            with open(filepath, 'w') as f:
                json.dump(cleaned_results, f, indent=2, default=str)
            log.info("ml-pandas: Results saved to: %s", filepath)
            return filepath
        except TypeError as e:
            log.error("ml-pandas: Error saving results - JSON serialization failed: %s", str(e))
            # Try to identify the problematic data
            try:
                json.dumps(results, default=str)
            except TypeError as e2:
                log.error("ml-pandas: JSON serialization error details: %s", str(e2))
            return f"Error saving results - JSON serialization failed: {str(e)}"
        except Exception as e:
            log.error("Error saving results: %s", str(e))
            return f"Error saving results: {str(e)}"

    def clean_data_for_json(self, obj: Any) -> Any:
        """Clean data for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, dict):
            # Handle tuple keys by converting them to strings
            cleaned_dict = {}
            for k, v in obj.items():
                if isinstance(k, tuple):
                    # Convert tuple key to string representation
                    cleaned_key = str(k)
                else:
                    cleaned_key = k
                cleaned_dict[cleaned_key] = self.clean_data_for_json(v)
            return cleaned_dict
        elif isinstance(obj, list):
            return [self.clean_data_for_json(item) for item in obj]
        elif isinstance(obj, tuple):
            # Convert tuples to lists
            return list(obj)
        elif isinstance(obj, (pd.Timestamp, pd.DatetimeTZDtype)):
            # Handle pandas timestamp objects
            return str(obj)
        elif hasattr(obj, 'dtype') and hasattr(obj, 'tolist'):
            # Handle other numpy/pandas objects
            return obj.tolist()
        else:
            return obj