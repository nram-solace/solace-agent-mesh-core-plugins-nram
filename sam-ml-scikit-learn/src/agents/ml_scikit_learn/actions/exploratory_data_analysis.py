"""Action for performing exploratory data analysis."""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
import json
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64
import io

from solace_agent_mesh.common.action import Action
from solace_agent_mesh.common.action_response import (
    ActionResponse,
    ErrorInfo,
    InlineFile,
)

from solace_ai_connector.common.log import log


class ExploratoryDataAnalysis(Action):
    """Action for performing exploratory data analysis."""

    def __init__(self, **kwargs):
        """Initialize the action."""
        super().__init__(
            {
                "name": "exploratory_data_analysis",
                "prompt_directive": (
                    "Perform comprehensive exploratory data analysis (EDA) on the provided dataset. "
                    "This includes statistical analysis, data visualization, data quality assessment, "
                    "and feature analysis. The analysis can be customized based on the specified parameters."
                ),
                "params": [
                    {
                        "name": "data_source",
                        "desc": "Type of data source (csv, excel, json, parquet)",
                        "type": "string",
                        "required": False,
                        "default": "csv",
                    },
                    {
                        "name": "file_path",
                        "desc": "Path to the data file",
                        "type": "string",
                        "required": True,
                    },
                    {
                        "name": "analysis_type",
                        "desc": "Type of analysis (basic, comprehensive, targeted)",
                        "type": "string",
                        "required": False,
                        "default": "comprehensive",
                    },
                    {
                        "name": "target_column",
                        "desc": "Target column for analysis (optional)",
                        "type": "string",
                        "required": False,
                    },
                    {
                        "name": "feature_columns",
                        "desc": "List of feature columns to analyze (optional)",
                        "type": "list",
                        "required": False,
                    },
                    {
                        "name": "include_visualizations",
                        "desc": "Whether to include visualizations in the output",
                        "type": "boolean",
                        "required": False,
                        "default": True,
                    },
                    {
                        "name": "visualization_format",
                        "desc": "Format for visualizations (png, html, json)",
                        "type": "string",
                        "required": False,
                        "default": "png",
                    },
                    {
                        "name": "response_format",
                        "desc": "Format of the response (yaml, json)",
                        "type": "string",
                        "required": False,
                        "default": "yaml",
                    },
                    {
                        "name": "save_visualizations",
                        "desc": "Whether to save visualizations to disk",
                        "type": "boolean",
                        "required": False,
                        "default": False,
                    },
                ],
                "required_scopes": ["<agent_name>:exploratory_data_analysis:execute"],
            },
            **kwargs,
        )

    def invoke(
        self, params: Dict[str, Any], meta: Dict[str, Any] = None
    ) -> ActionResponse:
        """Execute the exploratory data analysis.

        Args:
            params: Action parameters
            meta: Optional metadata

        Returns:
            ActionResponse containing the EDA results
        """
        try:
            # Extract parameters
            data_source = params.get("data_source", "csv")
            file_path = params.get("file_path")
            analysis_type = params.get("analysis_type", "comprehensive")
            target_column = params.get("target_column")
            feature_columns = params.get("feature_columns", [])
            include_visualizations = params.get("include_visualizations", True)
            visualization_format = params.get("visualization_format", "png")
            response_format = params.get("response_format", "yaml")
            save_visualizations = params.get("save_visualizations", False)

            if not file_path:
                raise ValueError("File path is required")

            # Load data
            agent = self.get_agent()
            df = agent.load_data(data_source, file_path)

            # Validate data
            validation_results = agent.validate_data(df, target_column, feature_columns)

            # Perform EDA
            ml_service = agent.get_ml_service()
            eda_results = ml_service.perform_eda(df, target_column, feature_columns)

            # Add validation results to EDA results
            eda_results["validation"] = validation_results

            # Generate visualizations if requested
            visualization_files = []
            if include_visualizations:
                visualization_files = self._generate_visualizations(
                    df, eda_results, target_column, feature_columns, 
                    visualization_format, save_visualizations, agent
                )

            # Format response
            if response_format.lower() == "json":
                response_content = json.dumps(eda_results, indent=2, default=str)
                response_file = InlineFile(
                    filename="eda_results.json",
                    content=response_content,
                    content_type="application/json"
                )
            else:
                response_content = yaml.dump(eda_results, default_flow_style=False, allow_unicode=True)
                response_file = InlineFile(
                    filename="eda_results.yaml",
                    content=response_content,
                    content_type="text/yaml"
                )

            # Create response
            files = [response_file] + visualization_files

            return ActionResponse(
                message=f"Exploratory Data Analysis completed successfully. "
                       f"Dataset shape: {eda_results['basic_info']['shape']}, "
                       f"Missing values: {eda_results['missing_values']['total_missing']}",
                files=files,
                metadata={
                    "analysis_type": analysis_type,
                    "data_shape": eda_results["basic_info"]["shape"],
                    "missing_values": eda_results["missing_values"]["total_missing"],
                    "visualizations_generated": len(visualization_files)
                }
            )

        except Exception as e:
            log.error("Error in exploratory data analysis: %s", str(e))
            return ActionResponse(
                message=f"Error performing exploratory data analysis: {str(e)}",
                error_info=ErrorInfo(str(e)),
            )

    def _generate_visualizations(self, df: pd.DataFrame, eda_results: Dict[str, Any], 
                                target_column: str, feature_columns: List[str],
                                format_type: str, save_to_disk: bool, agent) -> List[InlineFile]:
        """Generate visualizations for EDA.

        Args:
            df: Input DataFrame
            eda_results: EDA results
            target_column: Target column name
            feature_columns: Feature column names
            format_type: Visualization format
            save_to_disk: Whether to save to disk
            agent: Agent instance

        Returns:
            List of visualization files
        """
        visualization_files = []
        
        try:
            # 1. Distribution plots for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                fig = self._create_distribution_plots(df, numeric_cols)
                viz_file = self._save_visualization(
                    fig, "distributions", format_type, save_to_disk, agent
                )
                if viz_file:
                    visualization_files.append(viz_file)

            # 2. Correlation heatmap
            if len(numeric_cols) > 1:
                fig = self._create_correlation_heatmap(df, numeric_cols)
                viz_file = self._save_visualization(
                    fig, "correlation_heatmap", format_type, save_to_disk, agent
                )
                if viz_file:
                    visualization_files.append(viz_file)

            # 3. Missing values plot
            if eda_results["missing_values"]["total_missing"] > 0:
                fig = self._create_missing_values_plot(eda_results["missing_values"])
                viz_file = self._save_visualization(
                    fig, "missing_values", format_type, save_to_disk, agent
                )
                if viz_file:
                    visualization_files.append(viz_file)

            # 4. Target analysis plots
            if target_column and target_column in df.columns:
                fig = self._create_target_analysis_plots(df, target_column)
                viz_file = self._save_visualization(
                    fig, "target_analysis", format_type, save_to_disk, agent
                )
                if viz_file:
                    visualization_files.append(viz_file)

            # 5. Feature relationships (if target is specified)
            if target_column and target_column in df.columns and len(numeric_cols) > 1:
                feature_cols = [col for col in numeric_cols if col != target_column]
                if feature_cols:
                    fig = self._create_feature_relationship_plots(df, target_column, feature_cols[:5])  # Limit to 5 features
                    viz_file = self._save_visualization(
                        fig, "feature_relationships", format_type, save_to_disk, agent
                    )
                    if viz_file:
                        visualization_files.append(viz_file)

        except Exception as e:
            log.warning("Error generating visualizations: %s", str(e))

        return visualization_files

    def _create_distribution_plots(self, df: pd.DataFrame, numeric_cols: List[str]) -> go.Figure:
        """Create distribution plots for numeric columns."""
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=numeric_cols,
            specs=[[{"secondary_y": False}] * n_cols] * n_rows
        )

        for i, col in enumerate(numeric_cols):
            row = i // n_cols + 1
            col_idx = i % n_cols + 1
            
            fig.add_trace(
                go.Histogram(x=df[col], name=col, showlegend=False),
                row=row, col=col_idx
            )

        fig.update_layout(
            title="Distribution of Numeric Features",
            height=300 * n_rows,
            showlegend=False
        )
        
        return fig

    def _create_correlation_heatmap(self, df: pd.DataFrame, numeric_cols: List[str]) -> go.Figure:
        """Create correlation heatmap."""
        corr_matrix = df[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0
        ))
        
        fig.update_layout(
            title="Correlation Heatmap",
            xaxis_title="Features",
            yaxis_title="Features",
            height=600
        )
        
        return fig

    def _create_missing_values_plot(self, missing_data: Dict[str, Any]) -> go.Figure:
        """Create missing values plot."""
        missing_counts = missing_data["missing_counts"]
        missing_percentages = missing_data["missing_percentages"]
        
        # Filter columns with missing values
        cols_with_missing = {k: v for k, v in missing_counts.items() if v > 0}
        
        if not cols_with_missing:
            return None
            
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=list(cols_with_missing.keys()),
            y=list(cols_with_missing.values()),
            name="Missing Count",
            yaxis="y"
        ))
        
        fig.add_trace(go.Scatter(
            x=list(cols_with_missing.keys()),
            y=[missing_percentages[k] for k in cols_with_missing.keys()],
            name="Missing Percentage",
            yaxis="y2"
        ))
        
        fig.update_layout(
            title="Missing Values Analysis",
            xaxis_title="Features",
            yaxis=dict(title="Missing Count", side="left"),
            yaxis2=dict(title="Missing Percentage (%)", side="right", overlaying="y"),
            height=400
        )
        
        return fig

    def _create_target_analysis_plots(self, df: pd.DataFrame, target_column: str) -> go.Figure:
        """Create target analysis plots."""
        target_data = df[target_column]
        
        if target_data.dtype in [np.number]:
            # Numeric target - create histogram
            fig = go.Figure(data=go.Histogram(x=target_data))
            fig.update_layout(
                title=f"Distribution of Target Variable: {target_column}",
                xaxis_title=target_column,
                yaxis_title="Frequency",
                height=400
            )
        else:
            # Categorical target - create bar chart
            value_counts = target_data.value_counts()
            fig = go.Figure(data=go.Bar(x=value_counts.index, y=value_counts.values))
            fig.update_layout(
                title=f"Distribution of Target Variable: {target_column}",
                xaxis_title=target_column,
                yaxis_title="Count",
                height=400
            )
        
        return fig

    def _create_feature_relationship_plots(self, df: pd.DataFrame, target_column: str, 
                                          feature_cols: List[str]) -> go.Figure:
        """Create feature relationship plots."""
        n_features = len(feature_cols)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=[f"{col} vs {target_column}" for col in feature_cols],
            specs=[[{"secondary_y": False}] * n_cols] * n_rows
        )

        for i, feature in enumerate(feature_cols):
            row = i // n_cols + 1
            col_idx = i % n_cols + 1
            
            fig.add_trace(
                go.Scatter(x=df[feature], y=df[target_column], 
                          mode='markers', name=feature, showlegend=False),
                row=row, col=col_idx
            )

        fig.update_layout(
            title=f"Feature Relationships with Target: {target_column}",
            height=300 * n_rows,
            showlegend=False
        )
        
        return fig

    def _save_visualization(self, fig: go.Figure, name: str, format_type: str, 
                           save_to_disk: bool, agent) -> InlineFile:
        """Save visualization to file."""
        try:
            if format_type.lower() == "html":
                content = fig.to_html(include_plotlyjs=True)
                content_type = "text/html"
                filename = f"{name}.html"
            elif format_type.lower() == "json":
                content = fig.to_json()
                content_type = "application/json"
                filename = f"{name}.json"
            else:  # png
                img_bytes = fig.to_image(format="png")
                content = base64.b64encode(img_bytes).decode()
                content_type = "image/png"
                filename = f"{name}.png"

            # Save to disk if requested
            if save_to_disk:
                output_path = Path(agent.visualization_output_path) / filename
                if format_type.lower() == "png":
                    with open(output_path, "wb") as f:
                        f.write(img_bytes)
                else:
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(content)

            return InlineFile(
                filename=filename,
                content=content,
                content_type=content_type
            )

        except Exception as e:
            log.warning("Error saving visualization %s: %s", name, str(e))
            return None 