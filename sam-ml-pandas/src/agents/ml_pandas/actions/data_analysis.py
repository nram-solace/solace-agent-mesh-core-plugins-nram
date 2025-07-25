"""Data analysis action for exploratory data analysis."""

from typing import Dict, Any
import pandas as pd
import json

from solace_agent_mesh.common.action import Action
from solace_agent_mesh.common.action_response import ActionResponse, ErrorInfo
from solace_ai_connector.common.log import log


class DataAnalysisAction(Action):
    """Action for performing exploratory data analysis."""

    def __init__(self, **kwargs):
        """Initialize the action."""
        super().__init__(
            {
                "name": "data_analysis",
                "prompt_directive": (
                    "Perform exploratory data analysis on the loaded dataset. "
                    "This includes data summary, missing value analysis, correlation analysis, "
                    "and basic visualizations. You can specify the type of analysis to perform."
                ),
                "params": [
                    {
                        "name": "analysis_type",
                        "desc": "Type of analysis to perform",
                        "type": "string",
                        "required": True,
                        "enum": ["summary", "missing_data", "correlation", "preview", "visualization", "all"],
                    },
                    {
                        "name": "columns",
                        "desc": "Comma-separated list of specific columns to analyze (optional)",
                        "type": "string",
                        "required": False,
                    },
                    {
                        "name": "visualization_type",
                        "desc": "Type of visualization when analysis_type is 'visualization'",
                        "type": "string",
                        "required": False,
                        "enum": ["histogram", "correlation_heatmap", "boxplot", "scatter"],
                        "default": "histogram",
                    },
                    {
                        "name": "n_rows",
                        "desc": "Number of rows to show in preview (default: agent max_rows_display setting)",
                        "type": "integer",
                        "required": False,
                    },
                ],
            }
        )

    def invoke(self, request_data: Dict[str, Any], meta: Dict[str, Any] = None) -> ActionResponse:
        """Execute the data analysis action."""
        try:
            # Get parameters
            analysis_type = request_data.get("analysis_type", "summary")
            columns_str = request_data.get("columns", "")
            visualization_type = request_data.get("visualization_type", "histogram")
            n_rows = request_data.get("n_rows")

            # Validate analysis_type
            valid_analysis_types = ["summary", "missing_data", "correlation", "preview", "visualization", "all"]
            if analysis_type not in valid_analysis_types:
                return ActionResponse(
                    success=False,
                    error_info=ErrorInfo(f"Invalid analysis_type '{analysis_type}'. Valid types are: {', '.join(valid_analysis_types)}")
                )

            # Parse columns
            columns = [col.strip() for col in columns_str.split(",") if col.strip()] if columns_str else None

            # Get the agent and its data
            agent = self.agent
            data = agent.get_working_data()
            data_service = agent.get_data_service()

            # Validate columns if specified
            if columns:
                missing_cols = [col for col in columns if col not in data.columns]
                if missing_cols:
                    return ActionResponse(
                        success=False,
                        error_info=ErrorInfo(f"Columns not found in dataset: {missing_cols}")
                    )
                # Filter data to specified columns
                data = data[columns]

            result = {"analysis_type": analysis_type, "data_shape": data.shape}

            # Perform the requested analysis
            if analysis_type == "summary" or analysis_type == "all":
                log.info("Performing data summary analysis")
                result["summary"] = data_service.get_data_summary(data)

            if analysis_type == "preview" or analysis_type == "all":
                log.info("Generating data preview")
                result["preview"] = data_service.get_data_preview(data, n_rows)

            if analysis_type == "missing_data" or analysis_type == "all":
                log.info("Analyzing missing data")
                result["missing_data_analysis"] = data_service.analyze_missing_data(data)

            if analysis_type == "correlation" or analysis_type == "all":
                log.info("Performing correlation analysis")
                result["correlation_analysis"] = data_service.get_correlation_analysis(data)

            if analysis_type == "visualization" or analysis_type == "all":
                log.info("Creating visualization: %s", visualization_type)
                plot_base64 = data_service.create_visualization(
                    data, visualization_type, columns
                )
                result["visualization"] = {
                    "type": visualization_type,
                    "plot_base64": plot_base64,
                    "columns_used": columns or "all_numerical"
                }

            # Clean result for JSON serialization
            clean_result = data_service.clean_data_for_json(result)

            # Save results to file
            filename = f"data_analysis_{analysis_type}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
            saved_path = data_service.save_results(clean_result, filename)

            response_text = self._format_response(clean_result, saved_path)

            return ActionResponse(
                success=True,
                response_text=response_text,
                response_data=clean_result
            )

        except Exception as e:
            log.error("Error in data analysis action: %s", str(e))
            return ActionResponse(
                success=False,
                error_info=ErrorInfo(f"Failed to perform data analysis: {str(e)}")
            )

    def _format_response(self, result: Dict[str, Any], saved_path: str) -> str:
        """Format the response text."""
        response_lines = [
            f"# Data Analysis Results ({result['analysis_type']})",
            f"**Data Shape:** {result['data_shape'][0]} rows × {result['data_shape'][1]} columns",
            ""
        ]

        if "summary" in result:
            summary = result["summary"]
            response_lines.extend([
                "## Data Summary",
                f"- **Columns:** {len(summary['columns'])}",
                f"- **Memory Usage:** {summary['memory_usage']}",
                f"- **Data Types:** {len(set(summary['dtypes'].values()))} unique types",
                ""
            ])

            if "numerical_stats" in summary:
                response_lines.extend([
                    "### Numerical Columns Statistics",
                    f"Found {len(summary['numerical_stats'])} numerical columns with basic statistics.",
                    ""
                ])

            if "categorical_info" in summary:
                response_lines.extend([
                    "### Categorical Columns",
                    f"Found {len(summary['categorical_info'])} categorical columns:",
                ])
                for col, info in summary['categorical_info'].items():
                    response_lines.append(f"- **{col}:** {info['unique_values']} unique values")
                response_lines.append("")

        if "missing_data_analysis" in result:
            missing = result["missing_data_analysis"]
            response_lines.extend([
                "## Missing Data Analysis",
                f"- **Total Missing Values:** {missing['total_missing']}",
                f"- **Complete Rows:** {missing['complete_rows']}",
                f"- **Rows with Missing Data:** {missing['rows_with_missing']}",
                ""
            ])

            if missing["missing_by_column"]:
                response_lines.append("### Missing Values by Column:")
                for col, info in missing["missing_by_column"].items():
                    response_lines.append(f"- **{col}:** {info['count']} ({info['percentage']:.1f}%)")
                response_lines.append("")

        if "correlation_analysis" in result:
            corr = result["correlation_analysis"]
            if "error" not in corr:
                response_lines.extend([
                    "## Correlation Analysis",
                    "Correlation matrix computed for numerical columns.",
                    ""
                ])

                if corr["high_correlations"]:
                    response_lines.append("### High Correlations (|r| > 0.7):")
                    for hc in corr["high_correlations"]:
                        response_lines.append(
                            f"- **{hc['column1']}** ↔ **{hc['column2']}:** {hc['correlation']:.3f}"
                        )
                    response_lines.append("")
                else:
                    response_lines.extend([
                        "No high correlations (|r| > 0.7) found between numerical columns.",
                        ""
                    ])

        if "visualization" in result:
            viz = result["visualization"]
            response_lines.extend([
                "## Visualization",
                f"Generated **{viz['type']}** visualization.",
                f"Columns used: {viz['columns_used']}",
                ""
            ])

        if "preview" in result:
            preview = result["preview"]
            response_lines.extend([
                "## Data Preview",
                f"Showing first {len(preview['head'])} rows:",
                ""
            ])

        response_lines.extend([
            "---",
            f"**Results saved to:** `{saved_path}`",
            "",
            "💡 **Available analysis types:** summary, preview, missing_data, correlation, visualization, all",
            "📊 **Available visualizations:** histogram, correlation_heatmap, boxplot, scatter"
        ])

        return "\n".join(response_lines)