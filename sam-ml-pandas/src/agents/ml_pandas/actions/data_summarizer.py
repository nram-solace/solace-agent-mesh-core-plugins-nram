"""Data summarizer action for quick data summaries in collaborative workflows."""

from typing import Dict, Any
import pandas as pd
import numpy as np

from solace_agent_mesh.common.action import Action
from solace_agent_mesh.common.action_response import ActionResponse, ErrorInfo
from solace_ai_connector.common.log import log


class DataSummarizerAction(Action):
    """Action for providing quick data summaries in collaborative workflows."""

    def __init__(self, **kwargs):
        """Initialize the action."""
        super().__init__(
            {
                "name": "summarize_data",
                "prompt_directive": (
                    "Provide quick, focused summaries of data received from other agents. "
                    "This action is designed for collaborative workflows where you receive data "
                    "from SQL agents, database agents, or other sources and need to summarize it quickly. "
                    "Examples: 'Summarize sales data', 'Get key metrics from customer data'"
                ),
                "params": [
                    {
                        "name": "summary_type",
                        "desc": "Type of summary to generate",
                        "type": "string",
                        "required": True,
                        "enum": ["overview", "key_metrics", "trends", "comparison", "custom"],
                    },
                    {
                        "name": "focus_columns",
                        "desc": "Comma-separated list of columns to focus on (empty for all)",
                        "type": "string",
                        "required": False,
                    },
                    {
                        "name": "group_by",
                        "desc": "Column to group by for comparison summaries",
                        "type": "string",
                        "required": False,
                    },
                    {
                        "name": "time_column",
                        "desc": "Column containing time/date data for trend analysis",
                        "type": "string",
                        "required": False,
                    },
                    {
                        "name": "custom_metrics",
                        "desc": "Comma-separated list of custom metrics to calculate",
                        "type": "string",
                        "required": False,
                    },
                    {
                        "name": "include_visualization",
                        "desc": "Whether to include a simple visualization",
                        "type": "boolean",
                        "required": False,
                        "default": False,
                    },
                ],
            }
        )

    def invoke(self, request_data: Dict[str, Any], meta: Dict[str, Any] = None) -> ActionResponse:
        """Execute the data summarizer action."""
        try:
            # Get parameters
            summary_type = request_data.get("summary_type")
            focus_columns_str = request_data.get("focus_columns", "")
            group_by = request_data.get("group_by", "")
            time_column = request_data.get("time_column", "")
            custom_metrics_str = request_data.get("custom_metrics", "")
            include_visualization = request_data.get("include_visualization", False)

            # Validate summary_type
            valid_summary_types = ["overview", "key_metrics", "trends", "comparison", "custom"]
            if summary_type not in valid_summary_types:
                return ActionResponse(
                    message=f"Invalid summary_type '{summary_type}'. Valid types are: {', '.join(valid_summary_types)}",
                    error_info=ErrorInfo(f"Invalid summary_type '{summary_type}'. Valid types are: {', '.join(valid_summary_types)}")
                )

            # Get the agent and its data
            agent = self.get_agent()
            
            try:
                data = agent.get_working_data()
            except ValueError as e:
                return ActionResponse(
                    message=str(e),
                    error_info=ErrorInfo(str(e))
                )
            
            data_service = agent.get_data_service()

            # Parse focus columns
            focus_columns = [col.strip() for col in focus_columns_str.split(",") if col.strip()] if focus_columns_str else None

            # Parse custom metrics
            custom_metrics = [metric.strip() for metric in custom_metrics_str.split(",") if metric.strip()] if custom_metrics_str else None

            # Validate columns if specified
            if focus_columns:
                missing_cols = [col for col in focus_columns if col not in data.columns]
                if missing_cols:
                    return ActionResponse(
                        message=f"Focus columns not found in dataset: {missing_cols}",
                        error_info=ErrorInfo(f"Focus columns not found in dataset: {missing_cols}")
                    )
                # Filter data to focus columns
                data = data[focus_columns]

            # Generate summary based on type
            if summary_type == "overview":
                result = self._generate_overview_summary(data, agent)
            elif summary_type == "key_metrics":
                result = self._generate_key_metrics_summary(data, agent)
            elif summary_type == "trends":
                result = self._generate_trends_summary(data, time_column, agent)
            elif summary_type == "comparison":
                result = self._generate_comparison_summary(data, group_by, agent)
            elif summary_type == "custom":
                result = self._generate_custom_summary(data, custom_metrics, agent)
            else:
                return ActionResponse(
                    message=f"Unsupported summary type: {summary_type}",
                    error_info=ErrorInfo(f"Unsupported summary type: {summary_type}")
                )

            # Add visualization if requested
            if include_visualization:
                try:
                    plot_base64 = data_service.create_visualization(
                        data, "summary_chart", focus_columns
                    )
                    result["visualization"] = {
                        "type": "summary_chart",
                        "plot_base64": plot_base64
                    }
                except Exception as e:
                    log.warning("Failed to create visualization: %s", str(e))
                    result["visualization"] = {"error": str(e)}

            # Clean result for JSON serialization
            clean_result = data_service.clean_data_for_json(result)

            # Save results to file
            filename = f"data_summary_{summary_type}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
            saved_path = data_service.save_results(clean_result, filename)

            response_text = self._format_response(clean_result, saved_path)

            return ActionResponse(
                message=response_text
            )

        except Exception as e:
            log.error("ml-pandas: Error in data summarizer action: %s", str(e))
            return ActionResponse(
                message=f"Failed to summarize data: {str(e)}",
                error_info=ErrorInfo(f"Failed to summarize data: {str(e)}")
            )

    def _generate_overview_summary(self, data: pd.DataFrame, agent) -> Dict[str, Any]:
        """Generate an overview summary of the data."""
        return {
            "summary_type": "overview",
            "data_shape": data.shape,
            "total_records": len(data),
            "total_columns": len(data.columns),
            "column_info": {
                "numerical": len(data.select_dtypes(include=[np.number]).columns),
                "categorical": len(data.select_dtypes(include=['object', 'category']).columns),
                "datetime": len(data.select_dtypes(include=['datetime']).columns)
            },
            "memory_usage": data.memory_usage(deep=True).sum(),
            "missing_values": data.isnull().sum().to_dict(),
            "sample_data": data.head(3).to_dict('records')
        }

    def _generate_key_metrics_summary(self, data: pd.DataFrame, agent) -> Dict[str, Any]:
        """Generate key metrics summary."""
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        
        metrics = {}
        for col in numerical_cols:
            metrics[col] = {
                "mean": float(data[col].mean()),
                "median": float(data[col].median()),
                "std": float(data[col].std()),
                "min": float(data[col].min()),
                "max": float(data[col].max()),
                "sum": float(data[col].sum())
            }
        
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        categorical_summary = {}
        for col in categorical_cols:
            value_counts = data[col].value_counts()
            categorical_summary[col] = {
                "unique_values": int(value_counts.nunique()),
                "top_values": value_counts.head(5).to_dict()
            }
        
        return {
            "summary_type": "key_metrics",
            "numerical_metrics": metrics,
            "categorical_summary": categorical_summary,
            "total_records": len(data)
        }

    def _generate_trends_summary(self, data: pd.DataFrame, time_column: str, agent) -> Dict[str, Any]:
        """Generate trends summary."""
        if not time_column or time_column not in data.columns:
            return {
                "summary_type": "trends",
                "error": f"Time column '{time_column}' not found or not specified"
            }
        
        try:
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(data[time_column]):
                data[time_column] = pd.to_datetime(data[time_column])
            
            # Sort by time
            data_sorted = data.sort_values(time_column)
            
            # Get time range
            time_range = {
                "start": data_sorted[time_column].min().isoformat(),
                "end": data_sorted[time_column].max().isoformat(),
                "duration_days": (data_sorted[time_column].max() - data_sorted[time_column].min()).days
            }
            
            # Monthly trends for numerical columns
            numerical_cols = data.select_dtypes(include=[np.number]).columns
            monthly_trends = {}
            
            for col in numerical_cols:
                if col != time_column:
                    monthly_data = data_sorted.set_index(time_column)[col].resample('M').agg(['mean', 'sum', 'count'])
                    monthly_trends[col] = monthly_data.to_dict()
            
            return {
                "summary_type": "trends",
                "time_column": time_column,
                "time_range": time_range,
                "monthly_trends": monthly_trends
            }
        except Exception as e:
            return {
                "summary_type": "trends",
                "error": f"Failed to analyze trends: {str(e)}"
            }

    def _generate_comparison_summary(self, data: pd.DataFrame, group_by: str, agent) -> Dict[str, Any]:
        """Generate comparison summary grouped by a column."""
        if not group_by or group_by not in data.columns:
            return {
                "summary_type": "comparison",
                "error": f"Group by column '{group_by}' not found or not specified"
            }
        
        try:
            grouped = data.groupby(group_by)
            numerical_cols = data.select_dtypes(include=[np.number]).columns
            
            comparison_data = {}
            for col in numerical_cols:
                if col != group_by:
                    group_stats = grouped[col].agg(['count', 'mean', 'sum', 'std']).to_dict()
                    comparison_data[col] = group_stats
            
            return {
                "summary_type": "comparison",
                "group_by": group_by,
                "groups": list(grouped.groups.keys()),
                "group_count": len(grouped),
                "comparison_data": comparison_data
            }
        except Exception as e:
            return {
                "summary_type": "comparison",
                "error": f"Failed to generate comparison: {str(e)}"
            }

    def _generate_custom_summary(self, data: pd.DataFrame, custom_metrics: list, agent) -> Dict[str, Any]:
        """Generate custom summary based on specified metrics."""
        if not custom_metrics:
            return {
                "summary_type": "custom",
                "error": "No custom metrics specified"
            }
        
        try:
            custom_results = {}
            for metric in custom_metrics:
                if metric.lower() == "total_records":
                    custom_results[metric] = len(data)
                elif metric.lower() == "unique_values":
                    custom_results[metric] = {col: data[col].nunique() for col in data.columns}
                elif metric.lower() == "missing_percentage":
                    custom_results[metric] = {col: (data[col].isnull().sum() / len(data)) * 100 for col in data.columns}
                else:
                    # Try to apply as a pandas method
                    try:
                        if hasattr(data, metric):
                            custom_results[metric] = getattr(data, metric)()
                        else:
                            custom_results[metric] = f"Unknown metric: {metric}"
                    except Exception:
                        custom_results[metric] = f"Failed to calculate: {metric}"
            
            return {
                "summary_type": "custom",
                "custom_metrics": custom_results
            }
        except Exception as e:
            return {
                "summary_type": "custom",
                "error": f"Failed to generate custom summary: {str(e)}"
            }

    def _format_response(self, result: Dict[str, Any], saved_path: str) -> str:
        """Format the response text."""
        response_lines = [
            f"# Data Summary ({result['summary_type'].title()})",
            ""
        ]

        if "error" in result:
            response_lines.append(f"**Error:** {result['error']}")
        else:
            if result["summary_type"] == "overview":
                response_lines.extend([
                    f"**Data Shape:** {result['data_shape'][0]:,} rows × {result['data_shape'][1]} columns",
                    f"**Total Records:** {result['total_records']:,}",
                    f"**Memory Usage:** {result['memory_usage']:,} bytes",
                    "",
                    "### Column Types:",
                    f"- **Numerical:** {result['column_info']['numerical']}",
                    f"- **Categorical:** {result['column_info']['categorical']}",
                    f"- **Datetime:** {result['column_info']['datetime']}",
                    ""
                ])

            elif result["summary_type"] == "key_metrics":
                response_lines.extend([
                    f"**Total Records:** {result['total_records']:,}",
                    ""
                ])
                
                if "numerical_metrics" in result:
                    response_lines.append("### Numerical Metrics:")
                    for col, metrics in result["numerical_metrics"].items():
                        response_lines.append(f"**{col}:**")
                        response_lines.append(f"  - Mean: {metrics['mean']:,.2f}")
                        response_lines.append(f"  - Median: {metrics['median']:,.2f}")
                        response_lines.append(f"  - Std: {metrics['std']:,.2f}")
                        response_lines.append(f"  - Range: {metrics['min']:,.2f} to {metrics['max']:,.2f}")
                        response_lines.append("")

            elif result["summary_type"] == "trends":
                if "time_range" in result:
                    response_lines.extend([
                        f"**Time Column:** {result['time_column']}",
                        f"**Time Range:** {result['time_range']['start']} to {result['time_range']['end']}",
                        f"**Duration:** {result['time_range']['duration_days']} days",
                        ""
                    ])

            elif result["summary_type"] == "comparison":
                response_lines.extend([
                    f"**Grouped By:** {result['group_by']}",
                    f"**Number of Groups:** {result['group_count']}",
                    f"**Groups:** {', '.join(map(str, result['groups']))}",
                    ""
                ])

            elif result["summary_type"] == "custom":
                response_lines.append("### Custom Metrics:")
                for metric, value in result["custom_metrics"].items():
                    if isinstance(value, dict):
                        response_lines.append(f"**{metric}:**")
                        for key, val in value.items():
                            response_lines.append(f"  - {key}: {val}")
                        response_lines.append("")
                    else:
                        response_lines.append(f"**{metric}:** {value}")

        response_lines.extend([
            "---",
            f"**Results saved to:** `{saved_path}`",
            "",
            "💡 **Collaborative Workflow:** This summary was generated from data received from other agents"
        ])

        return "\n".join(response_lines) 