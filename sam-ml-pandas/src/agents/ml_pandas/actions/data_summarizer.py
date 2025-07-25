"""Data summarizer action for quick data summaries in collaborative workflows."""

from typing import Dict, Any, List
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
                    "Generate quick summaries of loaded data for collaborative workflows. "
                    "This action provides various summary types including basic statistics, "
                    "detailed analysis, grouped summaries, and simple record counts. "
                    "Useful for getting quick insights without displaying all the data."
                ),
                "params": [
                    {
                        "name": "summary_type",
                        "desc": "Type of summary to generate",
                        "type": "string",
                        "required": False,
                        "enum": ["basic", "detailed", "grouped", "count"],
                        "default": "basic",
                    },
                    {
                        "name": "columns",
                        "desc": "List of columns to analyze (for detailed and grouped summaries)",
                        "type": "array",
                        "required": False,
                        "items": {"type": "string"},
                    },
                    {
                        "name": "group_by",
                        "desc": "List of columns to group by (for grouped summaries)",
                        "type": "array",
                        "required": False,
                        "items": {"type": "string"},
                    },
                    {
                        "name": "filters",
                        "desc": "Filters to apply before summarization",
                        "type": "object",
                        "required": False,
                    },
                    {
                        "name": "max_rows",
                        "desc": "Maximum number of rows to display in summaries",
                        "type": "integer",
                        "required": False,
                        "default": 10,
                    },
                ],
            }
        )

    def invoke(self, request_data: Dict[str, Any], meta: Dict[str, Any] = None) -> ActionResponse:
        """Execute the data summarizer action."""
        try:
            # Get parameters
            summary_type = request_data.get("summary_type", "basic")
            columns = request_data.get("columns", [])
            group_by = request_data.get("group_by", [])
            filters = request_data.get("filters", {})
            max_rows = request_data.get("max_rows", 10)

            # Get the agent and data
            agent = self.get_agent()
            
            try:
                data = agent.get_working_data().copy()
            except ValueError as e:
                return ActionResponse(
                    message=f"No data available for summarization. {str(e)}",
                    error_info=ErrorInfo(str(e))
                )

            log.info("ml-pandas: Data summarizer called with summary_type: %s, data shape: %s", 
                    summary_type, data.shape)

            # Apply filters if provided
            if filters:
                data = self._apply_filters(data, filters)

            # Handle different summary types
            if summary_type == "count":
                # Just return the record count
                record_count = len(data)
                response_text = f"**Record Count**: {record_count:,} records"
                if filters:
                    response_text += f"\n\n**Applied Filters**: {filters}"
                
                return ActionResponse(message=response_text)

            elif summary_type == "basic":
                # Basic summary statistics
                response_text = self._generate_basic_summary(data, max_rows)
            elif summary_type == "detailed":
                # Detailed analysis
                response_text = self._generate_detailed_summary(data, columns, max_rows)
            elif summary_type == "grouped":
                # Grouped analysis
                response_text = self._generate_grouped_summary(data, group_by, columns, max_rows)
            else:
                return ActionResponse(
                    message=f"Invalid summary_type '{summary_type}'. Valid types are: basic, detailed, grouped, count",
                    error_info=ErrorInfo(f"Invalid summary_type: {summary_type}")
                )

            return ActionResponse(message=response_text)

        except Exception as e:
            log.error("Error in data summarizer action: %s", str(e))
            return ActionResponse(
                message=f"Failed to summarize data: {str(e)}",
                error_info=ErrorInfo(f"Failed to summarize data: {str(e)}")
            )

    def _apply_filters(self, data: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply filters to the data."""
        filtered_data = data.copy()
        
        for column, filter_value in filters.items():
            if column in filtered_data.columns:
                if isinstance(filter_value, dict):
                    # Handle range filters
                    if 'min' in filter_value:
                        filtered_data = filtered_data[filtered_data[column] >= filter_value['min']]
                    if 'max' in filter_value:
                        filtered_data = filtered_data[filtered_data[column] <= filter_value['max']]
                    if 'values' in filter_value:
                        filtered_data = filtered_data[filtered_data[column].isin(filter_value['values'])]
                else:
                    # Handle simple equality filter
                    filtered_data = filtered_data[filtered_data[column] == filter_value]
        
        return filtered_data

    def _generate_basic_summary(self, data: pd.DataFrame, max_rows: int) -> str:
        """Generate basic summary statistics."""
        summary_parts = []
        
        # Basic info
        summary_parts.append(f"**Data Overview**")
        summary_parts.append(f"- Total Records: {len(data):,}")
        summary_parts.append(f"- Total Columns: {len(data.columns)}")
        summary_parts.append(f"- Memory Usage: {data.memory_usage(deep=True).sum() / 1024:.1f} KB")
        
        # Column info
        summary_parts.append(f"\n**Columns**: {', '.join(data.columns.tolist())}")
        
        # Data types
        dtype_counts = data.dtypes.value_counts()
        summary_parts.append(f"\n**Data Types**:")
        for dtype, count in dtype_counts.items():
            summary_parts.append(f"- {dtype}: {count} columns")
        
        # Sample data
        if len(data) > 0:
            summary_parts.append(f"\n**Sample Data** (first {min(max_rows, len(data))} rows):")
            sample_data = data.head(max_rows)
            summary_parts.append("```")
            summary_parts.append(sample_data.to_string(index=False))
            summary_parts.append("```")
        
        return "\n".join(summary_parts)

    def _generate_detailed_summary(self, data: pd.DataFrame, columns: List[str], max_rows: int) -> str:
        """Generate detailed summary with statistics."""
        summary_parts = []
        
        # Use specified columns or all columns
        cols_to_analyze = columns if columns else data.columns.tolist()
        
        summary_parts.append(f"**Detailed Summary**")
        summary_parts.append(f"- Total Records: {len(data):,}")
        summary_parts.append(f"- Columns Analyzed: {len(cols_to_analyze)}")
        
        for col in cols_to_analyze:
            if col in data.columns:
                summary_parts.append(f"\n**{col}**:")
                
                # Basic stats
                if data[col].dtype in ['int64', 'float64']:
                    summary_parts.append(f"- Type: Numeric")
                    summary_parts.append(f"- Mean: {data[col].mean():.2f}")
                    summary_parts.append(f"- Median: {data[col].median():.2f}")
                    summary_parts.append(f"- Min: {data[col].min()}")
                    summary_parts.append(f"- Max: {data[col].max()}")
                    summary_parts.append(f"- Std Dev: {data[col].std():.2f}")
                else:
                    summary_parts.append(f"- Type: {data[col].dtype}")
                    summary_parts.append(f"- Unique Values: {data[col].nunique()}")
                    summary_parts.append(f"- Most Common: {data[col].mode().iloc[0] if not data[col].mode().empty else 'N/A'}")
                
                summary_parts.append(f"- Missing Values: {data[col].isnull().sum()}")
        
        return "\n".join(summary_parts)

    def _generate_grouped_summary(self, data: pd.DataFrame, group_by: List[str], columns: List[str], max_rows: int) -> str:
        """Generate grouped summary statistics."""
        summary_parts = []
        
        if not group_by:
            return "No group_by columns specified for grouped summary."
        
        # Validate group columns exist
        missing_groups = [col for col in group_by if col not in data.columns]
        if missing_groups:
            return f"Group columns not found: {missing_groups}"
        
        summary_parts.append(f"**Grouped Summary**")
        summary_parts.append(f"- Grouped by: {', '.join(group_by)}")
        summary_parts.append(f"- Total Groups: {data.groupby(group_by).ngroups}")
        
        # Group statistics
        grouped = data.groupby(group_by)
        
        # Count by group
        group_counts = grouped.size().reset_index(name='count')
        summary_parts.append(f"\n**Record Count by Group** (top {max_rows}):")
        summary_parts.append("```")
        summary_parts.append(group_counts.head(max_rows).to_string(index=False))
        summary_parts.append("```")
        
        # If numeric columns specified, show their stats by group
        if columns:
            numeric_cols = [col for col in columns if col in data.columns and data[col].dtype in ['int64', 'float64']]
            if numeric_cols:
                summary_parts.append(f"\n**Numeric Statistics by Group** (columns: {', '.join(numeric_cols)}):")
                for col in numeric_cols:
                    group_stats = grouped[col].agg(['mean', 'min', 'max', 'count']).reset_index()
                    summary_parts.append(f"\n**{col} Statistics**:")
                    summary_parts.append("```")
                    summary_parts.append(group_stats.head(max_rows).to_string(index=False))
                    summary_parts.append("```")
        
        return "\n".join(summary_parts)

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