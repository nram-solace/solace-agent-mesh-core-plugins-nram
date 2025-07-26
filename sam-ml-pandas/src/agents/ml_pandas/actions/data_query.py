"""Data query action for filtering and querying data using pandas."""

from typing import Dict, Any
import pandas as pd

from solace_agent_mesh.common.action import Action
from solace_agent_mesh.common.action_response import ActionResponse, ErrorInfo
from solace_ai_connector.common.log import log


class DataQueryAction(Action):
    """Action for filtering and querying data using pandas query syntax."""

    def __init__(self, **kwargs):
        """Initialize the action."""
        super().__init__(
            {
                "name": "query_data",
                "prompt_directive": (
                    "Filter and query data using pandas query syntax. "
                    "You can filter by conditions, select specific columns, and perform aggregations. "
                    "Examples: 'region == \"NY\"', 'total_sales > 1000', 'year >= 2022'"
                ),
                "params": [
                    {
                        "name": "query",
                        "desc": "Pandas query string to filter data (e.g., 'region == \"NY\" and year >= 2022')",
                        "type": "string",
                        "required": True,
                    },
                    {
                        "name": "columns",
                        "desc": "Comma-separated list of columns to include in results (empty for all columns)",
                        "type": "string",
                        "required": False,
                    },
                    {
                        "name": "aggregation",
                        "desc": "Type of aggregation to perform on filtered data",
                        "type": "string",
                        "required": False,
                        "enum": ["none", "summary", "groupby"],
                        "default": "none",
                    },
                    {
                        "name": "group_by",
                        "desc": "Column to group by when aggregation is 'groupby'",
                        "type": "string",
                        "required": False,
                    },
                    {
                        "name": "sort_by",
                        "desc": "Column to sort results by",
                        "type": "string",
                        "required": False,
                    },
                    {
                        "name": "sort_ascending",
                        "desc": "Whether to sort in ascending order",
                        "type": "boolean",
                        "required": False,
                        "default": True,
                    },
                    {
                        "name": "limit",
                        "desc": "Maximum number of rows to return",
                        "type": "integer",
                        "required": False,
                        "default": 100,
                    },
                ],
            }
        )

    def invoke(self, request_data: Dict[str, Any], meta: Dict[str, Any] = None) -> ActionResponse:
        """Execute the data query action."""
        try:
            # Get parameters
            query = request_data.get("query", "")
            columns_str = request_data.get("columns", "")
            aggregation = request_data.get("aggregation", "none")
            group_by = request_data.get("group_by", "")
            sort_by = request_data.get("sort_by", "")
            sort_ascending = request_data.get("sort_ascending", True)
            limit_raw = request_data.get("limit", 100)

            # Convert limit to integer
            try:
                limit = int(limit_raw)
                if limit < 1 or limit > 10000:
                    return ActionResponse(
                        message=f"limit must be between 1 and 10,000, got: {limit_raw}",
                        error_info=ErrorInfo(f"limit must be between 1 and 10,000, got: {limit_raw}")
                    )
            except (ValueError, TypeError):
                return ActionResponse(
                    message=f"limit must be a valid integer between 1 and 10,000, got: {limit_raw}",
                    error_info=ErrorInfo(f"limit must be a valid integer between 1 and 10,000, got: {limit_raw}")
                )

            # Validate query
            if not query or not query.strip():
                return ActionResponse(
                    message="query parameter is required",
                    error_info=ErrorInfo("query parameter is required")
                )

            # Validate aggregation
            valid_aggregations = ["none", "summary", "groupby"]
            if aggregation not in valid_aggregations:
                return ActionResponse(
                    message=f"Invalid aggregation '{aggregation}'. Valid types are: {', '.join(valid_aggregations)}",
                    error_info=ErrorInfo(f"Invalid aggregation '{aggregation}'. Valid types are: {', '.join(valid_aggregations)}")
                )

            # Validate group_by when aggregation is groupby
            if aggregation == "groupby" and not group_by:
                return ActionResponse(
                    message="group_by parameter is required when aggregation is 'groupby'",
                    error_info=ErrorInfo("group_by parameter is required when aggregation is 'groupby'")
                )

            # Get the agent and its data
            agent = self.get_agent()
            data = agent.get_working_data()

            # Parse columns
            columns = [col.strip() for col in columns_str.split(",") if col.strip()] if columns_str else None

            # Validate columns if specified
            if columns:
                missing_cols = [col for col in columns if col not in data.columns]
                if missing_cols:
                    return ActionResponse(
                        message=f"Columns not found in dataset: {missing_cols}",
                        error_info=ErrorInfo(f"Columns not found in dataset: {missing_cols}")
                    )

            # Apply query filter
            try:
                filtered_data = data.query(query)
            except Exception as e:
                return ActionResponse(
                    message=f"Invalid query syntax: {str(e)}",
                    error_info=ErrorInfo(f"Invalid query syntax: {str(e)}")
                )

            # Select columns if specified
            if columns:
                filtered_data = filtered_data[columns]

            # Apply aggregation
            if aggregation == "summary":
                # Generate summary statistics
                if filtered_data.empty:
                    result_data = {"message": "No data matches the query"}
                else:
                    result_data = {
                        "count": len(filtered_data),
                        "summary_stats": filtered_data.describe().to_dict(),
                        "sample_data": filtered_data.head(5).to_dict('records')
                    }
            elif aggregation == "groupby":
                # Validate group_by column exists
                if group_by not in filtered_data.columns:
                    return ActionResponse(
                        message=f"Group by column '{group_by}' not found in filtered data",
                        error_info=ErrorInfo(f"Group by column '{group_by}' not found in filtered data")
                    )
                
                # Perform groupby aggregation
                grouped = filtered_data.groupby(group_by)
                result_data = {
                    "group_by": group_by,
                    "group_count": len(grouped),
                    "group_summary": grouped.size().to_dict(),
                    "group_stats": grouped.describe().to_dict()
                }
            else:  # aggregation == "none"
                # Sort if specified
                if sort_by:
                    if sort_by not in filtered_data.columns:
                        return ActionResponse(
                            message=f"Sort column '{sort_by}' not found in filtered data",
                            error_info=ErrorInfo(f"Sort column '{sort_by}' not found in filtered data")
                        )
                    filtered_data = filtered_data.sort_values(sort_by, ascending=sort_ascending)

                # Apply limit
                if len(filtered_data) > limit:
                    filtered_data = filtered_data.head(limit)

                result_data = {
                    "count": len(filtered_data),
                    "total_matched": len(data.query(query)),  # Count before limit
                    "data": filtered_data.to_dict('records')
                }

            # Prepare result
            result = {
                "query": query,
                "original_shape": data.shape,
                "aggregation": aggregation,
                "result": result_data
            }

            if columns:
                result["selected_columns"] = columns
            if sort_by:
                result["sort_by"] = sort_by
                result["sort_ascending"] = sort_ascending
            if group_by:
                result["group_by"] = group_by

            # Clean result for JSON serialization
            clean_result = agent.get_data_service().clean_data_for_json(result)

            # Save results to file
            filename = f"data_query_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
            saved_path = agent.get_data_service().save_results(clean_result, filename)

            response_text = self._format_response(clean_result, saved_path)

            return ActionResponse(
                message=response_text
            )

        except Exception as e:
            log.error("Error in data query action: %s", str(e))
            return ActionResponse(
                message=f"Failed to query data: {str(e)}",
                error_info=ErrorInfo(f"Failed to query data: {str(e)}")
            )

    def _format_response(self, result: Dict[str, Any], saved_path: str) -> str:
        """Format the response text."""
        response_lines = [
            f"# Data Query Results",
            f"**Query:** `{result['query']}`",
            f"**Original Data:** {result['original_shape'][0]:,} rows × {result['original_shape'][1]} columns",
            f"**Aggregation:** {result['aggregation'].title()}",
            ""
        ]

        if "selected_columns" in result:
            response_lines.append(f"**Selected Columns:** {', '.join(result['selected_columns'])}")

        if "sort_by" in result:
            response_lines.append(f"**Sorted By:** {result['sort_by']} ({'ascending' if result['sort_ascending'] else 'descending'})")

        if "group_by" in result:
            response_lines.append(f"**Grouped By:** {result['group_by']}")

        response_lines.append("")

        # Show results based on aggregation type
        result_data = result["result"]

        if result["aggregation"] == "summary":
            if "message" in result_data:
                response_lines.append(f"**Result:** {result_data['message']}")
            else:
                response_lines.extend([
                    f"**Matched Records:** {result_data['count']:,}",
                    "",
                    "### Summary Statistics:",
                    ""
                ])
                
                # Show summary stats
                if "summary_stats" in result_data:
                    for col, stats in result_data["summary_stats"].items():
                        response_lines.append(f"**{col}:**")
                        for stat, value in stats.items():
                            if isinstance(value, (int, float)):
                                response_lines.append(f"  - {stat}: {value:,.2f}")
                            else:
                                response_lines.append(f"  - {stat}: {value}")
                        response_lines.append("")

        elif result["aggregation"] == "groupby":
            response_lines.extend([
                f"**Groups:** {result_data['group_count']}",
                "",
                "### Group Summary:",
                ""
            ])
            
            # Show group counts
            for group, count in result_data["group_summary"].items():
                response_lines.append(f"- **{group}:** {count:,} records")

        else:  # aggregation == "none"
            response_lines.extend([
                f"**Matched Records:** {result_data['total_matched']:,}",
                f"**Returned Records:** {result_data['count']:,}",
                ""
            ])
            
            if result_data['count'] > 0:
                response_lines.extend([
                    "### Sample Data:",
                    ""
                ])
                
                # Show sample data
                for i, row in enumerate(result_data["data"][:5], 1):
                    response_lines.append(f"**Row {i}:**")
                    for key, value in row.items():
                        response_lines.append(f"  - {key}: {value}")
                    response_lines.append("")
                
                if len(result_data["data"]) > 5:
                    response_lines.append(f"... and {len(result_data['data']) - 5} more rows")

        response_lines.extend([
            "---",
            f"**Results saved to:** `{saved_path}`",
            "",
            "💡 **Query Examples:**",
            "- `region == \"NY\"` - Filter by region",
            "- `total_sales > 1000 and year >= 2022` - Complex conditions",
            "- `region in [\"NY\", \"CA\"]` - Multiple values"
        ])

        return "\n".join(response_lines) 