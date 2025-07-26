"""Action for executing search queries based on natural language prompts."""

from typing import Dict, Any, List, Tuple
import yaml
import json
import random
import io
import re
import csv
import datetime
import dateutil.parser
from collections.abc import Mapping
import logging

from solace_agent_mesh.common.action import Action
from solace_agent_mesh.services.file_service import FileService
from solace_agent_mesh.common.action_response import (
    ActionResponse,
    ErrorInfo,
    InlineFile,
)

MAX_TOTAL_INLINE_FILE_SIZE = 100000  # 100KB

log = logging.getLogger(__name__)


class SearchQuery(Action):
    """Action for executing search queries based on natural language prompts."""

    def __init__(self, **kwargs):
        """Initialize the action."""
        super().__init__(
            {
                "name": "search_query",
                "prompt_directive": (
                    "Execute one or more search queries on the SQL database. "
                    "Converts natural language to SQL and returns results. "
                    "You can include multiple related questions in a single query for more efficient processing. "
                    "Each query will be returned as a separate file. "
                    "NOTE that there is no history stored for previous queries, so it is essential to provide all required context in the query."
                ),
                "params": [
                    {
                        "name": "query",
                        "desc": "Natural language description of the search query or queries, including any data required for context. Multiple related questions can be included for more efficient processing. Note that amfs links with resolve=true may be embedded in this parameter.",
                        "type": "string",
                        "required": True,
                    },
                    {
                        "name": "response_format",
                        "desc": "Format of the response (yaml, json or csv)",
                        "type": "string",
                        "required": False,
                        "default": "yaml",
                    },
                    {
                        "name": "inline_result",
                        "desc": "Whether to return the result as an inline file (True) or a regular file (False)",
                        "type": "boolean",
                        "required": False,
                        "default": False,
                    },
                ],
                "required_scopes": ["<agent_name>:search_query:execute"],
            },
            **kwargs,
        )

    def invoke(
        self, params: Dict[str, Any], meta: Dict[str, Any] = None
    ) -> ActionResponse:
        """Execute the search query based on the natural language prompt.

        Args:
            params: Action parameters including the natural language query and response format
            meta: Optional metadata

        Returns:
            ActionResponse containing the query results
        """
        try:
            query = params.get("query")
            if not query:
                raise ValueError("Natural language query is required")

            log.info("sql-db: Received natural language query: %s", query)

            response_format = params.get("response_format", "yaml").lower()
            if response_format not in ["yaml", "json", "csv"]:
                raise ValueError("Invalid response format. Choose 'yaml', 'json', or 'csv'")

            log.info("sql-db: Response format requested: %s", response_format)

            # Get the SQL queries from the natural language query
            sql_queries = self._generate_sql_queries(query)
            log.info("sql-db: Generated %d SQL queries from natural language query", len(sql_queries))

            # Execute each query and collect results
            db_handler = self.get_agent().get_db_handler()
            query_results = []
            failed_queries = []

            for i, (purpose, sql_query) in enumerate(sql_queries):
                log.info("sql-db: Executing query %d/%d - Purpose: %s", i+1, len(sql_queries), purpose)
                log.info("sql-db: SQL Query %d: %s", i+1, sql_query)
                
                try:
                    results = db_handler.execute_query(sql_query)
                    log.info("sql-db: Query %d successful - Returned %d records", i+1, len(results))
                    query_results.append((purpose, sql_query, results))
                except Exception as e:
                    log.error("sql-db: Query %d failed - Purpose: %s, Error: %s", i+1, purpose, str(e))
                    failed_queries.append((purpose, sql_query, str(e)))

            log.info("sql-db: Query execution summary - Successful: %d, Failed: %d", 
                    len(query_results), len(failed_queries))

            inline_result = params.get("inline_result", True)
            if isinstance(inline_result, str):
                inline_result = inline_result.lower() == "true"

            return_json_data = params.get("return_json_data", False)
            if isinstance(return_json_data, str):
                return_json_data = return_json_data.lower() == "true"

            # Create response with files for each successful query
            return self._create_multi_query_response(
                query_results=query_results,
                failed_queries=failed_queries,
                response_format=response_format,
                inline_result=inline_result,
                return_json_data=return_json_data,
                meta=meta,
                query={"query": query},
            )

        except Exception as e:
            log.error("sql-db: Error executing search query: %s", str(e))
            return ActionResponse(
                message=f"Error executing search query: {str(e)}",
                error_info=ErrorInfo(str(e)),
            )

    def _generate_sql_queries(
        self, natural_language_query: str
    ) -> List[Tuple[str, str]]:
        """Generate SQL queries from natural language prompt.

        Args:
            natural_language_query: Natural language description of the query

        Returns:
            List of tuples containing (query_purpose, sql_query)

        Raises:
            ValueError: If query generation fails
        """
        agent = self.get_agent()
        db_schema = agent.detailed_schema
        data_description = agent.data_description
        db_type = agent.db_type
        db_schema_yaml = yaml.dump(db_schema)
        current_timestamp = datetime.datetime.now().isoformat()

        log.info("sql-db: Generating SQL queries for database type: %s", db_type)
        log.debug("sql-db: Database schema has %d tables", len(db_schema))

        # Build the prompt for the LLM
        prompt = f"""You are a SQL expert. Generate SQL queries based on the natural language request.

Database Information:
- Database Type: {db_type}
- Database Purpose: {data_description}
- Current Timestamp: {current_timestamp}

Database Schema:
{db_schema_yaml}

Natural Language Query: {natural_language_query}

Generate one or more SQL queries to answer this request. For each query, provide:
1. A clear purpose/description of what the query does
2. The actual SQL query

Format your response using these XML-like tags:
<query_purpose>Description of what this query does</query_purpose>
<sql_query>SELECT ... FROM ... WHERE ...</sql_query>

If you need multiple queries, repeat the tags for each query.

Guidelines:
- Use appropriate SQL syntax for {db_type}
- Include LIMIT clauses for large result sets
- Use proper table and column names from the schema
- Handle date/time queries appropriately
- Consider performance and efficiency
- If the query might return many rows, add a reasonable LIMIT

Response Guidelines: {agent.response_guidelines if agent.response_guidelines else "Provide clear, accurate SQL queries."}"""

        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        log.info("sql-db: Sending request to LLM for SQL generation")
        log.debug("sql-db: LLM prompt length: %d characters", len(prompt))

        try:
            response = agent.do_llm_service_request(messages=messages)
            content = response.get("content", "").strip()

            log.info("sql-db: Received LLM response for SQL generation")
            log.debug("sql-db: LLM response length: %d characters", len(content))

            errors = self._get_all_tags(content, "error")
            if errors:
                log.error("sql-db: LLM returned error: %s", errors[0])
                raise ValueError(errors[0])

            sql_queries = self._get_all_tags(content, "sql_query")
            purposes = self._get_all_tags(content, "query_purpose")

            log.info("sql-db: Extracted %d SQL queries and %d purposes from LLM response", 
                    len(sql_queries), len(purposes))

            if not sql_queries:
                log.error("sql-db: No SQL queries found in LLM response")
                raise ValueError("Failed to generate SQL query")

            # Match purposes with queries
            if len(purposes) != len(sql_queries):
                log.warning("sql-db: Purpose count (%d) doesn't match query count (%d), using generic purposes", 
                           len(purposes), len(sql_queries))
                # If counts don't match, use generic purposes
                purposes = [f"Query {i+1}" for i in range(len(sql_queries))]

            result = list(zip(purposes, sql_queries))
            log.info("sql-db: Successfully generated %d SQL queries", len(result))
            
            return result

        except Exception as e:
            log.error("sql-db: Failed to generate SQL queries: %s", str(e))
            raise ValueError(f"Failed to generate SQL query: {str(e)}")

    def _get_all_tags(self, result_text: str, tag_name: str) -> list:
        """Extract content from XML-like tags in the text.

        Args:
            result_text: Text to search for tags
            tag_name: Name of the tag to find

        Returns:
            List of strings containing the content of each matching tag
        """
        pattern = f"<{tag_name}>(.*?)</{tag_name}>"
        return re.findall(pattern, result_text, re.DOTALL)

    def _create_multi_query_response(
        self,
        query_results: List[Tuple[str, str, List[Dict[str, Any]]]],
        failed_queries: List[Tuple[str, str, str]],
        response_format: str,
        inline_result: bool,
        return_json_data: bool,
        meta: Dict[str, Any],
        query: Dict[str, Any],
    ) -> ActionResponse:
        """Create a response with multiple query results.

        Args:
            query_results: List of (purpose, sql_query, results) tuples
            failed_queries: List of (purpose, sql_query, error) tuples
            response_format: Output format (yaml, json, csv)
            inline_result: Whether to include results inline
            return_json_data: Whether to return data directly as JSON
            meta: Optional metadata
            query: Original query parameters

        Returns:
            ActionResponse with results and optional files
        """
        log.info("sql-db: Creating multi-query response with %d successful and %d failed queries", 
                len(query_results), len(failed_queries))

        # If return_json_data is requested, return data directly as JSON
        if return_json_data and query_results:
            log.info("sql-db: Returning data directly as JSON for agent consumption")
            
            # Combine all results into a single JSON structure
            combined_data = {
                "query": query.get("query", ""),
                "total_queries": len(query_results),
                "successful_queries": len(query_results),
                "failed_queries": len(failed_queries),
                "results": []
            }
            
            for i, (purpose, sql_query, results) in enumerate(query_results):
                combined_data["results"].append({
                    "query_number": i + 1,
                    "purpose": purpose,
                    "sql_query": sql_query,
                    "record_count": len(results),
                    "data": results
                })
            
            # Add failed queries if any
            if failed_queries:
                combined_data["failed_queries_details"] = []
                for i, (purpose, sql_query, error) in enumerate(failed_queries):
                    combined_data["failed_queries_details"].append({
                        "query_number": i + 1,
                        "purpose": purpose,
                        "sql_query": sql_query,
                        "error": error
                    })
            
            json_data = json.dumps(combined_data, indent=2, default=str)
            
            response_message = f"**SQL Query Results**\n\n"
            response_message += f"Query: {query.get('query', '')}\n"
            response_message += f"Successful queries: {len(query_results)}\n"
            response_message += f"Failed queries: {len(failed_queries)}\n"
            response_message += f"Total records: {sum(len(results) for _, _, results in query_results)}\n\n"
            response_message += f"**JSON Data for Agent Consumption:**\n```json\n{json_data}\n```"
            
            return ActionResponse(
                message=response_message,
                files=[{
                    "filename": f"sql_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "content": json_data,
                    "content_type": "application/json"
                }]
            )

        # Original file-based response logic
        files = []
        response_parts = []

        # Process successful queries
        for i, (purpose, sql_query, results) in enumerate(query_results):
            log.info("sql-db: Processing successful query %d/%d - Purpose: %s, Records: %d", 
                    i+1, len(query_results), purpose, len(results))
            
            if not results:
                log.warning("sql-db: Query %d returned no results", i+1)
                response_parts.append(f"**{purpose}**: No results found")
                continue

            # Create file for this query result with short random name
            import random
            import string
            
            # Generate a short random string (6 characters)
            random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
            filename = f"q{i+1}_{random_suffix}_{datetime.datetime.now().strftime('%H%M%S')}"
            
            if response_format == "csv":
                filename += ".csv"
                content = self._convert_to_csv(results)
            elif response_format == "json":
                filename += ".json"
                content = json.dumps(results, indent=2, default=str)
            else:  # yaml
                filename += ".yaml"
                content = yaml.dump(results, default_flow_style=False, allow_unicode=True)

            # Save file to disk for debugging
            import os
            output_dir = "./sql_output"
            os.makedirs(output_dir, exist_ok=True)
            file_path = os.path.join(output_dir, filename)
            
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                log.info("sql-db: Saved file to disk: %s (%d bytes)", file_path, len(content))
            except Exception as e:
                log.error("sql-db: Failed to save file to disk: %s", str(e))

            files.append({
                "filename": filename,
                "content": content,
                "content_type": "text/plain"
            })

            log.info("sql-db: Created file for query %d: %s (%d bytes)", i+1, filename, len(content))

            # Add to response message
            if inline_result:
                response_parts.append(f"**{purpose}** ({len(results)} records):\n```{response_format}\n{content}\n```")
            else:
                response_parts.append(f"**{purpose}**: {len(results)} records saved to `{file_path}`")

        # Process failed queries
        if failed_queries:
            log.warning("sql-db: Processing %d failed queries", len(failed_queries))
            response_parts.append("\n**Failed Queries:**")
            for i, (purpose, sql_query, error) in enumerate(failed_queries):
                log.error("sql-db: Failed query %d - Purpose: %s, Error: %s", i+1, purpose, error)
                response_parts.append(f"- **{purpose}**: {error}")

        # Combine all parts
        response_message = "\n\n".join(response_parts)
        
        log.info("sql-db: Multi-query response created - Total files: %d, Response length: %d characters", 
                len(files), len(response_message))

        return ActionResponse(
            message=response_message,
            files=files if files else None
        )

    def _format_csv(self, results: List[Dict[str, Any]]) -> str:
        """Format results as a CSV string."""
        if not results:
            return "No results found."

        # Get all unique keys from all documents
        headers = set()
        for result in results:
            headers.update(result.keys())
        headers = sorted(list(headers))

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)
        return output.getvalue()

    def _stringify_non_standard_objects(self, data):
        """Recursively convert non-serializable data types to strings."""
        if isinstance(data, dict) or isinstance(data, Mapping):
            return {k: self._stringify_non_standard_objects(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._stringify_non_standard_objects(item) for item in data]
        elif isinstance(data, (int, float, bool, str)):
            return data
        elif isinstance(data, (datetime.datetime, datetime.date)):
            return data.isoformat()
        else:
            return str(data)

    def _convert_iso_dates_to_datetime(self, query_json):
        """
        Converts any occurrences of {"$date": "ISODateString"} in the JSON query to
        datetime.datetime objects.

        Args:
            query_json (dict or list): The JSON query to process.

        Returns:
            dict or list: The input query with ISODate strings converted to datetime objects.
        """

        def convert(obj):
            if isinstance(obj, dict):
                # If the object is a dictionary, iterate over the key-value pairs
                for key, value in obj.items():
                    if isinstance(value, dict) and "$date" in value:
                        # Convert the ISO date string to datetime
                        obj[key] = dateutil.parser.parse(value["$date"])
                    else:
                        # Recursively process nested dictionaries
                        obj[key] = convert(value)
            elif isinstance(obj, list):
                # If the object is a list, iterate over the items
                for i in range(len(obj)):
                    obj[i] = convert(obj[i])
            return obj

        query = query_json.copy()
        return convert(query)
