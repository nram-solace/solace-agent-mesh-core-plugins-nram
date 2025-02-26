"""Action for executing search queries based on natural language prompts."""

from typing import Dict, Any, List
import yaml
import json
import random
import io
import re
import csv
import datetime
import dateutil.parser
from collections.abc import Mapping

from solace_agent_mesh.common.action import Action
from solace_agent_mesh.services.file_service import FileService
from solace_agent_mesh.common.action_response import ActionResponse, ErrorInfo, InlineFile


class SearchQuery(Action):
    """Action for executing search queries based on natural language prompts."""

    def __init__(self, **kwargs):
        """Initialize the action."""
        super().__init__(
            {
                "name": "search_query",
                "prompt_directive": (
                    "Execute a search query on the SQL database. "
                    "Converts natural language to SQL and returns results. NOTE that there is no history stored for previous queries, so it is essential to provide all required context in the query."
                ),
                "params": [
                    {
                        "name": "query",
                        "desc": "Natural language description of the search query, including any data required for context. Note that amfs links with resolve=true may be embedded in this parameter.",
                        "type": "string",
                        "required": True,
                    },
                    {
                        "name": "response_format",
                        "desc": "Format of the response (yaml, markdown, json or csv)",
                        "type": "string",
                        "required": False,
                        "default": "yaml",
                    },
                    {
                        "name": "inline_result",
                        "desc": "Whether to return the result as an inline file (True) or a regular file (False)",
                        "type": "boolean",
                        "required": False,
                        "default": True,
                    }
                ],
                "required_scopes": ["<agent_name>:search_query:execute"],
            },
            **kwargs
        )

    def invoke(self, params: Dict[str, Any], meta: Dict[str, Any] = None) -> ActionResponse:
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

            response_format = params.get("response_format", "yaml").lower()
            if response_format not in ["yaml", "markdown", "json", "csv"]:
                raise ValueError("Invalid response format. Choose 'yaml', 'markdown', 'json', or 'csv'")

            # Get the SQL query from the natural language query
            sql_query = self._generate_sql_query(query)
            
            # Execute the query
            db_handler = self.get_agent().get_db_handler()
            results = db_handler.execute_query(sql_query)

            return self._create_response(
                results=results,
                response_format=response_format,
                inline_result=params.get("inline_result", True),
                meta=meta,
                use_file=False,
                query={"query": query},
            )

        except Exception as e:
            return ActionResponse(
                message=f"Error executing search query: {str(e)}",
                error_info=ErrorInfo(str(e))
            )

    def _generate_sql_query(self, natural_language_query: str) -> str:
        """Generate SQL query from natural language prompt.
        
        Args:
            natural_language_query: Natural language description of the query
            
        Returns:
            Generated SQL query string
            
        Raises:
            ValueError: If query generation fails
        """
        agent = self.get_agent()
        db_schema = agent.detailed_schema
        data_description = agent.data_description
        db_type = agent.db_type
        db_schema_yaml = yaml.dump(db_schema)

        system_prompt = f"""
You are an SQL expert and will convert the provided natural language query to a SQL query for {db_type}. 
Requests should have a clear context to identify the person or entity or use the word "all" to avoid ambiguity.
It is acceptable to raise an error if the context is missing or ambiguous.

The database schema is as follows:
<db_schema_yaml>
{db_schema_yaml}
</db_schema_yaml>

Additional information about the data:
<data_description>
{data_description}
</data_description>

Respond with the {db_type} query in the following format:

<query_purpose>
...Purpose of the query...
</query_purpose>
<sql_query>
...SQL query...
</sql_query>

Or if the request is invalid, respond with an error message:

<error>
...Error message...
</error>

Ensure that the SQL query is compatible with {db_type}.
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": natural_language_query},
        ]

        try:
            response = agent.do_llm_service_request(messages=messages)
            content = response.get("content", "").strip()

            errors = self._get_all_tags(content, "error")
            if errors:
                raise ValueError(errors[0])

            sql_queries = self._get_all_tags(content, "sql_query")
            if not sql_queries:
                raise ValueError("Failed to generate SQL query")

            return sql_queries[0]

        except Exception as e:
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


    def _create_response(
        self,
        results: List[Dict[str, Any]],
        response_format: str,
        inline_result: bool,
        meta: Dict[str, Any],
        use_file: bool,
        query: Dict[str, Any],
    ) -> ActionResponse:
        """Create a response with the query results as a file or inline file."""
        file_service = FileService()
        session_id = meta.get("session_id")
        updated_results = self._stringify_non_standard_objects(results)

        if response_format == "yaml":
            content = yaml.dump(updated_results)
            file_extension = "yaml"
        elif response_format == "markdown":
            content = self._format_markdown_table(updated_results)
            file_extension = "md"
        elif response_format == "json":
            content = json.dumps(updated_results, indent=2, default=str)
            file_extension = "json"
        else:  # CSV
            content = self._format_csv(updated_results)
            file_extension = "csv"

        file_name = f"query_results_{random.randint(100000, 999999)}.{file_extension}"

        if inline_result and not use_file:
            inline_file = InlineFile(content, file_name)
            return ActionResponse(
                message=f"Query results are available in the attached inline {response_format.upper()} file.",
                inline_files=[inline_file],
            )
        else:
            data_source = f"SQL Agent - Search Query Action - Query: {json.dumps(query)}"
            file_meta = file_service.upload_from_buffer(
                content.encode(), file_name, session_id, data_source=data_source
            )
            return ActionResponse(
                message=f"Query results are available in the attached {response_format.upper()} file.",
                files=[file_meta],
            )

    def _format_markdown_table(self, results: List[Dict[str, Any]]) -> str:
        """Format results as a Markdown table."""
        if not results:
            return "No results found."

        # Get all unique keys from all documents
        headers = set()
        for result in results:
            headers.update(result.keys())
        headers = sorted(list(headers))

        markdown = "| " + " | ".join(headers) + " |\n"
        markdown += "| " + " | ".join(["---" for _ in headers]) + " |\n"

        for row in results:
            markdown += "| " + " | ".join(str(row.get(header, "")) for header in headers) + " |\n"

        return markdown

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