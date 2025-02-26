"""Async server thread handler for MCP communication."""

import asyncio
import queue
from dataclasses import dataclass
from threading import Thread, Event
from typing import Any, Dict, Optional

from mcp import StdioServerParameters, ClientSession
from mcp.client.stdio import stdio_client
from solace_ai_connector.common.log import log


@dataclass
class ServerRequest:
    """Request message for server operations."""
    operation: str  # 'tool', 'resource', or 'prompt'
    name: str
    params: Dict[str, Any]
    response_queue: queue.Queue


class AsyncServerThread:
    """Handles async MCP server communication in a dedicated thread."""

    def __init__(self, server_params: StdioServerParameters, received_request_callback=None):
        """Initialize the async server thread.

        Args:
            server_params: Parameters for initializing the stdio server.
        """
        self.server_params = server_params
        self.request_queue: queue.Queue[ServerRequest] = queue.Queue()
        self.running = False
        self.thread: Optional[Thread] = None
        self.client_session: Optional[ClientSession] = None
        self.received_request_callback = received_request_callback
        
        # Store initialization results
        self.tools: list = []
        self.resources: list = []
        self.prompts: list = []
        self.initialized = Event()

    def start(self):
        """Start the async server thread."""
        if self.thread is not None:
            return

        self.running = True
        self.thread = Thread(target=self._run_server, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the async server thread."""
        self.running = False
        if self.thread is not None:
            self.request_queue.put(None)  # Sentinel to stop the thread
            self.thread.join()
            self.thread = None

    async def _initialize_capabilities(self):
        """Initialize server capabilities by fetching tools, resources and prompts."""
        try:
            # Get tools
            try:
                tool_result = await self.client_session.list_tools()
                self.tools = tool_result.tools
            except Exception as e:
                log.info("Server does not support tools: %s", str(e))
                self.tools = []

            # Get resources
            try:
                resources_result = await self.client_session.list_resources()
                self.resources = resources_result.resources
            except Exception as e:
                log.info("Server does not support resources: %s", str(e))
                self.resources = []

            # Get prompts
            try:
                prompts_result = await self.client_session.list_prompts()
                self.prompts = prompts_result.prompts
            except Exception as e:
                log.info("Server does not support prompts: %s", str(e))
                self.prompts = []

        finally:
            # Signal initialization is complete
            self.initialized.set()

    def _run_server(self):
        """Main server thread function."""
        async def server_main():
            try:
                # Initialize server connection using context managers
                async with stdio_client(self.server_params) as (read, write):
                    async with ClientSession(read, write) as self.client_session:
                        await self.client_session.initialize()
                        if self.received_request_callback:
                            # pylint: disable=protected-access
                            self.client_session._received_request = self.received_request_callback
                        
                        # Initialize capabilities
                        await self._initialize_capabilities()

                        # Process requests until stopped
                        while self.running:
                            try:
                                # Non-blocking check for requests
                                request = self.request_queue.get_nowait()
                                if request is None:  # Sentinel value
                                    break

                                try:
                                    # Execute requested operation
                                    if request.operation == 'tool':
                                        result = await self.client_session.call_tool(
                                            request.name, request.params)
                                    elif request.operation == 'resource':
                                        result = await self.client_session.get_resource(request.name)
                                    elif request.operation == 'prompt':
                                        result = await self.client_session.get_prompt(
                                            request.name, request.params)
                                    else:
                                        raise ValueError(
                                            f"Unknown operation: {request.operation}")

                                    request.response_queue.put(('success', result))
                                except Exception as e:
                                    request.response_queue.put(('error', str(e)))

                            except queue.Empty:
                                await asyncio.sleep(0.1)  # Prevent busy-waiting

            except Exception as e:
                log.error("Server thread error: %s", str(e))
                raise

        # Run the async code in this thread
        asyncio.run(server_main())

    def execute(self, operation: str, name: str,
               params: Dict[str, Any]) -> Any:
        """Execute a server operation.

        Args:
            operation: Type of operation ('tool', 'resource', or 'prompt')
            name: Name of the operation to execute
            params: Parameters for the operation

        Returns:
            The operation result

        Raises:
            RuntimeError: If the server thread is not running or operation fails
        """
        if not self.running or self.thread is None:
            raise RuntimeError("Server thread is not running")

        response_queue: queue.Queue = queue.Queue()
        request = ServerRequest(
            operation=operation,
            name=name,
            params=params,
            response_queue=response_queue
        )

        self.request_queue.put(request)
        status, result = response_queue.get()

        if status == 'error':
            raise RuntimeError(f"Server operation failed: {result}")
        return result
