"""
OneDrive Data Source implementation for the scanner component.

This module provides OneDrive integration with Microsoft Graph API authentication,
batch scanning, real-time monitoring, and file processing capabilities.
"""

import os
import tempfile
import threading
import time
from typing import Dict, List, Any, Optional
from solace_ai_connector.common.log import log as logger

from ..cloud_storage import CloudStorageDataSource

# Try to import OneDrive/Microsoft Graph dependencies
try:
    import msal
    import requests

    ONEDRIVE_AVAILABLE = True
except ImportError:
    ONEDRIVE_AVAILABLE = False
    logger.warning(
        "OneDrive dependencies not available. Install with: pip install msal requests"
    )


class OneDriveDataSource(CloudStorageDataSource):
    """
    OneDrive implementation of CloudStorageDataSource.

    Provides Microsoft Graph API authentication, batch scanning, real-time monitoring,
    and file processing for OneDrive documents (personal and business).
    """

    # Microsoft Graph API scopes
    SCOPES = [
        "https://graph.microsoft.com/Files.Read.All",
        "https://graph.microsoft.com/Sites.Read.All",
    ]

    # Microsoft Graph API endpoints
    GRAPH_API_ENDPOINT = "https://graph.microsoft.com/v1.0"

    # Office format mappings for download
    OFFICE_FORMATS = {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
        "application/vnd.ms-excel": ".xls",
        "application/msword": ".doc",
        "application/vnd.ms-powerpoint": ".ppt",
    }

    def __init__(self, config: Dict, ingested_documents: List[str], pipeline):
        """
        Initialize the OneDriveDataSource.

        Args:
            config: Configuration dictionary for OneDrive.
            ingested_documents: List of already ingested documents.
            pipeline: The processing pipeline instance.
        """
        if not ONEDRIVE_AVAILABLE:
            raise ImportError(
                "OneDrive dependencies not available. "
                "Install with: pip install msal requests"
            )

        super().__init__(config, ingested_documents, pipeline)
        self.provider_name = "onedrive"

        # OneDrive specific configuration
        self.app = None
        self.access_token = None
        self.client_id = ""
        self.client_secret = ""
        self.tenant_id = ""
        self.authority = ""
        self.account_type = "personal"  # personal or business

        # Initialize the service
        self.process_config(config)

    def process_config(self, source: Dict = {}) -> None:
        """
        Process OneDrive specific configuration.

        Args:
            source: Configuration dictionary containing OneDrive settings.
        """
        # Get authentication configuration
        self.client_id = source.get("client_id", "")
        self.client_secret = source.get("client_secret", "")
        self.tenant_id = source.get("tenant_id", "")
        self.account_type = source.get("account_type", "personal")

        if not self.client_id:
            raise ValueError("OneDrive client_id is required")

        # Set authority based on account type
        if self.account_type == "business" and self.tenant_id:
            self.authority = f"https://login.microsoftonline.com/{self.tenant_id}"
        else:
            self.authority = "https://login.microsoftonline.com/common"

        # Get folder configurations
        self.folders = source.get("folders", [])
        if not self.folders:
            # Default to root folder
            self.folders = [{"path": "/", "name": "Root", "recursive": True}]

        # Get real-time monitoring configuration
        real_time_config = source.get("real_time", {})
        self.real_time_enabled = real_time_config.get("enabled", False)
        self.webhook_url = real_time_config.get("webhook_url")
        self.polling_interval = real_time_config.get("polling_interval", 300)

        logger.info(
            f"OneDrive configuration processed: {len(self.folders)} folders, "
            f"account_type: {self.account_type}, real-time: {self.real_time_enabled}"
        )

    def _authenticate(self) -> bool:
        """
        Authenticate with Microsoft Graph API using MSAL.

        Returns:
            True if authentication successful, False otherwise.
        """
        try:
            # Create MSAL application
            if self.client_secret:
                # Confidential client (business apps)
                self.app = msal.ConfidentialClientApplication(
                    client_id=self.client_id,
                    client_credential=self.client_secret,
                    authority=self.authority,
                )
            else:
                # Public client (personal apps)
                self.app = msal.PublicClientApplication(
                    client_id=self.client_id, authority=self.authority
                )

            # Try to get token from cache
            accounts = self.app.get_accounts()
            if accounts:
                # Try silent authentication first
                result = self.app.acquire_token_silent(self.SCOPES, account=accounts[0])
                if result and "access_token" in result:
                    self.access_token = result["access_token"]
                    logger.info("OneDrive authentication successful (cached token)")
                    return True

            # Interactive authentication
            if self.client_secret:
                # Client credentials flow for business apps
                result = self.app.acquire_token_for_client(scopes=self.SCOPES)
            else:
                # Interactive flow for personal apps
                result = self.app.acquire_token_interactive(scopes=self.SCOPES)

            if result and "access_token" in result:
                self.access_token = result["access_token"]
                logger.info("OneDrive authentication successful")
                return True
            else:
                error = result.get("error_description", "Unknown error")
                logger.error(f"OneDrive authentication failed: {error}")
                return False

        except Exception as e:
            logger.error(f"OneDrive authentication failed: {str(e)}")
            return False

    def _make_graph_request(
        self, endpoint: str, method: str = "GET", params: Dict = None
    ) -> Dict:
        """
        Make a request to Microsoft Graph API.

        Args:
            endpoint: The API endpoint (relative to graph API base).
            method: HTTP method (GET, POST, etc.).
            params: Query parameters.

        Returns:
            Response data as dictionary.
        """
        if not self.access_token:
            logger.error("No access token available for Graph API request")
            return {}

        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

        url = f"{self.GRAPH_API_ENDPOINT}/{endpoint.lstrip('/')}"

        try:
            response = requests.request(method, url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Graph API request failed: {str(e)}")
            return {}

    def _list_files(
        self, folder_path: str = None, recursive: bool = True
    ) -> List[Dict[str, Any]]:
        """
        List files in OneDrive.

        Args:
            folder_path: The path of the folder to list files from.
            recursive: Whether to list files recursively.

        Returns:
            A list of file metadata dictionaries.
        """
        if not self.access_token:
            logger.error("OneDrive not authenticated")
            return []

        files = []

        try:
            # Build endpoint for folder
            if folder_path and folder_path != "/":
                # Encode folder path for URL
                encoded_path = folder_path.replace(" ", "%20")
                endpoint = f"me/drive/root:{encoded_path}:/children"
            else:
                endpoint = "me/drive/root/children"

            # Get files with pagination
            while endpoint:
                response = self._make_graph_request(endpoint)
                items = response.get("value", [])

                for item in items:
                    if item.get("folder"):
                        # Handle folders recursively
                        if recursive:
                            folder_files = self._list_files(
                                item.get("parentReference", {}).get("path", "")
                                + "/"
                                + item["name"],
                                recursive,
                            )
                            files.extend(folder_files)
                    else:
                        # Add file metadata
                        files.append(
                            {
                                "id": item["id"],
                                "name": item["name"],
                                "mime_type": item.get("file", {}).get(
                                    "mimeType", "application/octet-stream"
                                ),
                                "size": item.get("size", 0),
                                "modified_time": item.get("lastModifiedDateTime"),
                                "download_url": item.get(
                                    "@microsoft.graph.downloadUrl"
                                ),
                                "web_url": item.get("webUrl"),
                                "path": item.get("parentReference", {}).get("path", "")
                                + "/"
                                + item["name"],
                            }
                        )

                # Handle pagination
                endpoint = response.get("@odata.nextLink")
                if endpoint:
                    # Extract relative endpoint from full URL
                    endpoint = endpoint.replace(self.GRAPH_API_ENDPOINT + "/", "")

            logger.info(
                f"Listed {len(files)} files from OneDrive folder {folder_path or 'root'}"
            )
            return files

        except Exception as e:
            logger.error(f"Error listing OneDrive files: {str(e)}")
            return []

    def _download_file(self, file_id: str, file_name: str) -> str:
        """
        Download a file from OneDrive to a temporary location.

        Args:
            file_id: The ID of the file to download.
            file_name: The name of the file.

        Returns:
            The path to the downloaded temporary file.
        """
        if not self.access_token:
            logger.error("OneDrive not authenticated")
            return ""

        try:
            # Get download URL
            endpoint = f"me/drive/items/{file_id}"
            response = self._make_graph_request(endpoint)
            download_url = response.get("@microsoft.graph.downloadUrl")

            if not download_url:
                logger.error(f"No download URL available for OneDrive file {file_name}")
                return ""

            # Download file content
            file_response = requests.get(download_url)
            file_response.raise_for_status()

            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, suffix=f"_{file_name}", dir=self.temp_dir
            )

            # Write content to temporary file
            temp_file.write(file_response.content)
            temp_file.close()

            logger.info(f"Downloaded OneDrive file to: {temp_file.name}")
            return temp_file.name

        except Exception as e:
            logger.error(f"Error downloading OneDrive file {file_name}: {str(e)}")
            return ""

    def _setup_real_time_monitoring(self) -> None:
        """
        Set up real-time monitoring for OneDrive.

        This implementation uses polling as a fallback. In production,
        you would set up Microsoft Graph webhooks (subscriptions).
        """
        if self.webhook_url:
            logger.info(
                "OneDrive webhook monitoring not yet implemented, using polling"
            )

        # Start polling thread
        self._start_polling()

    def batch_scan(self) -> None:
        """
        Perform batch scanning of all files in configured OneDrive folders.
        """
        logger.info("Starting OneDrive batch scan")

        if not self._authenticate():
            logger.error("Failed to authenticate with OneDrive")
            return

        for folder_config in self.folders:
            folder_path = folder_config.get("path", "/")
            folder_name = folder_config.get("name", "Unknown")
            recursive = folder_config.get("recursive", True)

            logger.info(
                f"Scanning OneDrive folder: {folder_name} (path: {folder_path})"
            )

            try:
                files = self._list_files(folder_path, recursive)
                for file_info in files:
                    self._process_cloud_file(file_info)
            except Exception as e:
                logger.error(f"Error scanning OneDrive folder {folder_name}: {str(e)}")

        logger.info("OneDrive batch scan completed")
