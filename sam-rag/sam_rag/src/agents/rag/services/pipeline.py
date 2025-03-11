"""
RAG Pipeline that combines preprocessing, splitting, embedding, and ingestion components.

This module provides a complete pipeline for processing documents through the RAG workflow:
1. Preprocessing: Convert documents to clean text
2. Splitting: Split text into chunks
3. Embedding: Convert chunks to vector embeddings
4. Ingestion: Store embeddings in a vector database

Usage:
    python -m src.agents.rag.services.pipeline
"""

import os
import yaml
import logging
from typing import Dict, List, Any, Optional, Tuple

from src.agents.rag.services.preprocessor.document_processor import DocumentProcessor
from src.agents.rag.services.splitter.splitter_service import SplitterService
from src.agents.rag.services.embedder.embedder_service import EmbedderService
from src.agents.rag.services.ingestor.ingestor_service import IngestorService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    RAG Pipeline that combines preprocessing, splitting, embedding, and ingestion components.
    """

    def __init__(self, config_path: str = None):
        """
        Initialize the RAG pipeline.

        Args:
            config_path: Path to the configuration file. If None, uses the default path.
        """
        # Set default config path if not provided
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(
                            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                        )
                    )
                ),
                "configs",
                "agents",
                "rag.yaml",
            )

        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize components
        self.preprocessor = DocumentProcessor(self.config.get("preprocessor", {}))
        self.splitter = SplitterService(self.config.get("splitter", {}))
        self.embedder = EmbedderService(self.config.get("embedder", {}))
        self.ingestor = IngestorService(
            {
                "preprocessor": self.config.get("preprocessor", {}),
                "splitter": self.config.get("splitter", {}),
                "embedder": self.config.get("embedder", {}),
                "vector_db": self.config.get("vector_db", {}),
            }
        )

        logger.info("RAG Pipeline initialized")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from a YAML file.

        Args:
            config_path: Path to the configuration file.

        Returns:
            A dictionary containing the configuration.
        """
        try:
            with open(config_path, "r") as file:
                yaml_config = yaml.safe_load(file)

            # Extract the component configurations
            for flow in yaml_config.get("flows", []):
                if flow.get("name") == "rag_action_request_processor":
                    for component in flow.get("components", []):
                        if (
                            component.get("component_name")
                            == "action_request_processor"
                        ):
                            return component.get("component_config", {})

            logger.warning("Could not find RAG configuration in the YAML file")
            return {}
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {str(e)}")
            return {}

    def process_documents(
        self, documents: List[str], document_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Process documents through the RAG pipeline.

        Args:
            documents: List of document file paths or content strings.
            document_types: Optional list of document types (e.g., "pdf", "text", "html").
                If not provided, types will be inferred from file extensions.

        Returns:
            A dictionary containing the processing results.
        """
        logger.info(f"Processing {len(documents)} documents through the RAG pipeline")

        # Step 1: Preprocess documents
        preprocessed_docs = []
        preprocessed_metadata = []

        for i, doc in enumerate(documents):
            try:
                # Check if the document is a file path or content string
                if os.path.exists(doc):
                    # Process as a file
                    text = self.preprocessor.process_document(doc)
                    doc_type = (
                        document_types[i]
                        if document_types and i < len(document_types)
                        else self._get_file_type(doc)
                    )
                    source = doc
                else:
                    # Process as a content string
                    text = self.preprocessor.clean_text(doc)
                    doc_type = (
                        document_types[i]
                        if document_types and i < len(document_types)
                        else "text"
                    )
                    source = f"document_{i}"

                if text:
                    preprocessed_docs.append(text)
                    preprocessed_metadata.append(
                        {"source": source, "document_type": doc_type, "index": i}
                    )
                    logger.info(f"Successfully preprocessed document {i} ({doc_type})")
                else:
                    logger.warning(f"Failed to preprocess document {i}")
            except Exception as e:
                logger.error(f"Error preprocessing document {i}: {str(e)}")

        if not preprocessed_docs:
            logger.warning("No documents were successfully preprocessed")
            return {
                "success": False,
                "message": "No documents were successfully preprocessed",
                "document_ids": [],
            }

        # Step 2: Split documents into chunks
        chunks = []
        chunks_metadata = []

        for i, (doc, meta) in enumerate(zip(preprocessed_docs, preprocessed_metadata)):
            try:
                # Get the document type
                doc_type = meta.get("document_type", "text")

                # Split the document
                doc_chunks = self.splitter.split_text(doc, doc_type)

                # Add chunks and metadata
                chunks.extend(doc_chunks)
                chunks_metadata.extend([meta.copy() for _ in range(len(doc_chunks))])

                logger.info(f"Split document {i} into {len(doc_chunks)} chunks")
            except Exception as e:
                logger.error(f"Error splitting document {i}: {str(e)}")

        if not chunks:
            logger.warning("No chunks were created from the documents")
            return {
                "success": False,
                "message": "No chunks were created from the documents",
                "document_ids": [],
            }

        # Step 3: Embed chunks
        try:
            embeddings = self.embedder.embed_texts(chunks)
            logger.info(f"Created {len(embeddings)} embeddings")
        except Exception as e:
            logger.error(f"Error embedding chunks: {str(e)}")
            return {
                "success": False,
                "message": f"Error embedding chunks: {str(e)}",
                "document_ids": [],
            }

        # Step 4: Ingest embeddings into vector database
        try:
            # Use the ingestor service to store the embeddings
            result = self.ingestor.ingest_texts(chunks, chunks_metadata)
            logger.info(f"Ingestion result: {result['message']}")
            return result
        except Exception as e:
            logger.error(f"Error ingesting embeddings: {str(e)}")
            return {
                "success": False,
                "message": f"Error ingesting embeddings: {str(e)}",
                "document_ids": [],
            }

    def _get_file_type(self, file_path: str) -> str:
        """
        Get the file type from a file path.

        Args:
            file_path: Path to the file.

        Returns:
            The file type (e.g., "pdf", "text", "html").
        """
        _, ext = os.path.splitext(file_path.lower())
        return ext[1:] if ext else "text"  # Remove the leading dot

    def search(
        self, query: str, top_k: int = 5, filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query.

        Args:
            query: The query text.
            top_k: The number of results to return.
            filter: Optional filter to apply to the search.

        Returns:
            A list of dictionaries containing the search results.
        """
        return self.ingestor.search(query, top_k, filter)


def main():
    """
    Main function to demonstrate the RAG pipeline.
    """
    # Sample documents for demonstration
    sample_documents = [
        # Sample text content
        """
        # Retrieval-Augmented Generation (RAG)
        
        Retrieval-Augmented Generation (RAG) is a technique that combines retrieval-based and generation-based approaches
        for natural language processing tasks. It enhances large language models by retrieving relevant information from
        external knowledge sources before generating responses.
        
        ## How RAG Works
        
        1. **Retrieval**: Given a query, retrieve relevant documents from a corpus.
        2. **Augmentation**: Augment the query with the retrieved documents.
        3. **Generation**: Generate a response based on the augmented query.
        
        ## Benefits of RAG
        
        - Improved accuracy
        - Reduced hallucinations
        - Better factual grounding
        - More up-to-date information
        """,
        # Sample JSON content
        """
        {
            "title": "Vector Databases",
            "description": "A comparison of vector databases for RAG applications",
            "databases": [
                {
                    "name": "Qdrant",
                    "description": "An open-source vector search engine",
                    "features": ["High performance", "Scalable", "Python API"]
                },
                {
                    "name": "FAISS",
                    "description": "A library for efficient similarity search",
                    "features": ["Fast", "Scalable", "Memory-efficient"]
                },
                {
                    "name": "Milvus",
                    "description": "An open-source vector database",
                    "features": ["Cloud-native", "Scalable", "High performance"]
                }
            ]
        }
        """,
        # Sample HTML content
        """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Text Splitter Example</title>
        </head>
        <body>
            <h1>Text Splitter Example</h1>
            <p>This is an example of HTML content that will be split by the HTML splitter.</p>
            
            <h2>Features</h2>
            <ul>
                <li>Splits HTML content based on tags</li>
                <li>Preserves the structure of the HTML</li>
                <li>Handles nested elements</li>
            </ul>
            
            <h2>How it Works</h2>
            <p>The HTML splitter uses BeautifulSoup to parse the HTML and extract text from specified tags.</p>
            <p>It can be configured to split by different tags, such as paragraphs, divs, or sections.</p>
        </body>
        </html>
        """,
    ]

    # Document types
    document_types = ["markdown", "json", "html"]

    # Initialize the RAG pipeline
    pipeline = RAGPipeline()

    # Process the documents
    result = pipeline.process_documents(sample_documents, document_types)

    # Print the result
    print(f"\nProcessing result: {result['message']}")
    print(f"Document IDs: {result['document_ids']}")

    # Perform a search
    query = "What is RAG and how does it work?"
    search_results = pipeline.search(query, top_k=3)

    # Print the search results
    print(f"\nSearch results for query: '{query}'")
    for i, result in enumerate(search_results):
        print(f"\nResult {i + 1}:")
        print(f"Document: {result['document'][:150]}...")
        print(f"Metadata: {result['metadata']}")
        print(f"Distance: {result['distance']}")


if __name__ == "__main__":
    main()
