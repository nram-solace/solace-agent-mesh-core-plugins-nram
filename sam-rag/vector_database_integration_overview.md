# Vector Database Integration Overview in the RAG System

This document outlines how vector databases are supported, configured, and utilized for data ingestion and retrieval within the RAG (Retrieval Augmented Generation) system.

## I. Supported Vector Databases

Based on the system's architecture, specifically the imports in [`src/agents/rag/services/database/vector_db_service.py`](src/agents/rag/services/database/vector_db_service.py:8-14) and the factory method within it, the following vector databases are supported:

*   **ChromaDB** (This is the default if `db_type` is not specified in the configuration)
*   **PineconeDB**
*   **QdrantDB**
*   **RedisDB**
*   **PgVectorDB** (PostgreSQL with the pgvector extension)

## II. Configuration

Vector database configuration is managed within the [`configs/agents/rag.yaml`](configs/agents/rag.yaml) file, under the main `vector_db` key (see line [`configs/agents/rag.yaml:281`](configs/agents/rag.yaml:281)). The structure involves two primary sub-keys:

*   **`db_type`**: A string that specifies which database implementation to use (e.g., `"qdrant"`, `"chroma"`, `"pinecone"`).
*   **`db_params`**: A dictionary containing parameters specific to the chosen `db_type`. These parameters are typically, and recommended to be, populated using environment variables for security and flexibility.

### Configuration Examples:

**1. Qdrant (from [`configs/agents/rag.yaml:283-288`](configs/agents/rag.yaml:283)):**
```yaml
vector_db:
  db_type: "qdrant"
  db_params:
    url: ${QDRANT_URL}
    api_key: ${QDRANT_API_KEY}
    collection_name: ${QDRANT_COLLECTION}
    embedding_dimension: ${QDRANT_EMBEDDING_DIMENSION}
```

**2. ChromaDB (from [`configs/agents/rag.yaml:291-298`](configs/agents/rag.yaml:291), shown commented out):**
```yaml
# vector_db:
#   db_type: "chroma"
#   db_params:
#     host: ${CHROMA_HOST}
#     port: ${CHROMA_PORT}
#     collection_name: ${CHROMA_COLLECTION}
#     persist_directory: ${CHROMA_PERSIST_DIR, "./chroma_db"} # Default path if env var not set
#     embedding_dimension: ${CHROMA_EMBEDDING_DIMENSION} # Ensure this matches your embedding model
```

Similar configuration structures are present for Pinecone and PgVector, requiring parameters such as API keys, index/table names, connection details, and crucially, the `embedding_dimension` which must match the output dimension of the embedding model being used.

The [`VectorDBService`](src/agents/rag/services/database/vector_db_service.py) is responsible for reading this configuration block. It then instantiates the appropriate database client class found in [`src/agents/rag/services/database/vector_db_implementations.py`](src/agents/rag/services/database/vector_db_implementations.py). Each specific implementation (e.g., `QdrantDB`, `PineconeDB`) uses its respective `db_params` to establish a connection to the database. If the target collection or index does not already exist, the client typically attempts to create it, using the provided `embedding_dimension` and a default distance metric (commonly Cosine similarity).

## III. Ingesting Embedded Data

The process of ingesting data into the vector database involves several components:

1.  **Data Preparation (Handled by the RAG Pipeline):**
    *   **Document Scanning:** The `FileChangeTracker` ([`src/agents/rag/services/scanner/file_tracker.py`](src/agents/rag/services/scanner/file_tracker.py)) monitors configured sources for new or modified documents.
    *   **Preprocessing:** The `PreprocessorService` ([`src/agents/rag/services/preprocessor/preprocessor_service.py`](src/agents/rag/services/preprocessor/preprocessor_service.py)) cleans and prepares the raw content of these documents.
    *   **Chunking:** The `SplitterService` ([`src/agents/rag/services/splitter/splitter_service.py`](src/agents/rag/services/splitter/splitter_service.py)) divides the preprocessed content into smaller, manageable text chunks.
    *   **Embedding:** The `EmbedderService` ([`src/agents/rag/services/embedder/embedder_service.py`](src/agents/rag/services/embedder/embedder_service.py)) generates numerical vector embeddings for each of these text chunks.
    These steps are orchestrated by the `RAGAgentComponent` ([`src/agents/rag/rag_agent_component.py`](src/agents/rag/rag_agent_component.py)) and the `Pipeline` ([`src/agents/rag/services/pipeline/pipeline.py`](src/agents/rag/services/pipeline/pipeline.py)).

2.  **Role of `IngestionService`:**
    The [`IngestionService`](src/agents/rag/services/ingestor/ingestion_service.py) acts as the bridge to the vector database. It receives the pre-processed text chunks and their corresponding vector embeddings from the preceding pipeline stages.

3.  **Storing Data in the Vector Database:**
    *   The `IngestionService.ingest_embeddings()` method (defined in [`src/agents/rag/services/ingestor/ingestion_service.py:82`](src/agents/rag/services/ingestor/ingestion_service.py:82)) is invoked with the lists of texts, embeddings, and any associated metadata.
    *   This method, in turn, calls `VectorDBService.add_documents()` (defined in [`src/agents/rag/services/database/vector_db_service.py:58`](src/agents/rag/services/database/vector_db_service.py:58)).
    *   The `VectorDBService` then delegates this call to the `add_documents()` method of the currently configured and instantiated vector database implementation (e.g., `QdrantDB.add_documents()` from [`src/agents/rag/services/database/vector_db_implementations.py:346`](src/agents/rag/services/database/vector_db_implementations.py:346)).
    *   The specific database implementation (e.g., `QdrantDB`, `PineconeDB`) takes over:
        *   It formats the incoming data (text chunks, embeddings, metadata) into the precise structure required by its native client library (e.g., Qdrant uses `PointStruct`).
        *   It is common practice for these implementations to store the original text chunk as part of the metadata within the vector database itself, allowing for direct retrieval of content alongside similarity scores.
        *   Finally, it upserts (updates or inserts) this formatted data into the designated collection or index within the vector database.

## IV. Retrieving Chunks from the Vector Database

The retrieval of relevant chunks is primarily managed by the [`Retriever`](src/agents/rag/services/rag/retriever.py) component:

1.  **Query Input:** The process begins when the `Retriever.retrieve()` method (defined in [`src/agents/rag/services/rag/retriever.py:48`](src/agents/rag/services/rag/retriever.py:48)) receives a user's textual query.

2.  **Query Embedding:**
    *   The `Retriever` utilizes its internally initialized `EmbedderService` instance.
    *   It calls `Retriever.get_query_embedding()` (defined in [`src/agents/rag/services/rag/retriever.py:85`](src/agents/rag/services/rag/retriever.py:85)), which in turn uses `EmbedderService.embed_texts()` to convert the natural language query into a numerical vector embedding.

3.  **Searching the Vector Database:**
    *   With the query embedding generated, the `Retriever` calls `VectorDBService.search()` (defined in [`src/agents/rag/services/database/vector_db_service.py:79`](src/agents/rag/services/database/vector_db_service.py:79)).
    *   It passes the `query_embedding`, the `top_k` value (number of results to retrieve, configured in [`configs/agents/rag.yaml`](configs/agents/rag.yaml) under the `retrieval` key), and any optional metadata filters to this service.
    *   The `VectorDBService` delegates the search operation to the `search()` method of the active vector database implementation (e.g., `QdrantDB.search()` from [`src/agents/rag/services/database/vector_db_implementations.py:405`](src/agents/rag/services/database/vector_db_implementations.py:405)).
    *   The specific database implementation then uses its client library to perform a similarity search (e.g., k-nearest neighbors) against the stored vectors using the query vector.

4.  **Returning Results:**
    *   The search results, typically a list of dictionaries where each dictionary represents a retrieved chunk and includes its text content, metadata, and a similarity score (or distance), are returned up the call stack. This information is then used by the `AugmentationService` to construct the final response to the user.

This detailed flow illustrates the robust and pluggable architecture for vector database interactions within the RAG system, allowing flexibility in choosing the underlying database technology while maintaining a consistent operational interface.