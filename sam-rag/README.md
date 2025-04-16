# Solace Agent Mesh RAG

A document-ingesting agent that monitors specified directories, keeping stored documents up to date in a vector database for Retrieval-Augmented Generation (RAG) queries.

## Add a RAG Agent to SAM

Add the plugin to your SAM instance

```sh
solace-agent-mesh plugin add sam_rag --pip -u git+https://github.com/SolaceLabs/solace-agent-mesh-core-plugins#subdirectory=sam-rag
```

To instantiate the agent, you can use the following code: (you can repeat this step to connect to multiple collections/databases)

```sh
solace-agent-mesh add agent rag --copy-from sam_rag
```

Rebuild the Solace Agent Mesh to add the agent configurations.
```sh
solace-agent-mesh build
```

## Configuration Parameters

The RAG agent is configured through the `configs/agents/rag.yaml` file. Below is a detailed explanation of the configuration parameters:

### Scanner Configuration

The scanner monitors directories for documents to ingest into the vector database.

```yaml
scanner:
  batch: true                # Process documents in batch mode
  use_memory_storage: true   # Use in-memory storage for tracking files
  source:
    type: filesystem         # Source type (filesystem)
    directories:
      - "DIRECTORY PATH"     # Path to directory containing documents
    filters:
      file_formats:          # Supported file formats
        - ".txt"
        - ".pdf"
        - ".docx"
        - ".md"
        - ".html"
        - ".csv"
        - ".json"
        - ".odt"
        - ".xlsx"
        - ".xls"
      max_file_size: 10240   # Maximum file size in KB (10MB)
  database:                  # Database for storing metadata
    type: postgresql
    dbname: rag_metadata
    host: localhost
    port: 5432
    user: admin
    password: admin
  schedule:
    interval: 60             # Scanning interval in seconds
```

### Preprocessor Configuration

The preprocessor cleans and normalizes text from different document types.

```yaml
preprocessor:
  default_preprocessor:      # Default settings for all document types
    type: enhanced
    params:
      lowercase: true              # Convert text to lowercase
      normalize_whitespace: true   # Normalize whitespace characters
      remove_stopwords: false      # Remove common stopwords
      remove_punctuation: false    # Remove punctuation
      remove_numbers: false        # Remove numeric characters
      remove_non_ascii: false      # Remove non-ASCII characters
      remove_urls: true            # Remove URLs
      remove_emails: false         # Remove email addresses
      remove_html_tags: false      # Remove HTML tags
  
  preprocessors:             # Type-specific preprocessor settings
    # Text file configurations
    text:
      type: text
      params:
        # Text-specific preprocessing parameters
    
    # Document file configurations
    pdf:
      type: document
      params:
        # PDF-specific preprocessing parameters
    
    # Additional file type configurations for:
    # docx, odt, json, html, markdown, csv, xls
```

### Text Splitter Configuration

The text splitter breaks documents into smaller chunks for embedding.

```yaml
splitter:
  default_splitter:          # Default settings for all document types
    type: character
    params:
      chunk_size: 4096       # Size of each chunk
      chunk_overlap: 800     # Overlap between chunks
      separator: " "         # Text separator
  
  splitters:                 # Type-specific splitter settings
    # Text file configurations
    text:
      type: character
      params:
        chunk_size: 4096
        chunk_overlap: 800
        separator: " "
        is_separator_regex: false
        keep_separator: true
        strip_whitespace: true
    
    # Additional file type configurations for:
    # txt, json, html, markdown, csv
```

### Embedding Configuration

Settings for generating vector embeddings from text chunks.

```yaml
embedding:
  embedder_type: "openai"    # Type of embedding model
  embedder_params:
    model: ${OPENAI_EMBEDDING_MODEL}
    api_key: ${OPENAI_API_KEY}
    base_url: ${OPENAI_API_ENDPOINT}
    batch_size: 1
  normalize_embeddings: True # Whether to normalize embedding vectors
```

### Vector Database Configuration

Settings for the vector database used to store and retrieve embeddings.

```yaml
vector_db:
  # Qdrant configuration
  db_type: "qdrant"
  db_params:
    url: ${QDRANT_URL}
    api_key: ${QDRANT_API_KEY}
    collection_name: ${QDRANT_COLLECTION, "documents"}
    embedding_dimension: ${QDRANT_EMBEDDING_DIMENSION, 1024}
  
  # Alternative vector database options include:
  # - Chroma DB
  # - Pinecone
  # - Weaviate
  # - Milvus
  # - FAISS
  # - PostgreSQL with pgvector
  # - SQLite with sqlite-vss
```

### LLM Configuration

Settings for the language models used for augmentation.

```yaml
llm:
  load_balancer:
    - model_name: "gpt-4o"   # Model alias
      litellm_params:
        model: openai/${OPENAI_MODEL_NAME}
        api_key: ${OPENAI_API_KEY}
        api_base: ${OPENAI_API_ENDPOINT}
        temperature: 0.01
```

### Retrieval Configuration

Settings for document retrieval.

```yaml
retrieval:
  top_k: 7                   # Number of chunks to retrieve
```
