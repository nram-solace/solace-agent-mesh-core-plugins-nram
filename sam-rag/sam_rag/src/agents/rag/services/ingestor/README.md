# Ingestor Service

The Ingestor Service is a component of the RAG (Retrieval-Augmented Generation) system that handles the ingestion of documents into a vector database. It integrates preprocessing, splitting, embedding, and vector database storage into a single workflow.

## Features

- Ingest documents from file paths
- Ingest texts directly
- Delete documents from the vector database
- Search for documents similar to a query
- Support for multiple vector databases:
  - ChromaDB
  - FAISS
  - Milvus
  - Pinecone
  - Weaviate
  - Qdrant
  - Redis (with vector search)
  - Elasticsearch (with vector support)
  - pgvector (PostgreSQL extension)

## Usage

### Initialization

```python
from src.agents.rag.services.ingestor.ingestor_service import IngestorService

# Initialize the ingestor service with configuration
config = {
    "preprocessor": {
        # Preprocessor configuration
    },
    "splitter": {
        # Text splitter configuration
    },
    "embedder": {
        # Embedder configuration
    },
    "vector_db": {
        "db_type": "chroma",  # Options: chroma, faiss, milvus, pinecone, weaviate, qdrant, redis, elasticsearch, pgvector
        "db_params": {
            # Database-specific parameters
        }
    }
}

ingestor = IngestorService(config)
```

### Ingesting Documents

```python
# Ingest documents from file paths
file_paths = [
    "/path/to/document1.pdf",
    "/path/to/document2.docx",
    "/path/to/document3.txt"
]

# Optional metadata for each document
metadata = [
    {"source": "document1.pdf", "author": "John Doe"},
    {"source": "document2.docx", "author": "Jane Smith"},
    {"source": "document3.txt", "author": "Bob Johnson"}
]

result = ingestor.ingest_documents(file_paths, metadata)
print(result["message"])  # Successfully ingested X chunks from Y documents
print(result["document_ids"])  # List of document IDs in the vector database
```

### Ingesting Texts

```python
# Ingest texts directly
texts = [
    "This is the first document.",
    "This is the second document.",
    "This is the third document."
]

# Optional metadata for each text
metadata = [
    {"source": "text1", "category": "example"},
    {"source": "text2", "category": "example"},
    {"source": "text3", "category": "example"}
]

# Optional IDs for each text
ids = ["doc1", "doc2", "doc3"]

result = ingestor.ingest_texts(texts, metadata, ids)
print(result["message"])  # Successfully ingested X chunks from Y texts
print(result["document_ids"])  # List of document IDs in the vector database
```

### Deleting Documents

```python
# Delete documents from the vector database
document_ids = ["doc1", "doc2", "doc3"]
ingestor.delete_documents(document_ids)
```

### Searching

```python
# Search for documents similar to a query
query = "What is RAG?"
results = ingestor.search(query, top_k=5)

for result in results:
    print(f"Document: {result['document']}")
    print(f"Metadata: {result['metadata']}")
    print(f"Distance: {result['distance']}")
    print()
```

## Vector Database Configuration

### ChromaDB

```python
config = {
    "vector_db": {
        "db_type": "chroma",
        "db_params": {
            "persist_directory": "./chroma_db",
            "collection_name": "documents",
            "embedding_dimension": 768
        }
    }
}
```

### FAISS

```python
config = {
    "vector_db": {
        "db_type": "faiss",
        "db_params": {
            "index_path": "./faiss_index",
            "embedding_dimension": 768,
            "index_type": "Flat"  # Options: Flat, IVF, HNSW
        }
    }
}
```

### Milvus

```python
config = {
    "vector_db": {
        "db_type": "milvus",
        "db_params": {
            "host": "localhost",
            "port": 19530,
            "collection_name": "documents",
            "embedding_dimension": 768
        }
    }
}
```

### Pinecone

```python
config = {
    "vector_db": {
        "db_type": "pinecone",
        "db_params": {
            "api_key": "your-api-key",
            "environment": "your-environment",
            "index_name": "your-index-name",
            "namespace": "default",
            "embedding_dimension": 768
        }
    }
}
```

### Weaviate

```python
config = {
    "vector_db": {
        "db_type": "weaviate",
        "db_params": {
            "url": "http://localhost:8080",
            "api_key": "your-api-key",  # Optional
            "class_name": "Document",
            "batch_size": 100
        }
    }
}
```

### Qdrant

```python
config = {
    "vector_db": {
        "db_type": "qdrant",
        "db_params": {
            "url": "http://localhost:6333",
            "api_key": "your-api-key",  # Optional
            "collection_name": "documents",
            "embedding_dimension": 768
        }
    }
}
```

### Redis

```python
config = {
    "vector_db": {
        "db_type": "redis",
        "db_params": {
            "host": "localhost",
            "port": 6379,
            "password": "your-password",  # Optional
            "index_name": "documents",
            "prefix": "doc:",
            "embedding_dimension": 768
        }
    }
}
```

### Elasticsearch

```python
config = {
    "vector_db": {
        "db_type": "elasticsearch",
        "db_params": {
            "hosts": ["http://localhost:9200"],
            "api_key": "your-api-key",  # Optional
            "username": "your-username",  # Optional
            "password": "your-password",  # Optional
            "index_name": "documents",
            "embedding_dimension": 768
        }
    }
}
```

### pgvector (PostgreSQL)

```python
config = {
    "vector_db": {
        "db_type": "pgvector",
        "db_params": {
            "connection_string": "postgresql://username:password@localhost:5432/database",
            "table_name": "documents",
            "embedding_dimension": 768
        }
    }
}
