# Embedder Service

A service for converting text chunks to vector embeddings for use in RAG applications.

## Features

- Supports multiple embedding providers:
  - Local embedders:
    - SentenceTransformer
    - HuggingFace
    - OpenAI-compatible local models
  - Cloud embedders:
    - OpenAI
    - Azure OpenAI
    - Cohere
    - Google Vertex AI
- Provides a unified interface for embedding text
- Supports batch processing for efficiency
- Includes utilities for embedding normalization and similarity calculation

## Usage

### Basic Usage with EmbedderService

```python
from src.agents.rag.services.embedder import EmbedderService

# Create an embedder service with default configuration (SentenceTransformer)
embedder = EmbedderService()

# Embed a single text
embedding = embedder.embed_text("This is a sample text to embed.")

# Embed multiple texts
embeddings = embedder.embed_texts([
    "This is the first text to embed.",
    "This is the second text to embed."
])

# Calculate similarity between two embeddings
similarity = embedder.cosine_similarity(embeddings[0], embeddings[1])
print(f"Similarity: {similarity}")
```

### Using Azure OpenAI Embedder

```python
from src.agents.rag.services.embedder import EmbedderService, AzureOpenAIEmbedder

# Method 1: Using EmbedderService
config = {
    "embedder_type": "azure_openai",
    "embedder_params": {
        "api_key": "your-azure-openai-api-key",
        "api_version": "2023-05-15",
        "endpoint": "https://your-resource-name.openai.azure.com",
        "deployment": "your-embedding-deployment-name"
    },
    "normalize_embeddings": True
}
embedder = EmbedderService(config)

# Method 2: Using AzureOpenAIEmbedder directly
azure_config = {
    "api_key": "your-azure-openai-api-key",
    "api_version": "2023-05-15",
    "endpoint": "https://your-resource-name.openai.azure.com",
    "deployment": "your-embedding-deployment-name"
}
embedder = AzureOpenAIEmbedder(azure_config)

# Embed text
embedding = embedder.embed_text("This is a sample text to embed.")
```

## Azure OpenAI Example

The module includes an example script that demonstrates how to use the Azure OpenAI embedder:

```bash
# Create a configuration file based on the template
cp src/agents/rag/services/embedder/azure_openai_config_template.json azure_config.json
# Edit the configuration file with your Azure OpenAI credentials
nano azure_config.json

# Run the example with just the configuration
python -m src.agents.rag.services.embedder.azure_openai_example azure_config.json

# Run the example with a document to process and embed
python -m src.agents.rag.services.embedder.azure_openai_example azure_config.json /path/to/document.pdf
```

## Integration with Document Processor

The embedder service can be integrated with the document processor to create a complete RAG pipeline:

```python
from src.agents.rag.services.preprocessor import DocumentProcessor
from src.agents.rag.services.embedder import EmbedderService

# Create a document processor
processor = DocumentProcessor()

# Process a document
text = processor.process_document("/path/to/document.pdf")

# Create an embedder service
embedder = EmbedderService({
    "embedder_type": "azure_openai",
    "embedder_params": {
        "api_key": "your-azure-openai-api-key",
        "api_version": "2023-05-15",
        "endpoint": "https://your-resource-name.openai.azure.com",
        "deployment": "your-embedding-deployment-name"
    }
})

# Embed the processed text
embedding = embedder.embed_text(text)

# Now you can store the embedding in a vector database for retrieval
```

## Dependencies

The embedder service uses different libraries depending on the embedder type:

- SentenceTransformer: `sentence-transformers`
- HuggingFace: `transformers`, `torch`
- OpenAI: `openai`
- Cohere: `cohere`
- Vertex AI: `google-cloud-aiplatform`, `vertexai`

These dependencies are imported only when needed, so you only need to install the ones you'll use.
