
# Text Splitter Service

A service for splitting documents into chunks for embedding in RAG applications.

## Features

- Supports multiple text splitting strategies:
  - Character-based splitting
  - Recursive character-based splitting with multiple separators
  - Token-based splitting (requires tiktoken)
- Handles various document formats:
  - Plain text
  - JSON
  - HTML
  - Markdown
  - CSV
- Provides a unified interface for splitting text
- Automatically selects the appropriate splitter based on the document type

## Text Splitters

### CharacterTextSplitter

Splits text by characters using a specified separator.

```python
from src.agents.rag.services.splitter import CharacterTextSplitter

splitter = CharacterTextSplitter({
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "separator": "\n\n"
})

chunks = splitter.split_text(text)
```

### RecursiveCharacterTextSplitter

Splits text recursively by different separators. This is the recommended default splitter for most use cases.

```python
from src.agents.rag.services.splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter({
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "separators": ["\n\n", "\n", " ", ""]
})

chunks = splitter.split_text(text)
```

### TokenTextSplitter

Splits text by tokens using tiktoken. This is useful when you need to ensure chunks fit within token limits for LLMs.

```python
from src.agents.rag.services.splitter import TokenTextSplitter

splitter = TokenTextSplitter({
    "chunk_size": 500,
    "chunk_overlap": 100,
    "encoding_name": "cl100k_base"
})

chunks = splitter.split_text(text)
```

## Structured Data Splitters

### JSONSplitter

Splits JSON data into chunks.

```python
from src.agents.rag.services.splitter import JSONSplitter

splitter = JSONSplitter({
    "chunk_size": 1000,
    "chunk_overlap": 200
})

chunks = splitter.split_text(json_text)
```

### HTMLSplitter

Splits HTML data into chunks based on HTML tags.

```python
from src.agents.rag.services.splitter import HTMLSplitter

splitter = HTMLSplitter({
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "split_by_tag": ["div", "p", "section", "article"]
})

chunks = splitter.split_text(html_text)
```

### MarkdownSplitter

Splits Markdown data into chunks based on headings.

```python
from src.agents.rag.services.splitter import MarkdownSplitter

splitter = MarkdownSplitter({
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "split_by_heading": True
})

chunks = splitter.split_text(markdown_text)
```

### CSVSplitter

Splits CSV data into chunks based on rows.

```python
from src.agents.rag.services.splitter import CSVSplitter

splitter = CSVSplitter({
    "chunk_size": 100,  # rows
    "include_header": True
})

chunks = splitter.split_text(csv_text)
```

## SplitterService

The SplitterService provides a unified interface for splitting text and automatically selects the appropriate splitter based on the document type.

```python
from src.agents.rag.services.splitter import SplitterService

# Create a splitter service with default configuration
service = SplitterService()

# Split text with a specific data type
chunks = service.split_text(text, "markdown")

# Split a file (data type inferred from file extension)
chunks = service.split_file("/path/to/document.md")

# Split multiple files
results = service.split_files(["/path/to/document1.md", "/path/to/document2.html"])
```

## Custom Configuration

You can customize the SplitterService with your own configuration:

```python
config = {
    "splitters": {
        "text": {
            "type": "recursive_character",
            "params": {
                "chunk_size": 1500,
                "chunk_overlap": 300,
                "separators": ["\n\n", "\n", " ", ""]
            }
        },
        "json": {
            "type": "recursive_json",
            "params": {
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "include_metadata": True
            }
        }
    },
    "default_splitter": {
        "type": "recursive_character",
        "params": {
            "chunk_size": 1000,
            "chunk_overlap": 200
        }
    }
}

service = SplitterService(config)
```

## Example Usage

The module includes an example script that demonstrates how to use the text splitters:

```bash
# Run the example with default text
python -m src.agents.rag.services.splitter.text_splitter_example

# Run the example with a file
python -m src.agents.rag.services.splitter.text_splitter_example --file /path/to/document.txt

# Run the example with custom chunk size and overlap
python -m src.agents.rag.services.splitter.text_splitter_example --chunk-size 1500 --chunk-overlap 300

# Run the example with a specific data type
python -m src.agents.rag.services.splitter.text_splitter_example --data-type markdown
```

## Integration with RAG Pipeline

The text splitter is an essential component of the RAG pipeline:

1. **Document Preprocessing**: Convert documents to text
2. **Text Splitting**: Split text into chunks
3. **Embedding**: Convert chunks to vector embeddings
4. **Storage**: Store embeddings in a vector database
5. **Retrieval**: Retrieve relevant chunks based on a query
6. **Generation**: Generate responses based on retrieved chunks

Example integration:

```python
from src.agents.rag.services.preprocessor import DocumentProcessor
from src.agents.rag.services.splitter import SplitterService
from src.agents.rag.services.embedder import EmbedderService

# Process a document
processor = DocumentProcessor()
text = processor.process_document("/path/to/document.pdf")

# Split the text into chunks
splitter = SplitterService()
chunks = splitter.split_text(text, "text")

# Embed the chunks
embedder = EmbedderService()
embeddings = embedder.embed_texts(chunks)

# Now you can store the embeddings in a vector database
```

## Dependencies

The text splitter service uses different libraries depending on the splitter type:

- TokenTextSplitter: `tiktoken`
- HTMLSplitter: `beautifulsoup4`
- MarkdownSplitter: `markdown` (optional)

These dependencies are imported only when needed, so you only need to install the ones you'll use.
