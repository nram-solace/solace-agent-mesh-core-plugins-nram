# Document Preprocessor

A generic document preprocessor component that converts various file formats to clean text for embedding and RAG applications.

## Features

- Supports multiple document formats:
  - PDF (`.pdf`)
  - Microsoft Word (`.doc`, `.docx`)
  - HTML (`.html`, `.htm`)
  - Plain text (`.txt`)
  - Markdown (`.md`)
  - CSV (`.csv`)
  - XML (`.xml`)
  - JSON (`.json`)
  - YAML (`.yaml`, `.yml`)
  - Excel (`.xls`, `.xlsx`)
  - OpenDocument Text (`.odt`)

- Text preprocessing capabilities:
  - Lowercasing
  - Unicode normalization
  - Whitespace normalization
  - Punctuation removal
  - Special character removal
  - URL removal
  - HTML tag removal
  - Number removal
  - Non-ASCII character removal

## Architecture

The preprocessor is designed with a modular architecture:

- `PreprocessorBase`: Abstract base class for document preprocessors
- `TextPreprocessor`: Handles text cleaning and normalization
- `Document-specific preprocessors`: Handle different file formats (PDF, DOCX, etc.)
- `PreprocessorService`: Main service that orchestrates preprocessing

## Usage

### Basic Usage

```python
from src.agents.rag.services.preprocessor import PreprocessorService

# Create a preprocessor service with default configuration
preprocessor = PreprocessorService()

# Preprocess a single file
text = preprocessor.preprocess_file("/path/to/document.pdf")

# Preprocess multiple files
results = preprocessor.preprocess_files([
    "/path/to/document1.pdf",
    "/path/to/document2.docx",
    "/path/to/document3.txt"
])

# Process results
for file_path, text in results:
    if text:
        print(f"Successfully preprocessed: {file_path}")
        # Use the preprocessed text for embedding, etc.
    else:
        print(f"Failed to preprocess: {file_path}")
```

### Custom Configuration

You can customize the preprocessing behavior:

```python
config = {
    "lowercase": True,
    "normalize_unicode": True,
    "normalize_whitespace": True,
    "remove_punctuation": True,
    "remove_special_chars": True,
    "remove_urls": True,
    "remove_html_tags": True,
    "remove_numbers": False,  # Keep numbers
    "remove_non_ascii": False  # Keep non-ASCII characters
}

preprocessor = PreprocessorService(config)
```

## Dependencies

The preprocessor uses different libraries depending on the file format:

- PDF: `PyPDF2`
- DOCX: `python-docx`
- HTML: `beautifulsoup4`
- Excel: `pandas`
- ODT: `odfpy`

These dependencies are imported only when needed, so you only need to install the ones you'll use.

## Example

See `example.py` for a complete example of how to use the preprocessor.

To run the example:

```bash
python -m src.agents.rag.services.preprocessor.example /path/to/file1.pdf /path/to/file2.docx
```

## Integration with RAG

The preprocessor is integrated with the RAG system through the `IngestionAction` class, which uses the preprocessor to convert documents to text before ingestion into the vector database.
