"""
Text splitters for unstructured text.
"""

from typing import Dict, Any, List
from .splitter_base import SplitterBase

try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


class CharacterTextSplitter(SplitterBase):
    """
    Split text by characters.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the character text splitter.

        Args:
            config: A dictionary containing configuration parameters.
                - chunk_size: The size of each chunk (default: 1000).
                - chunk_overlap: The overlap between chunks (default: 200).
                - separator: The separator to use for splitting (default: "\n\n").
        """
        super().__init__(config)
        self.chunk_size = self.config.get("chunk_size", 1000)
        self.chunk_overlap = self.config.get("chunk_overlap", 200)
        self.separator = self.config.get("separator", "\n\n")

    def split_text(self, text: str) -> List[str]:
        """
        Split the text into chunks.

        Args:
            text: The text to split.

        Returns:
            A list of text chunks.
        """
        if not text:
            return []

        # Split the text by separator
        splits = text.split(self.separator)

        # Initialize chunks
        chunks = []
        current_chunk = []
        current_length = 0

        # Process each split
        for split in splits:
            # If adding this split would exceed the chunk size, add the current chunk to chunks
            if current_length + len(split) > self.chunk_size and current_chunk:
                chunks.append(self.separator.join(current_chunk))
                # Keep some overlap for context
                overlap_splits = current_chunk[-self.chunk_overlap :]
                current_chunk = overlap_splits
                current_length = self.chunk_overlap

            # Add the current split to the current chunk
            current_chunk.append(split)
            current_length += len(split)

        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(self.separator.join(current_chunk))

        return chunks

    def can_handle(self, data_type: str) -> bool:
        """
        Check if this splitter can handle the given data type.

        Args:
            data_type: The type of data to split.

        Returns:
            True if this splitter can handle the data type, False otherwise.
        """
        return data_type.lower() in ["text", "txt", "plaintext"]


class RecursiveCharacterTextSplitter(SplitterBase):
    """
    Split text recursively by different separators.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the recursive character text splitter.

        Args:
            config: A dictionary containing configuration parameters.
                - chunk_size: The size of each chunk (default: 1000).
                - chunk_overlap: The overlap between chunks (default: 200).
                - separators: The separators to use for splitting (default: ["\n\n", "\n", " "]).
        """
        super().__init__(config)
        self.chunk_size = self.config.get("chunk_size", 1000)
        self.chunk_overlap = self.config.get("chunk_overlap", 200)
        self.separators = self.config.get("separators", ["\n\n", "\n", " "])

    def split_text(self, text: str) -> List[str]:
        """
        Split the text into chunks recursively.

        Args:
            text: The text to split.

        Returns:
            A list of text chunks.
        """
        if not text:
            return []

        # Use the first separator to split the text
        separator = self.separators[0]
        splits = text.split(separator)

        # If we're at the final separator or the splits are small enough, join and return
        if len(self.separators) == 1 or all(len(s) <= self.chunk_size for s in splits):
            return self._merge_splits(splits, separator)

        # Otherwise, recursively split each piece using the next separator
        chunks = []
        for split in splits:
            if len(split) > self.chunk_size:
                # Create a new splitter with the remaining separators
                sub_splitter = RecursiveCharacterTextSplitter(
                    {
                        "chunk_size": self.chunk_size,
                        "chunk_overlap": self.chunk_overlap,
                        "separators": self.separators[1:],
                    }
                )
                # Recursively split this piece
                sub_chunks = sub_splitter.split_text(split)
                chunks.extend(sub_chunks)
            else:
                chunks.append(split)

        # Merge the chunks with the current separator
        return self._merge_splits(chunks, separator)

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """
        Merge the splits into chunks of the desired size.

        Args:
            splits: The text splits.
            separator: The separator to use for joining.

        Returns:
            A list of text chunks.
        """
        chunks = []
        current_chunk = []
        current_length = 0

        for split in splits:
            # If adding this split would exceed the chunk size, add the current chunk to chunks
            if current_length + len(split) > self.chunk_size and current_chunk:
                chunks.append(separator.join(current_chunk))
                # Keep some overlap for context
                overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                current_chunk = current_chunk[overlap_start:]
                current_length = sum(len(s) for s in current_chunk)

            # Add the current split to the current chunk
            current_chunk.append(split)
            current_length += len(split)

        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(separator.join(current_chunk))

        return chunks

    def can_handle(self, data_type: str) -> bool:
        """
        Check if this splitter can handle the given data type.

        Args:
            data_type: The type of data to split.

        Returns:
            True if this splitter can handle the data type, False otherwise.
        """
        return data_type.lower() in ["text", "txt", "plaintext"]


class TokenTextSplitter(SplitterBase):
    """
    Split text by tokens using tiktoken.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the token text splitter.

        Args:
            config: A dictionary containing configuration parameters.
                - chunk_size: The size of each chunk in tokens (default: 500).
                - chunk_overlap: The overlap between chunks in tokens (default: 100).
                - encoding_name: The name of the tiktoken encoding to use (default: "cl100k_base").
        """
        super().__init__(config)
        self.chunk_size = self.config.get("chunk_size", 500)
        self.chunk_overlap = self.config.get("chunk_overlap", 100)
        self.encoding_name = self.config.get("encoding_name", "cl100k_base")

        if not TIKTOKEN_AVAILABLE:
            raise ImportError(
                "The tiktoken package is required for TokenTextSplitter. "
                "Please install it with `pip install tiktoken`."
            )

        self.tokenizer = tiktoken.get_encoding(self.encoding_name)

    def split_text(self, text: str) -> List[str]:
        """
        Split the text into chunks by tokens.

        Args:
            text: The text to split.

        Returns:
            A list of text chunks.
        """
        if not text:
            return []

        # Tokenize the text
        tokens = self.tokenizer.encode(text)

        # Split the tokens into chunks
        chunks = []
        for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
            # Get the tokens for this chunk
            chunk_tokens = tokens[i : i + self.chunk_size]
            # Decode the tokens back to text
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)

        return chunks

    def can_handle(self, data_type: str) -> bool:
        """
        Check if this splitter can handle the given data type.

        Args:
            data_type: The type of data to split.

        Returns:
            True if this splitter can handle the data type, False otherwise.
        """
        return data_type.lower() in ["text", "txt", "plaintext"]
