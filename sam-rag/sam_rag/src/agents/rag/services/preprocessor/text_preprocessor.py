"""
Text preprocessing utilities for cleaning and normalizing text.
"""

import re
import unicodedata
from typing import Dict, Any


class TextPreprocessor:
    """
    Utility class for text preprocessing operations.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the text preprocessor with the given configuration.

        Args:
            config: A dictionary containing configuration parameters.
                - lowercase: Whether to convert text to lowercase.
                - normalize_unicode: Whether to normalize Unicode characters.
                - normalize_whitespace: Whether to normalize whitespace.
                - remove_punctuation: Whether to remove punctuation.
                - remove_special_chars: Whether to remove special characters.
                - remove_urls: Whether to remove URLs.
                - remove_html_tags: Whether to remove HTML tags.
                - remove_numbers: Whether to remove numbers.
                - remove_non_ascii: Whether to remove non-ASCII characters.
        """
        self.config = config or {}

        # Default configuration
        self.lowercase = self.config.get("lowercase", True)
        self.normalize_unicode = self.config.get("normalize_unicode", True)
        self.normalize_whitespace = self.config.get("normalize_whitespace", True)
        self.remove_punctuation = self.config.get("remove_punctuation", True)
        self.remove_special_chars = self.config.get("remove_special_chars", True)
        self.remove_urls = self.config.get("remove_urls", True)
        self.remove_html_tags = self.config.get("remove_html_tags", True)
        self.remove_numbers = self.config.get("remove_numbers", False)
        self.remove_non_ascii = self.config.get("remove_non_ascii", False)

    def preprocess(self, text: str) -> str:
        """
        Apply preprocessing steps to clean and normalize text.

        Args:
            text: The input text to preprocess.

        Returns:
            Preprocessed text.
        """
        if not text:
            return ""

        # Unicode normalization
        if self.normalize_unicode:
            text = self._normalize_unicode(text)

        # Lowercase conversion
        if self.lowercase:
            text = text.lower()

        # Remove URLs
        if self.remove_urls:
            text = self._remove_urls(text)

        # Remove HTML tags
        if self.remove_html_tags:
            text = self._remove_html_tags(text)

        # Remove special characters
        if self.remove_special_chars:
            text = self._remove_special_chars(text)

        # Remove punctuation
        if self.remove_punctuation:
            text = self._remove_punctuation(text)

        # Remove numbers
        if self.remove_numbers:
            text = self._remove_numbers(text)

        # Remove non-ASCII characters
        if self.remove_non_ascii:
            text = self._remove_non_ascii(text)

        # Normalize whitespace
        if self.normalize_whitespace:
            text = self._normalize_whitespace(text)

        return text

    def _normalize_unicode(self, text: str) -> str:
        """
        Normalize Unicode characters to their canonical form.

        Args:
            text: The input text.

        Returns:
            Text with normalized Unicode characters.
        """
        return unicodedata.normalize("NFKC", text)

    def _remove_urls(self, text: str) -> str:
        """
        Remove URLs from text.

        Args:
            text: The input text.

        Returns:
            Text with URLs removed.
        """
        url_pattern = r"https?://\S+|www\.\S+"
        return re.sub(url_pattern, " ", text)

    def _remove_html_tags(self, text: str) -> str:
        """
        Remove HTML tags from text.

        Args:
            text: The input text.

        Returns:
            Text with HTML tags removed.
        """
        html_pattern = r"<.*?>"
        return re.sub(html_pattern, " ", text)

    def _remove_special_chars(self, text: str) -> str:
        """
        Remove special characters from text.

        Args:
            text: The input text.

        Returns:
            Text with special characters removed.
        """
        special_chars_pattern = r"[^\w\s]"
        return re.sub(special_chars_pattern, " ", text)

    def _remove_punctuation(self, text: str) -> str:
        """
        Remove punctuation from text, except for specified characters.

        Preserves:
        - Standard punctuation: ., ,, ;, :, !, ?, ', ", -
        - Brackets and parentheses: (), [], {}, <>
        - Mathematical symbols: +, *, /, =, %
        - Currency symbols: $, €, £, ¥

        Args:
            text: The input text.

        Returns:
            Text with unwanted punctuation removed.
        """
        # Define punctuation to keep
        preserved_punct = r".,;:!?\'\"\-\(\)\[\]\{\}<>\+\*/=\%\$€£¥"
        punctuation_pattern = f"[^\\w\\s{preserved_punct}]"
        return re.sub(punctuation_pattern, " ", text)

    def _remove_numbers(self, text: str) -> str:
        """
        Remove numbers from text.

        Args:
            text: The input text.

        Returns:
            Text with numbers removed.
        """
        return re.sub(r"\d+", " ", text)

    def _remove_non_ascii(self, text: str) -> str:
        """
        Remove non-ASCII characters from text.

        Args:
            text: The input text.

        Returns:
            Text with non-ASCII characters removed.
        """
        return re.sub(r"[^\x00-\x7F]+", " ", text)

    def _normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace in text.

        Args:
            text: The input text.

        Returns:
            Text with normalized whitespace.
        """
        # Replace multiple whitespace characters with a single space
        text = re.sub(r"\s+", " ", text)
        # Trim leading and trailing whitespace
        return text.strip()
