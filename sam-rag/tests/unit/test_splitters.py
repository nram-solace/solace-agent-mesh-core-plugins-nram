import os
import sys
import unittest
import json
from unittest.mock import patch, MagicMock
from typing import Dict, Any, List, Union

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.agents.rag.services.splitter import (
    SplitterBase,
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    JSONSplitter,
    RecursiveJSONSplitter,
    HTMLSplitter,
    MarkdownSplitter,
    CSVSplitter,
    SplitterService,
)


class TestSplitterBase(unittest.TestCase):
    """Test the SplitterBase abstract class."""

    def test_abstract_methods(self):
        """Test that SplitterBase cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            SplitterBase()

    def test_init_with_config(self):
        """Test initialization with a configuration."""

        # Create a concrete subclass for testing
        class ConcreteSplitter(SplitterBase):

            def split_text(self, text):
                return [text]

            def can_handle(self, data_type):
                return True

        config = {"key": "value"}
        splitter = ConcreteSplitter(config)
        self.assertEqual(splitter.config, config)

    def test_init_without_config(self):
        """Test initialization without a configuration."""

        # Create a concrete subclass for testing
        class ConcreteSplitter(SplitterBase):

            def split_text(self, text):
                return [text]

            def can_handle(self, data_type):
                return True

        splitter = ConcreteSplitter()
        self.assertEqual(splitter.config, {})


class TestCharacterTextSplitter(unittest.TestCase):
    """Test the CharacterTextSplitter class."""

    def setUp(self):
        """Set up test fixtures."""
        self.default_splitter = CharacterTextSplitter()
        self.custom_splitter = CharacterTextSplitter(
            {
                "chunk_size": 10,
                "chunk_overlap": 2,
                "separator": " ",
                "is_separator_regex": False,
                "keep_separator": True,
                "strip_whitespace": True,
            }
        )

    def test_init_default(self):
        """Test initialization with default parameters."""
        self.assertEqual(self.default_splitter.chunk_size, 1000)
        self.assertEqual(self.default_splitter.chunk_overlap, 200)
        self.assertEqual(self.default_splitter.separator, "\n")
        self.assertEqual(self.default_splitter.is_separator_regex, False)
        self.assertEqual(self.default_splitter.keep_separator, True)
        self.assertEqual(self.default_splitter.strip_whitespace, True)

    def test_init_custom(self):
        """Test initialization with custom parameters."""
        self.assertEqual(self.custom_splitter.chunk_size, 10)
        self.assertEqual(self.custom_splitter.chunk_overlap, 2)
        self.assertEqual(self.custom_splitter.separator, " ")
        self.assertEqual(self.custom_splitter.is_separator_regex, False)
        self.assertEqual(self.custom_splitter.keep_separator, True)
        self.assertEqual(self.custom_splitter.strip_whitespace, True)

    def test_split_text_empty(self):
        """Test splitting empty text."""
        result = self.default_splitter.split_text("")
        self.assertEqual(result, [])

    def test_split_text_small(self):
        """Test splitting text smaller than chunk size."""
        text = "This is a small text."
        result = self.default_splitter.split_text(text)
        self.assertEqual(result, [text])

    def test_split_text_large(self):
        """Test splitting text larger than chunk size."""
        text = "word1 word2 word3 word4 word5"
        result = self.custom_splitter.split_text(text)
        # Expected: Split by spaces, with chunk size 10 and overlap 2
        self.assertGreater(len(result), 1)
        # Check that each chunk is not larger than chunk_size
        for chunk in result:
            self.assertLessEqual(len(chunk), self.custom_splitter.chunk_size)

    def test_split_text_with_separator(self):
        """Test splitting text with a specific separator."""
        text = "part1\npart2\npart3\npart4"
        result = self.default_splitter.split_text(text)
        # Since the default separator is "\n", we expect the text to be split by newlines
        self.assertEqual(len(result), 4)

    def test_split_text_with_regex_separator(self):
        """Test splitting text with a regex separator."""
        splitter = CharacterTextSplitter(
            {
                "separator": r"\s+",
                "is_separator_regex": True,
                "chunk_size": 10,
                "chunk_overlap": 0,
            }
        )
        text = "word1  word2\tword3\nword4"
        result = splitter.split_text(text)
        # The regex \s+ should match any whitespace sequence
        self.assertEqual(len(result), 4)

    def test_can_handle(self):
        """Test the can_handle method."""
        self.assertTrue(self.default_splitter.can_handle("text"))
        self.assertTrue(self.default_splitter.can_handle("txt"))
        self.assertTrue(self.default_splitter.can_handle("plaintext"))
        self.assertTrue(self.default_splitter.can_handle("TEXT"))  # Case insensitive
        self.assertFalse(self.default_splitter.can_handle("json"))
        self.assertFalse(self.default_splitter.can_handle("html"))


class TestRecursiveCharacterTextSplitter(unittest.TestCase):
    """Test the RecursiveCharacterTextSplitter class."""

    def setUp(self):
        """Set up test fixtures."""
        self.default_splitter = RecursiveCharacterTextSplitter()
        self.custom_splitter = RecursiveCharacterTextSplitter(
            {
                "chunk_size": 10,
                "chunk_overlap": 2,
                "separators": ["\n\n", "\n", " ", ""],
            }
        )

    def test_init_default(self):
        """Test initialization with default parameters."""
        self.assertEqual(self.default_splitter.chunk_size, 1000)
        self.assertEqual(self.default_splitter.chunk_overlap, 200)
        self.assertEqual(self.default_splitter.separators, ["\n\n", "\n", " ", ""])

    def test_init_custom(self):
        """Test initialization with custom parameters."""
        self.assertEqual(self.custom_splitter.chunk_size, 10)
        self.assertEqual(self.custom_splitter.chunk_overlap, 2)
        self.assertEqual(self.custom_splitter.separators, ["\n\n", "\n", " ", ""])

    def test_split_text_empty(self):
        """Test splitting empty text."""
        result = self.default_splitter.split_text("")
        self.assertEqual(result, [])

    def test_split_text_small(self):
        """Test splitting text smaller than chunk size."""
        text = "This is a small text."
        result = self.default_splitter.split_text(text)
        self.assertEqual(result, [text])

    def test_split_text_with_paragraphs(self):
        """Test splitting text with paragraphs."""
        text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
        result = self.default_splitter.split_text(text)
        # Since the first separator is "\n\n", we expect the text to be split by paragraphs
        self.assertEqual(len(result), 3)

    def test_split_text_recursive(self):
        """Test recursive splitting of text."""
        # Create a text that will require recursive splitting
        text = "word1 word2\nword3 word4\n\nword5 word6\nword7 word8"
        splitter = RecursiveCharacterTextSplitter(
            {
                "chunk_size": 10,
                "chunk_overlap": 0,
                "separators": ["\n\n", "\n", " ", ""],
            }
        )
        result = splitter.split_text(text)
        # The text should be split first by "\n\n", then by "\n", and finally by " "
        self.assertGreater(len(result), 4)
        # Check that each chunk is not larger than chunk_size
        for chunk in result:
            self.assertLessEqual(len(chunk), splitter.chunk_size)

    def test_can_handle(self):
        """Test the can_handle method."""
        self.assertTrue(self.default_splitter.can_handle("text"))
        self.assertTrue(self.default_splitter.can_handle("txt"))
        self.assertTrue(self.default_splitter.can_handle("plaintext"))
        self.assertFalse(self.default_splitter.can_handle("json"))
        self.assertFalse(self.default_splitter.can_handle("html"))


class TestTokenTextSplitter(unittest.TestCase):
    """Test the TokenTextSplitter class."""

    @patch("src.agents.rag.services.splitter.text_splitter.TIKTOKEN_AVAILABLE", True)
    @patch("src.agents.rag.services.splitter.text_splitter.tiktoken")
    def setUp(self, mock_tiktoken):
        """Set up test fixtures with mocked tiktoken."""
        # Mock the tiktoken encoding
        self.mock_encoding = MagicMock()
        mock_tiktoken.get_encoding.return_value = self.mock_encoding

        # Create splitters
        self.default_splitter = TokenTextSplitter()
        self.custom_splitter = TokenTextSplitter(
            {"chunk_size": 10, "chunk_overlap": 2, "encoding_name": "test_encoding"}
        )

    def test_init_default(self):
        """Test initialization with default parameters."""
        self.assertEqual(self.default_splitter.chunk_size, 500)
        self.assertEqual(self.default_splitter.chunk_overlap, 100)
        self.assertEqual(self.default_splitter.encoding_name, "cl100k_base")

    def test_init_custom(self):
        """Test initialization with custom parameters."""
        self.assertEqual(self.custom_splitter.chunk_size, 10)
        self.assertEqual(self.custom_splitter.chunk_overlap, 2)
        self.assertEqual(self.custom_splitter.encoding_name, "test_encoding")

    @patch("src.agents.rag.services.splitter.text_splitter.TIKTOKEN_AVAILABLE", False)
    def test_init_without_tiktoken(self):
        """Test initialization when tiktoken is not available."""
        with self.assertRaises(ImportError):
            TokenTextSplitter()

    def test_split_text_empty(self):
        """Test splitting empty text."""
        result = self.default_splitter.split_text("")
        self.assertEqual(result, [])

    def test_split_text_small(self):
        """Test splitting text smaller than chunk size."""
        text = "This is a small text."
        # Mock the tokenizer to return a small number of tokens
        self.mock_encoding.encode.return_value = [1, 2, 3, 4, 5]
        result = self.default_splitter.split_text(text)
        self.assertEqual(result, [text])

    def test_split_text_large(self):
        """Test splitting text larger than chunk size."""
        text = "This is a large text that needs to be split into chunks."
        # Mock the tokenizer to return a large number of tokens
        tokens = list(range(1000))
        self.mock_encoding.encode.return_value = tokens
        self.mock_encoding.decode.side_effect = lambda x: text[: len(x)]

        result = self.default_splitter.split_text(text)
        # Since the tokens are more than chunk_size, we expect multiple chunks
        self.assertGreater(len(result), 1)

    def test_can_handle(self):
        """Test the can_handle method."""
        self.assertTrue(self.default_splitter.can_handle("text"))
        self.assertTrue(self.default_splitter.can_handle("txt"))
        self.assertTrue(self.default_splitter.can_handle("plaintext"))
        self.assertFalse(self.default_splitter.can_handle("json"))
        self.assertFalse(self.default_splitter.can_handle("html"))


class TestJSONSplitter(unittest.TestCase):
    """Test the JSONSplitter class."""

    def setUp(self):
        """Set up test fixtures."""
        self.default_splitter = JSONSplitter()
        self.custom_splitter = JSONSplitter({"chunk_size": 100, "chunk_overlap": 20})

        # Sample JSON data
        self.sample_json = {
            "name": "Test",
            "items": [1, 2, 3],
            "nested": {"key": "value", "list": [4, 5, 6]},
        }
        self.sample_json_str = json.dumps(self.sample_json)

    def test_init_default(self):
        """Test initialization with default parameters."""
        self.assertEqual(self.default_splitter.chunk_size, 1000)
        self.assertEqual(self.default_splitter.chunk_overlap, 200)

    def test_init_custom(self):
        """Test initialization with custom parameters."""
        self.assertEqual(self.custom_splitter.chunk_size, 100)
        self.assertEqual(self.custom_splitter.chunk_overlap, 20)

    def test_split_text_empty(self):
        """Test splitting empty text."""
        result = self.default_splitter.split_text("")
        self.assertEqual(result, [])

    def test_split_text_valid_json(self):
        """Test splitting valid JSON text."""
        result = self.default_splitter.split_text(self.sample_json_str)
        # Since the JSON is small, we expect a single chunk
        self.assertEqual(len(result), 1)
        # The result should be the formatted JSON
        self.assertIn("name", result[0])
        self.assertIn("items", result[0])
        self.assertIn("nested", result[0])

    def test_split_text_invalid_json(self):
        """Test splitting invalid JSON text."""
        invalid_json = "{invalid: json}"
        result = self.default_splitter.split_text(invalid_json)
        # For invalid JSON, it should fall back to the text splitter
        self.assertEqual(result, [invalid_json])

    def test_can_handle(self):
        """Test the can_handle method."""
        self.assertTrue(self.default_splitter.can_handle("json"))
        self.assertTrue(self.default_splitter.can_handle("JSON"))  # Case insensitive
        self.assertFalse(self.default_splitter.can_handle("text"))
        self.assertFalse(self.default_splitter.can_handle("html"))


class TestRecursiveJSONSplitter(unittest.TestCase):
    """Test the RecursiveJSONSplitter class."""

    def setUp(self):
        """Set up test fixtures."""
        self.default_splitter = RecursiveJSONSplitter()
        self.custom_splitter = RecursiveJSONSplitter(
            {"chunk_size": 100, "chunk_overlap": 20, "include_metadata": False}
        )

        # Sample JSON data
        self.sample_json = {
            "name": "Test",
            "items": [1, 2, 3],
            "nested": {"key": "value", "list": [4, 5, 6]},
        }
        self.sample_json_str = json.dumps(self.sample_json)

        # Large JSON data that will need to be split
        self.large_json = {"items": [{"id": i, "value": "x" * 1000} for i in range(10)]}
        self.large_json_str = json.dumps(self.large_json)

    def test_init_default(self):
        """Test initialization with default parameters."""
        self.assertEqual(self.default_splitter.chunk_size, 1000)
        self.assertEqual(self.default_splitter.chunk_overlap, 200)
        self.assertEqual(self.default_splitter.include_metadata, True)

    def test_init_custom(self):
        """Test initialization with custom parameters."""
        self.assertEqual(self.custom_splitter.chunk_size, 100)
        self.assertEqual(self.custom_splitter.chunk_overlap, 20)
        self.assertEqual(self.custom_splitter.include_metadata, False)

    def test_split_text_empty(self):
        """Test splitting empty text."""
        result = self.default_splitter.split_text("")
        self.assertEqual(result, [])

    def test_split_text_small_json(self):
        """Test splitting small JSON text."""
        result = self.default_splitter.split_text(self.sample_json_str)
        # Since the JSON is small, we expect it to be processed as a whole
        self.assertGreaterEqual(len(result), 1)

    def test_split_text_large_json(self):
        """Test splitting large JSON text that needs recursive processing."""
        # Use a smaller chunk size for this test
        splitter = RecursiveJSONSplitter({"chunk_size": 50})
        result = splitter.split_text(self.large_json_str)
        # The large JSON should be split into multiple chunks
        self.assertGreater(len(result), 1)

    def test_split_text_with_metadata(self):
        """Test splitting JSON text with metadata included."""
        result = self.default_splitter.split_text(self.sample_json_str)
        # With include_metadata=True, the chunks should include path information
        for chunk in result:
            if "items" in chunk or "nested" in chunk:
                # Check if any chunk contains path information
                self.assertTrue(
                    any(chunk.startswith(path) for path in ["items", "nested"])
                )

    def test_split_text_without_metadata(self):
        """Test splitting JSON text without metadata."""
        result = self.custom_splitter.split_text(self.sample_json_str)
        # With include_metadata=False, the chunks should not include path information
        for chunk in result:
            # Check that no chunk starts with path information
            self.assertFalse(chunk.startswith("items"))
            self.assertFalse(chunk.startswith("nested"))

    def test_split_text_invalid_json(self):
        """Test splitting invalid JSON text."""
        invalid_json = "{invalid: json}"
        result = self.default_splitter.split_text(invalid_json)
        # For invalid JSON, it should fall back to the text splitter
        self.assertEqual(result, [invalid_json])

    def test_can_handle(self):
        """Test the can_handle method."""
        self.assertTrue(self.default_splitter.can_handle("json"))
        self.assertTrue(self.default_splitter.can_handle("JSON"))  # Case insensitive
        self.assertFalse(self.default_splitter.can_handle("text"))
        self.assertFalse(self.default_splitter.can_handle("html"))


class TestHTMLSplitter(unittest.TestCase):
    """Test the HTMLSplitter class."""

    @patch(
        "src.agents.rag.services.splitter.structured_splitter.BEAUTIFULSOUP_AVAILABLE",
        True,
    )
    @patch("src.agents.rag.services.splitter.structured_splitter.BeautifulSoup")
    def setUp(self, mock_bs):
        """Set up test fixtures with mocked BeautifulSoup."""
        # Mock BeautifulSoup
        self.mock_soup = MagicMock()
        mock_bs.return_value = self.mock_soup

        # Create splitters
        self.default_splitter = HTMLSplitter()
        self.custom_splitter = HTMLSplitter(
            {"chunk_size": 100, "chunk_overlap": 20, "tags_to_extract": ["div", "p"]}
        )

        # Sample HTML
        self.sample_html = """
        <!DOCTYPE html>
        <html>
        <head><title>Test</title></head>
        <body>
            <div>Div content</div>
            <p>Paragraph 1</p>
            <p>Paragraph 2</p>
            <section>Section content</section>
        </body>
        </html>
        """

    def test_init_default(self):
        """Test initialization with default parameters."""
        self.assertEqual(self.default_splitter.chunk_size, 1000)
        self.assertEqual(self.default_splitter.chunk_overlap, 200)
        self.assertEqual(
            self.default_splitter.tags_to_extract, ["div", "p", "section", "article"]
        )

    def test_init_custom(self):
        """Test initialization with custom parameters."""
        self.assertEqual(self.custom_splitter.chunk_size, 100)
        self.assertEqual(self.custom_splitter.chunk_overlap, 20)
        self.assertEqual(self.custom_splitter.tags_to_extract, ["div", "p"])

    @patch(
        "src.agents.rag.services.splitter.structured_splitter.BEAUTIFULSOUP_AVAILABLE",
        False,
    )
    def test_init_without_beautifulsoup(self):
        """Test initialization when BeautifulSoup is not available."""
        with self.assertRaises(ImportError):
            HTMLSplitter()

    def test_split_text_empty(self):
        """Test splitting empty text."""
        result = self.default_splitter.split_text("")
        self.assertEqual(result, [])

    def test_split_text_with_tags(self):
        """Test splitting HTML text with specified tags."""
        # Mock the find_all method to return elements
        div_element = MagicMock()
        div_element.get_text.return_value = "Div content"

        p1_element = MagicMock()
        p1_element.get_text.return_value = "Paragraph 1"

        p2_element = MagicMock()
        p2_element.get_text.return_value = "Paragraph 2"

        section_element = MagicMock()
        section_element.get_text.return_value = "Section content"

        # Set up the find_all method to return different elements based on the tag
        def mock_find_all(tag):
            if tag == "div":
                return [div_element]
            elif tag == "p":
                return [p1_element, p2_element]
            elif tag == "section":
                return [section_element]
            else:
                return []

        self.mock_soup.find_all.side_effect = mock_find_all

        result = self.default_splitter.split_text(self.sample_html)
        # We expect one chunk for each element (div, p1, p2, section)
        self.assertEqual(len(result), 4)
        self.assertIn("Div content", result)
        self.assertIn("Paragraph 1", result)
        self.assertIn("Paragraph 2", result)
        self.assertIn("Section content", result)

    def test_split_text_no_tags_found(self):
        """Test splitting HTML text when no tags are found."""
        # Mock the find_all method to return no elements
        self.mock_soup.find_all.return_value = []
        # Mock the get_text method for the fallback case
        self.mock_soup.get_text.return_value = "Fallback text"

        # Mock the text splitter's split_text method
        with patch.object(
            self.default_splitter.text_splitter,
            "split_text",
            return_value=["Fallback chunk"],
        ):
            result = self.default_splitter.split_text(self.sample_html)
            # When no tags are found, it should fall back to the text splitter
            self.assertEqual(result, ["Fallback chunk"])

    def test_split_text_parsing_error(self):
        """Test splitting HTML text when parsing fails."""
        # Mock BeautifulSoup to raise an exception
        with patch(
            "src.agents.rag.services.splitter.structured_splitter.BeautifulSoup",
            side_effect=Exception("Parsing error"),
        ):
            # Mock the text splitter's split_text method
            with patch.object(
                self.default_splitter.text_splitter,
                "split_text",
                return_value=["Fallback chunk"],
            ):
                result = self.default_splitter.split_text(self.sample_html)
                # When parsing fails, it should fall back to the text splitter
                self.assertEqual(result, ["Fallback chunk"])

    def test_can_handle(self):
        """Test the can_handle method."""
        self.assertTrue(self.default_splitter.can_handle("html"))
        self.assertTrue(self.default_splitter.can_handle("htm"))
        self.assertTrue(self.default_splitter.can_handle("HTML"))  # Case insensitive
        self.assertFalse(self.default_splitter.can_handle("text"))
        self.assertFalse(self.default_splitter.can_handle("json"))


class TestMarkdownSplitter(unittest.TestCase):
    """Test the MarkdownSplitter class."""

    def setUp(self):
        """Set up test fixtures."""
        self.default_splitter = MarkdownSplitter()
        self.custom_splitter = MarkdownSplitter(
            {
                "chunk_size": 100,
                "chunk_overlap": 20,
                "headers_to_split_on": ["#", "##", "###"],
                "strip_headers": True,
            }
        )

        # Sample Markdown
        self.sample_markdown = """
        # Heading 1
        
        Content under heading 1.
        
        ## Heading 2
        
        Content under heading 2.
        
        ### Heading 3
        
        Content under heading 3.
        """

        # Sample Markdown with indented headers
        self.indented_markdown = """
        # Heading 1
        
            ## Heading 2
        
        Content under heading 2.
        
          ### Heading 3
        
        Content under heading 3.
        """

    def test_init_default(self):
        """Test initialization with default parameters."""
        self.assertEqual(self.default_splitter.chunk_size, 1000)
        self.assertEqual(self.default_splitter.chunk_overlap, 200)
        self.assertEqual(
            self.default_splitter.headers_to_split_on,
            ["#", "##", "###", "####", "#####", "######"],
        )
        self.assertEqual(self.default_splitter.return_each_line, False)
        self.assertEqual(self.default_splitter.strip_headers, False)

    def test_init_custom(self):
        """Test initialization with custom parameters."""
        self.assertEqual(self.custom_splitter.chunk_size, 100)
        self.assertEqual(self.custom_splitter.chunk_overlap, 20)
        self.assertEqual(self.custom_splitter.headers_to_split_on, ["#", "##", "###"])
        self.assertEqual(self.custom_splitter.strip_headers, True)

    def test_split_text_empty(self):
        """Test splitting empty text."""
        result = self.default_splitter.split_text("")
        self.assertEqual(result, [])

    def test_split_text_with_headers(self):
        """Test splitting Markdown text with headers."""
        result = self.default_splitter.split_text(self.sample_markdown)
        # We expect one chunk for each header (h1, h2, h3)
        self.assertEqual(len(result), 3)

        # Check that each chunk has the correct metadata
        self.assertEqual(result[0]["metadata"]["Header 1"], "Heading 1")
        self.assertEqual(result[1]["metadata"]["Header 1"], "Heading 1")
        self.assertEqual(result[1]["metadata"]["Header 2"], "Heading 2")
        self.assertEqual(result[2]["metadata"]["Header 1"], "Heading 1")
        self.assertEqual(result[2]["metadata"]["Header 2"], "Heading 2")
        self.assertEqual(result[2]["metadata"]["Header 3"], "Heading 3")

    def test_split_text_with_indented_headers(self):
        """Test splitting Markdown text with indented headers."""
        result = self.default_splitter.split_text(self.indented_markdown)
        # We expect one chunk for each header (h1, h2, h3)
        self.assertEqual(len(result), 3)

        # Check that each chunk has the correct metadata
        self.assertEqual(result[0]["metadata"]["Header 1"], "Heading 1")
        self.assertEqual(result[1]["metadata"]["Header 1"], "Heading 1")
        self.assertEqual(result[1]["metadata"]["Header 2"], "Heading 2")
        self.assertEqual(result[2]["metadata"]["Header 1"], "Heading 1")
        self.assertEqual(result[2]["metadata"]["Header 2"], "Heading 2")
        self.assertEqual(result[2]["metadata"]["Header 3"], "Heading 3")

    def test_split_text_with_strip_headers(self):
        """Test splitting Markdown text with strip_headers=True."""
        result = self.custom_splitter.split_text(self.sample_markdown)
        # We expect one chunk for each header (h1, h2, h3)
        self.assertEqual(len(result), 3)

        # Check that the headers are stripped from the content
        for chunk in result:
            self.assertFalse(chunk["content"].startswith("#"))

    def test_split_text_no_headers(self):
        """Test splitting Markdown text with no headers."""
        text = "This is a plain text with no headers."
        # Mock the text splitter's split_text method
        with patch.object(
            self.default_splitter.text_splitter, "split_text", return_value=["Chunk 1"]
        ):
            result = self.default_splitter.split_text(text)
            # When no headers are found, it should fall back to the text splitter
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["content"], "Chunk 1")
            self.assertEqual(result[0]["metadata"], {})

    def test_split_text_large_chunks(self):
        """Test splitting Markdown text with chunks larger than chunk_size."""
        # Create a large markdown text
        large_text = "# Heading\n\n" + ("Content " * 100)

        # Mock the text splitter's split_text method
        with patch.object(
            self.default_splitter.text_splitter,
            "split_text",
            return_value=["Chunk 1", "Chunk 2"],
        ):
            result = self.default_splitter.split_text(large_text)
            # The large content should be split into multiple chunks
            self.assertEqual(len(result), 2)
            # Each chunk should have the same metadata
            self.assertEqual(result[0]["metadata"], result[1]["metadata"])
            self.assertEqual(result[0]["metadata"]["Header 1"], "Heading")


class TestCSVSplitter(unittest.TestCase):
    """Test the CSVSplitter class."""

    def setUp(self):
        """Set up test fixtures."""
        self.default_splitter = CSVSplitter()
        self.custom_splitter = CSVSplitter({"chunk_size": 3, "include_header": True})

        # Sample CSV data
        self.sample_csv = """id,name,age
1,John,30
2,Jane,25
3,Bob,40
4,Alice,35
5,Charlie,28
6,Eva,32
7,Frank,45
8,Grace,22
9,Henry,38
10,Ivy,29"""

    def test_init_default(self):
        """Test initialization with default parameters."""
        self.assertEqual(self.default_splitter.chunk_size, 100)
        self.assertEqual(self.default_splitter.include_header, True)

    def test_init_custom(self):
        """Test initialization with custom parameters."""
        self.assertEqual(self.custom_splitter.chunk_size, 3)
        self.assertEqual(self.custom_splitter.include_header, True)

    def test_split_text_empty(self):
        """Test splitting empty text."""
        result = self.default_splitter.split_text("")
        self.assertEqual(result, [])

    def test_split_text_small_csv(self):
        """Test splitting small CSV text."""
        small_csv = """id,name,age
1,John,30
2,Jane,25"""
        result = self.default_splitter.split_text(small_csv)
        # Since the CSV is small, we expect a single chunk
        self.assertEqual(len(result), 1)
        # The result should contain the CSV content
        self.assertIn("John", result[0])
        self.assertIn("Jane", result[0])

    def test_split_text_large_csv(self):
        """Test splitting large CSV text."""
        result = self.custom_splitter.split_text(self.sample_csv)
        # With chunk_size=3, we expect the CSV to be split into multiple chunks
        # 10 data rows / 3 rows per chunk = 4 chunks (ceiling)
        self.assertEqual(len(result), 4)

        # Check that each chunk contains the header
        for chunk in result:
            self.assertIn("id,name,age", chunk)

    def test_split_text_without_header(self):
        """Test splitting CSV text without including the header."""
        splitter = CSVSplitter({"chunk_size": 3, "include_header": False})
        result = splitter.split_text(self.sample_csv)
        # With chunk_size=3, we expect the CSV to be split into multiple chunks
        self.assertEqual(len(result), 4)

        # Check that only the first chunk contains the header
        self.assertIn("id,name,age", result[0])
        for chunk in result[1:]:
            self.assertNotIn("id,name,age", chunk)

    def test_split_text_invalid_csv(self):
        """Test splitting invalid CSV text."""
        invalid_csv = "This is not a CSV"
        # Mock the text splitter's split_text method
        with patch.object(
            RecursiveCharacterTextSplitter,
            "split_text",
            return_value=["Fallback chunk"],
        ):
            result = self.default_splitter.split_text(invalid_csv)
            # For invalid CSV, it should fall back to the text splitter
            self.assertEqual(result, ["Fallback chunk"])

    def test_can_handle(self):
        """Test the can_handle method."""
        self.assertTrue(self.default_splitter.can_handle("csv"))
        self.assertTrue(self.default_splitter.can_handle("CSV"))  # Case insensitive
        self.assertFalse(self.default_splitter.can_handle("text"))
        self.assertFalse(self.default_splitter.can_handle("json"))


class TestSplitterService(unittest.TestCase):
    """Test the SplitterService class."""

    def setUp(self):
        """Set up test fixtures."""
        self.default_service = SplitterService()
        self.custom_service = SplitterService(
            {
                "splitters": {
                    "text": {
                        "type": "character",
                        "params": {"chunk_size": 100, "chunk_overlap": 20},
                    },
                    "json": {
                        "type": "recursive_json",
                        "params": {"chunk_size": 200, "chunk_overlap": 50},
                    },
                },
                "default_splitter": {
                    "type": "recursive_character",
                    "params": {"chunk_size": 500, "chunk_overlap": 100},
                },
            }
        )

    def test_init_default(self):
        """Test initialization with default parameters."""
        # Check that default splitters are registered
        self.assertIsInstance(
            self.default_service.splitters["text"], RecursiveCharacterTextSplitter
        )
        self.assertIsInstance(
            self.default_service.splitters["json"], RecursiveJSONSplitter
        )
        self.assertIsInstance(self.default_service.splitters["html"], HTMLSplitter)
        self.assertIsInstance(
            self.default_service.splitters["markdown"], MarkdownSplitter
        )
        self.assertIsInstance(self.default_service.splitters["csv"], CSVSplitter)

        # Check that the default splitter is set
        self.assertIsInstance(
            self.default_service.default_splitter, RecursiveCharacterTextSplitter
        )

    def test_init_custom(self):
        """Test initialization with custom parameters."""
        # Check that custom splitters are registered
        self.assertIsInstance(
            self.custom_service.splitters["text"], CharacterTextSplitter
        )
        self.assertIsInstance(
            self.custom_service.splitters["json"], RecursiveJSONSplitter
        )

        # Check that the custom splitter parameters are set
        self.assertEqual(self.custom_service.splitters["text"].chunk_size, 100)
        self.assertEqual(self.custom_service.splitters["text"].chunk_overlap, 20)
        self.assertEqual(self.custom_service.splitters["json"].chunk_size, 200)
        self.assertEqual(self.custom_service.splitters["json"].chunk_overlap, 50)

        # Check that the default splitter is set
        self.assertIsInstance(
            self.custom_service.default_splitter, RecursiveCharacterTextSplitter
        )
        self.assertEqual(self.custom_service.default_splitter.chunk_size, 500)
        self.assertEqual(self.custom_service.default_splitter.chunk_overlap, 100)

    def test_get_splitter(self):
        """Test getting a splitter for a specific data type."""
        # Test getting registered splitters
        self.assertIsInstance(
            self.default_service.get_splitter("text"), RecursiveCharacterTextSplitter
        )
        self.assertIsInstance(
            self.default_service.get_splitter("json"), RecursiveJSONSplitter
        )
        self.assertIsInstance(self.default_service.get_splitter("html"), HTMLSplitter)
        self.assertIsInstance(
            self.default_service.get_splitter("markdown"), MarkdownSplitter
        )
        self.assertIsInstance(self.default_service.get_splitter("csv"), CSVSplitter)

        # Test case insensitivity
        self.assertIsInstance(
            self.default_service.get_splitter("TEXT"), RecursiveCharacterTextSplitter
        )
        self.assertIsInstance(
            self.default_service.get_splitter("JSON"), RecursiveJSONSplitter
        )

        # Test getting splitter for unregistered data type
        self.assertIsInstance(
            self.default_service.get_splitter("unknown"), RecursiveCharacterTextSplitter
        )

    def test_split_text(self):
        """Test splitting text with the appropriate splitter."""
        # Mock the splitters
        text_splitter = MagicMock()
        text_splitter.split_text.return_value = ["Text chunk 1", "Text chunk 2"]

        json_splitter = MagicMock()
        json_splitter.split_text.return_value = ["JSON chunk 1", "JSON chunk 2"]

        # Replace the splitters in the service
        self.default_service.splitters["text"] = text_splitter
        self.default_service.splitters["json"] = json_splitter

        # Test splitting text
        result = self.default_service.split_text("Sample text", "text")
        self.assertEqual(result, ["Text chunk 1", "Text chunk 2"])
        text_splitter.split_text.assert_called_once_with("Sample text")

        # Test splitting JSON
        result = self.default_service.split_text('{"key": "value"}', "json")
        self.assertEqual(result, ["JSON chunk 1", "JSON chunk 2"])
        json_splitter.split_text.assert_called_once_with('{"key": "value"}')

    def test_split_file(self):
        """Test splitting a file."""
        # Mock the open function
        mock_open = mock_open(read_data="Sample text")

        # Mock the split_text method
        with patch("builtins.open", mock_open), patch.object(
            self.default_service, "split_text", return_value=["Chunk 1", "Chunk 2"]
        ):

            # Test with explicit data type
            result = self.default_service.split_file("sample.txt", "text")
            self.assertEqual(result, ["Chunk 1", "Chunk 2"])
            self.default_service.split_text.assert_called_with("Sample text", "text")

            # Test with inferred data type
            result = self.default_service.split_file("sample.json")
            self.assertEqual(result, ["Chunk 1", "Chunk 2"])
            self.default_service.split_text.assert_called_with("Sample text", "json")

            # Test with non-existent file
            with patch("os.path.exists", return_value=False):
                result = self.default_service.split_file("nonexistent.txt")
                self.assertEqual(result, [])

    def test_split_files(self):
        """Test splitting multiple files."""
        # Mock the split_file method
        with patch.object(self.default_service, "split_file") as mock_split_file:
            mock_split_file.side_effect = [
                ["File 1 Chunk 1", "File 1 Chunk 2"],
                ["File 2 Chunk 1", "File 2 Chunk 2"],
            ]

            # Test with explicit data types
            result = self.default_service.split_files(
                ["file1.txt", "file2.json"], ["text", "json"]
            )
            self.assertEqual(
                result,
                [
                    ("file1.txt", ["File 1 Chunk 1", "File 1 Chunk 2"]),
                    ("file2.json", ["File 2 Chunk 1", "File 2 Chunk 2"]),
                ],
            )
            mock_split_file.assert_any_call("file1.txt", "text")
            mock_split_file.assert_any_call("file2.json", "json")

            # Test with inferred data types
            mock_split_file.reset_mock()
            mock_split_file.side_effect = [
                ["File 1 Chunk 1", "File 1 Chunk 2"],
                ["File 2 Chunk 1", "File 2 Chunk 2"],
            ]

            result = self.default_service.split_files(["file1.txt", "file2.json"])
            self.assertEqual(
                result,
                [
                    ("file1.txt", ["File 1 Chunk 1", "File 1 Chunk 2"]),
                    ("file2.json", ["File 2 Chunk 1", "File 2 Chunk 2"]),
                ],
            )
            mock_split_file.assert_any_call("file1.txt", None)
            mock_split_file.assert_any_call("file2.json", None)

    def test_get_supported_data_types(self):
        """Test getting supported data types."""
        # The default service should support text, json, html, markdown, and csv
        supported_types = self.default_service.get_supported_data_types()
        self.assertIn("text", supported_types)
        self.assertIn("json", supported_types)
        self.assertIn("html", supported_types)
        self.assertIn("markdown", supported_types)
        self.assertIn("csv", supported_types)
