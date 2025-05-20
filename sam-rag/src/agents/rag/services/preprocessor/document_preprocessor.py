"""
Document preprocessors for various file formats.
"""

import os
from typing import Dict, Any, List
from .preprocessor_base import PreprocessorBase, PreprocessedOutput
from .text_preprocessor import TextPreprocessor
import csv
from solace_ai_connector.common.log import log as logger


def filter_config(config: Dict[str, Any], key: str) -> Dict[str, Any]:
    """
    Filter the configuration dictionary to get the specific settings for a given key.

    Args:
        config: The original configuration dictionary.
        key: The key to filter by (e.g., 'text', 'pdf', 'html').

    Returns:
        A filtered configuration dictionary with the parameters for the specified file type.
    """
    if not config:
        return {}

    # First try to get file-specific preprocessor config
    preprocessors = config.get("preprocessors", {})
    file_config = preprocessors.get(key, {})

    # Extract the params section if it exists
    if "params" in file_config:
        return file_config.get("params", {})

    # If no specific config found, try to use default preprocessor
    default_preprocessor = config.get("default_preprocessor", {})
    return default_preprocessor.get("params", {})


class TextFilePreprocessor(PreprocessorBase):
    """
    Preprocessor for plain text files.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the text file preprocessor.

        Args:
            config: Configuration dictionary.
        """
        super().__init__(config)
        self.extensions = [".txt", ".md", ".csv", ".json", ".yaml", ".yml", ".xml"]

    def can_process(self, file_path: str) -> bool:
        """
        Check if this preprocessor can handle the given file.

        Args:
            file_path: Path to the file.

        Returns:
            True if this preprocessor can handle the file, False otherwise.
        """
        ext = os.path.splitext(file_path.lower())[1]
        return ext in self.extensions

    def preprocess(self, file_path: str) -> PreprocessedOutput:
        """
        Preprocess a text file and extract metadata.

        Args:
            file_path: Path to the text file.

        Returns:
            A dictionary containing preprocessed text content and metadata.
        """
        metadata: Dict[str, Any] = {
            "file_path": file_path,
            "custom_tags": [],  # Initialize custom_tags
        }
        text_content = ""
        file_extension = os.path.splitext(file_path.lower())[1]
        metadata["file_type"] = file_extension.lstrip(".")

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                if metadata["file_type"] == "csv":
                    # For CSV, try to extract headers and then read the full content as text
                    try:
                        reader = csv.reader(file)
                        headers = next(reader, None)
                        if headers:
                            metadata["csv_headers"] = headers
                        file.seek(0)  # Reset file pointer to read the whole content
                        text_content = file.read()
                    except csv.Error as e:
                        logger.warning(
                            f"Could not parse CSV headers for {file_path}: {e}"
                        )
                        file.seek(
                            0
                        )  # Ensure reading from start if header extraction failed
                        text_content = file.read()  # Fallback to reading as plain text
                    except Exception as e:
                        logger.warning(
                            f"Unexpected error during CSV processing for {file_path}: {e}"
                        )
                        file.seek(0)
                        text_content = file.read()
                else:
                    text_content = file.read()

            text_config = filter_config(
                self.config, "text"
            )  # Assuming "text" is the generic config key
            # If specific config for md, json, etc. is needed, filter_config might need adjustment
            # or use file_type as key: filter_config(self.config, metadata["file_type"])
            self.text_preprocessor = TextPreprocessor(text_config)
            processed_text = self.text_preprocessor.preprocess(text_content)

            return {"text_content": processed_text, "metadata": metadata}
        except Exception as e:
            logger.error(f"Error preprocessing text file {file_path}: {e}")
            # Return minimal metadata in case of error
            return {"text_content": "", "metadata": metadata}


class PDFPreprocessor(PreprocessorBase):
    """
    Preprocessor for PDF files.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the PDF preprocessor.

        Args:
            config: Configuration dictionary.
        """
        super().__init__(config)
        # We'll import PyPDF2 only when needed to avoid unnecessary dependencies
        self.pdf_reader = None

    def can_process(self, file_path: str) -> bool:
        """
        Check if this preprocessor can handle the given file.

        Args:
            file_path: Path to the file.

        Returns:
            True if this preprocessor can handle the file, False otherwise.
        """
        return file_path.lower().endswith(".pdf")

    def preprocess(self, file_path: str) -> PreprocessedOutput:
        """
        Preprocess a PDF file and extract metadata.

        Args:
            file_path: Path to the PDF file.

        Returns:
            A dictionary containing preprocessed text content and metadata.
        """
        metadata: Dict[str, Any] = {
            "file_path": file_path,
            "file_type": "pdf",
            "custom_tags": [],
            "keywords": [],  # PyPDF2 doesn't easily expose /Keywords
        }
        text_content = ""

        try:
            # Import PyPDF2 only when needed
            import PyPDF2
            from datetime import datetime

            pdf_config = filter_config(self.config, "pdf")
            self.text_preprocessor = TextPreprocessor(pdf_config)

            with open(file_path, "rb") as file:
                print(f"Processing PDF file: {file_path}")
                pdf_reader = PyPDF2.PdfReader(file)
                print(f"PDF reader initialized for: {file_path}")

                # Extract metadata from PDF properties (dictionary-style)
                try:
                    doc_info = pdf_reader.metadata
                    if doc_info:
                        title = doc_info.get("/Title")
                        if title:
                            metadata["title"] = str(title)
                        author = doc_info.get("/Author")
                        if author:
                            metadata["author"] = str(author)
                        # Creation date handling
                        creation_date = doc_info.get("/CreationDate")
                        if (
                            creation_date
                            and isinstance(creation_date, str)
                            and creation_date.startswith("D:")
                        ):
                            import re

                            try:
                                # Extract up to 14 digits after 'D:' (YYYYMMDDHHMMSS)
                                match = re.match(r"D:(\d{8,14})", creation_date)
                                if match:
                                    date_digits = match.group(1)
                                    if len(date_digits) >= 14:
                                        parsed_date = datetime.strptime(
                                            date_digits[:14], "%Y%m%d%H%M%S"
                                        )
                                    else:
                                        parsed_date = datetime.strptime(
                                            date_digits[:8], "%Y%m%d"
                                        )
                                    metadata["creation_date"] = parsed_date.strftime(
                                        "%Y-%m-%d"
                                    )
                                else:
                                    raise ValueError(
                                        "No valid date digits found in creation date string"
                                    )
                            except Exception as e:
                                logger.warning(
                                    f"Could not parse creation date string '{creation_date}' for {file_path}: {e}"
                                )
                                metadata["creation_date"] = ""
                        else:
                            metadata["creation_date"] = ""
                except Exception as meta_ex:
                    logger.warning(
                        f"Failed to extract PDF metadata for {file_path}: {meta_ex}"
                    )
                    metadata["title"] = metadata.get("title", "")
                    metadata["author"] = metadata.get("author", "")
                    metadata["creation_date"] = metadata.get("creation_date", "")

                metadata["page_count"] = len(pdf_reader.pages)

                print("Extracting text from PDF pages")
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    extracted_page_text = page.extract_text()
                    if extracted_page_text:
                        text_content += extracted_page_text + "\n"

            processed_text = self.text_preprocessor.preprocess(text_content)
            return {"text_content": processed_text, "metadata": metadata}

        except ImportError:
            logger.error(
                "PyPDF2 is not installed. Please install it using: pip install PyPDF2"
            )
            return {"text_content": "", "metadata": metadata}  # Return minimal metadata
        except Exception as e:
            logger.error(f"Error preprocessing PDF file {file_path}: {e}")
            return {"text_content": "", "metadata": metadata}  # Return minimal metadata


class DocxPreprocessor(PreprocessorBase):
    """
    Preprocessor for Microsoft Word (DOCX) files.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the DOCX preprocessor.

        Args:
            config: Configuration dictionary.
        """
        super().__init__(config)

    def can_process(self, file_path: str) -> bool:
        """
        Check if this preprocessor can handle the given file.

        Args:
            file_path: Path to the file.

        Returns:
            True if this preprocessor can handle the file, False otherwise.
        """
        return file_path.lower().endswith((".doc", ".docx"))

    def preprocess(self, file_path: str) -> PreprocessedOutput:
        """
        Preprocess a DOCX file and extract metadata.

        Args:
            file_path: Path to the DOCX file.

        Returns:
            A dictionary containing preprocessed text content and metadata.
        """
        file_extension = os.path.splitext(file_path.lower())[1].lstrip(".")
        metadata: Dict[str, Any] = {
            "file_path": file_path,
            "file_type": file_extension,
            "custom_tags": [],
            "keywords": [],
        }
        text_content = ""

        try:
            # Import docx only when needed
            import docx
            from datetime import datetime  # Ensure datetime is imported if not already at top level

            doc_config = filter_config(
                self.config, "doc"
            )  # Assuming "doc" covers .docx as well
            self.text_preprocessor = TextPreprocessor(doc_config)

            doc = docx.Document(file_path)

            # Extract core properties
            core_props = doc.core_properties
            if core_props:
                if core_props.title:
                    metadata["title"] = core_props.title
                if core_props.author:
                    metadata["author"] = core_props.author
                if core_props.keywords:  # This is often a string, not a list
                    kw_str = core_props.keywords
                    if kw_str:
                        metadata["keywords"] = [
                            k.strip() for k in kw_str.split(",") if k.strip()
                        ]
                if core_props.created:  # This is a datetime object
                    try:
                        metadata["creation_date"] = core_props.created.strftime(
                            "%Y-%m-%d"
                        )
                    except (
                        AttributeError
                    ):  # If 'created' is not a datetime object for some reason
                        logger.warning(
                            f"Could not format 'created' date for {file_path}: {core_props.created}"
                        )
                        metadata["creation_date"] = str(core_props.created)
                # python-docx does not directly provide a total page count easily.
                # metadata["page_count"] = ... # Omitted

            text_content = "\n".join([paragraph.text for paragraph in doc.paragraphs])

            processed_text = self.text_preprocessor.preprocess(text_content)
            return {"text_content": processed_text, "metadata": metadata}

        except ImportError:
            logger.error(
                "python-docx is not installed. Please install it using: pip install python-docx"
            )
            return {"text_content": "", "metadata": metadata}
        except Exception as e:
            logger.error(f"Error preprocessing DOCX file {file_path}: {e}")
            return {"text_content": "", "metadata": metadata}


class HTMLPreprocessor(PreprocessorBase):
    """
    Preprocessor for HTML files.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the HTML preprocessor.

        Args:
            config: Configuration dictionary.
        """
        super().__init__(config)

    def can_process(self, file_path: str) -> bool:
        """
        Check if this preprocessor can handle the given file.

        Args:
            file_path: Path to the file.

        Returns:
            True if this preprocessor can handle the file, False otherwise.
        """
        return file_path.lower().endswith((".html", ".htm"))

    def preprocess(self, file_path: str) -> PreprocessedOutput:
        """
        Preprocess an HTML file and extract metadata.

        Args:
            file_path: Path to the HTML file.

        Returns:
            A dictionary containing preprocessed text content and metadata.
        """
        file_extension = os.path.splitext(file_path.lower())[1].lstrip(".")
        metadata: Dict[str, Any] = {
            "file_path": file_path,
            "file_type": file_extension,
            "custom_tags": [],
            "keywords": [],
        }
        text_content = ""

        try:
            # Import BeautifulSoup only when needed
            from bs4 import BeautifulSoup

            html_config = filter_config(self.config, "html")
            self.text_preprocessor = TextPreprocessor(html_config)

            with open(file_path, "r", encoding="utf-8") as file:
                soup = BeautifulSoup(file.read(), "html.parser")

                # Extract html_title_tag
                if soup.title and soup.title.string:
                    metadata["html_title_tag"] = soup.title.string.strip()

                # Attempt to extract other metadata from meta tags
                meta_author = soup.find(
                    "meta", attrs={"name": lambda x: x and x.lower() == "author"}
                )
                if meta_author and meta_author.get("content"):
                    metadata["author"] = meta_author["content"].strip()

                meta_keywords = soup.find(
                    "meta", attrs={"name": lambda x: x and x.lower() == "keywords"}
                )
                if meta_keywords and meta_keywords.get("content"):
                    kw_str = meta_keywords["content"].strip()
                    if kw_str:
                        metadata["keywords"] = [
                            k.strip() for k in kw_str.split(",") if k.strip()
                        ]

                # Common meta tags for date: 'date', 'dcterms.created', 'article.published_time'
                date_meta_names = [
                    "date",
                    "creation_date",
                    "dcterms.created",
                    "article:published_time",
                    "og:article:published_time",
                ]
                for name in date_meta_names:
                    meta_date = soup.find(
                        "meta", attrs={"name": lambda x: x and x.lower() == name}
                    )
                    if not meta_date:  # Try property attribute for OpenGraph tags
                        meta_date = soup.find(
                            "meta",
                            attrs={"property": lambda x: x and x.lower() == name},
                        )
                    if meta_date and meta_date.get("content"):
                        # Basic parsing, assuming YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS
                        date_str = meta_date["content"].strip()
                        if date_str:
                            metadata["creation_date"] = date_str.split("T")[
                                0
                            ]  # Take only date part
                            break  # Found a date

                # Extract text from HTML
                text_content = soup.get_text(separator=" ", strip=True)

            processed_text = self.text_preprocessor.preprocess(text_content)
            return {"text_content": processed_text, "metadata": metadata}

        except ImportError:
            logger.error(
                "BeautifulSoup is not installed. Please install it using: pip install beautifulsoup4"
            )
            return {"text_content": "", "metadata": metadata}
        except Exception as e:
            logger.error(f"Error preprocessing HTML file {file_path}: {e}")
            return {"text_content": "", "metadata": metadata}


class ExcelPreprocessor(PreprocessorBase):
    """
    Preprocessor for Excel files.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Excel preprocessor.

        Args:
            config: Configuration dictionary.
        """
        super().__init__(config)

    def can_process(self, file_path: str) -> bool:
        """
        Check if this preprocessor can handle the given file.

        Args:
            file_path: Path to the file.

        Returns:
            True if this preprocessor can handle the file, False otherwise.
        """
        return file_path.lower().endswith((".xls", ".xlsx"))

    def preprocess(self, file_path: str) -> PreprocessedOutput:
        """
        Preprocess an Excel file and extract metadata.

        Args:
            file_path: Path to the Excel file.

        Returns:
            A dictionary containing preprocessed text content and metadata.
        """
        file_extension = os.path.splitext(file_path.lower())[1].lstrip(".")
        metadata: Dict[str, Any] = {
            "file_path": file_path,
            "file_type": file_extension,
            "custom_tags": [],
            # title, author, keywords, creation_date are not typically standard in Excel files
            # in a way that pandas can easily extract without custom logic or conventions.
        }
        text_content = ""

        try:
            # Import pandas only when needed
            import pandas as pd

            xls_config = filter_config(self.config, "xls")  # or "excel"
            self.text_preprocessor = TextPreprocessor(xls_config)

            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            if sheet_names:  # Add sheet names to metadata if they exist
                metadata["sheet_names"] = sheet_names

            for sheet_name in sheet_names:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                # Convert dataframe to string
                text_content += f"Sheet: {sheet_name}\n"
                # Optionally, add first row as headers for each sheet if relevant
                # if not df.empty:
                #     metadata[f"sheet_{sheet_name}_headers"] = list(df.columns)
                text_content += df.to_string(index=False) + "\n\n"

            processed_text = self.text_preprocessor.preprocess(text_content)
            return {"text_content": processed_text, "metadata": metadata}

        except ImportError:
            logger.error(
                "pandas is not installed. Please install it using: pip install pandas"
            )
            return {"text_content": "", "metadata": metadata}
        except Exception as e:
            logger.error(f"Error preprocessing Excel file {file_path}: {e}")
            return {"text_content": "", "metadata": metadata}


class ODTPreprocessor(PreprocessorBase):
    """
    Preprocessor for OpenDocument Text (ODT) files.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the ODT preprocessor.

        Args:
            config: Configuration dictionary.
        """
        super().__init__(config)

    def can_process(self, file_path: str) -> bool:
        """
        Check if this preprocessor can handle the given file.

        Args:
            file_path: Path to the file.

        Returns:
            True if this preprocessor can handle the file, False otherwise.
        """
        return file_path.lower().endswith(".odt")

    def preprocess(self, file_path: str) -> PreprocessedOutput:
        """
        Preprocess an ODT file and extract metadata.

        Args:
            file_path: Path to the ODT file.

        Returns:
            A dictionary containing preprocessed text content and metadata.
        """
        metadata: Dict[str, Any] = {
            "file_path": file_path,
            "file_type": "odt",
            "custom_tags": [],
            "keywords": [],
        }
        text_content = ""

        try:
            # Import odfpy only when needed
            from odf import text, teletype
            from odf.opendocument import load
            from odf.meta import Meta  # For metadata extraction
            from datetime import datetime  # For date parsing

            odt_config = filter_config(self.config, "odt")
            self.text_preprocessor = TextPreprocessor(odt_config)

            textdoc = load(file_path)

            # Extract metadata
            meta = textdoc.getElementsByType(Meta)
            if meta:
                meta_obj = meta[0]  # Assuming one meta element
                if meta_obj.getTitle():
                    metadata["title"] = meta_obj.getTitle()
                if meta_obj.getCreator():  # Author
                    metadata["author"] = meta_obj.getCreator()
                if meta_obj.getCreationDate():  # Returns datetime object
                    try:
                        metadata["creation_date"] = meta_obj.getCreationDate().strftime(
                            "%Y-%m-%d"
                        )
                    except AttributeError:
                        logger.warning(
                            f"Could not format creation date for ODT {file_path}: {meta_obj.getCreationDate()}"
                        )
                        metadata["creation_date"] = str(meta_obj.getCreationDate())

                # Keywords in ODF are stored as multiple <meta:keyword> elements
                keywords_elements = meta_obj.getElementsByType(text.Keyword)
                if keywords_elements:
                    keywords_list = [
                        teletype.extractText(kw)
                        for kw in keywords_elements
                        if teletype.extractText(kw)
                    ]
                    if keywords_list:
                        metadata["keywords"] = keywords_list

            allparas = textdoc.getElementsByType(text.P)
            text_content = "\n".join([teletype.extractText(para) for para in allparas])

            processed_text = self.text_preprocessor.preprocess(text_content)
            return {"text_content": processed_text, "metadata": metadata}

        except ImportError:
            logger.error(
                "odfpy is not installed. Please install it using: pip install odfpy"
            )
            return {"text_content": "", "metadata": metadata}
        except Exception as e:
            logger.error(f"Error preprocessing ODT file {file_path}: {e}")
            return {"text_content": "", "metadata": metadata}


class CSVFilePreprocessor(PreprocessorBase):
    """
    Preprocessor for plain text files.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the text file preprocessor.

        Args:
            config: Configuration dictionary.
        """
        super().__init__(config)
        self.extensions = [".csv"]

    def can_process(self, file_path: str) -> bool:
        """
        Check if this preprocessor can handle the given file.

        Args:
            file_path: Path to the file.

        Returns:
            True if this preprocessor can handle the file, False otherwise.
        """
        ext = os.path.splitext(file_path.lower())[1]
        return ext in self.extensions

    def preprocess(self, file_path: str) -> PreprocessedOutput:
        """
        Preprocess a CSV file and extract metadata.

        Args:
            file_path: Path to the CSV file.

        Returns:
            A dictionary containing preprocessed text content and metadata.
        """
        metadata: Dict[str, Any] = {
            "file_path": file_path,
            "file_type": "csv",
            "custom_tags": [],
        }
        text_content = ""

        try:
            # Ensure csv module is imported if not already at the top
            # import csv

            with open(file_path, "r", encoding="utf-8", newline="") as file:
                # Extract headers
                try:
                    reader = csv.reader(file)
                    headers = next(reader, None)
                    if headers:
                        metadata["csv_headers"] = headers
                    file.seek(0)  # Reset file pointer to read the whole content
                    text_content = file.read()
                except csv.Error as e:
                    logger.warning(f"Could not parse CSV headers for {file_path}: {e}")
                    file.seek(0)
                    text_content = file.read()  # Fallback to reading as plain text
                except (
                    Exception
                ) as e:  # Catch other potential errors during header reading
                    logger.warning(
                        f"Unexpected error reading CSV headers for {file_path}: {e}"
                    )
                    file.seek(0)
                    text_content = file.read()

            # CSV content is typically not further preprocessed by TextPreprocessor in the same way
            # unless specific cleaning (like removing extra spaces in data) is desired.
            # For now, we return the raw text content.
            # If TextPreprocessor is needed:
            # csv_config = filter_config(self.config, "csv") # or "text"
            # self.text_preprocessor = TextPreprocessor(csv_config)
            # processed_text = self.text_preprocessor.preprocess(text_content)
            # return {"text_content": processed_text, "metadata": metadata}

            return {"text_content": text_content, "metadata": metadata}

        except Exception as e:
            logger.error(f"Error preprocessing CSV file {file_path}: {e}")
            return {"text_content": "", "metadata": metadata}
