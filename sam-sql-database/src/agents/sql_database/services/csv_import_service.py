"""Service for importing CSV files into database tables."""

import os
import csv
from typing import List, Dict, Any, Optional
import sqlalchemy as sa
from sqlalchemy.engine import Engine
from sqlalchemy import text, inspect
import re

from solace_ai_connector.common.log import log


class CsvImportService:
    """Service for importing CSV files into database tables."""

    def __init__(self, engine: Engine):
        """Initialize the CSV import service.
        
        Args:
            engine: SQLAlchemy engine instance
        """
        self.engine = engine

    def import_csv_files(self, files: Optional[List[str]] = None, 
                        directories: Optional[List[str]] = None) -> None:
        """Import CSV files into database tables.
        
        Args:
            files: List of CSV file paths
            directories: List of directory paths containing CSV files
        """
        if not files:
            files = []
        elif isinstance(files, str):
            files = [file.strip() for file in files.split(',')]
        if not directories:
            directories = []
        elif isinstance(directories, str):
            directories = [directory.strip() for directory in directories.split(',')]

        # Collect all CSV files
        csv_files = []
        csv_files.extend(files or [])
        
        # Add files from directories
        for directory in directories or []:
            if os.path.isdir(directory):
                log.info("sql-db: Scanning directory for CSV files: %s", directory)
                for filename in os.listdir(directory):
                    if filename.lower().endswith('.csv'):
                        csv_files.append(os.path.join(directory, filename))
                        log.debug("sql-db: Found CSV file: %s", filename)

        log.info("sql-db: Total CSV files to process: %d", len(csv_files))

        # Process each CSV file
        for i, csv_file in enumerate(csv_files, 1):
            log.info("sql-db: Processing CSV file %d/%d: %s", i, len(csv_files), csv_file)
            try:
                self._import_csv_file(csv_file)
                log.info("sql-db: Successfully imported CSV file: %s", csv_file)
            except Exception as e:
                log.error("sql-db: Error importing CSV file %s: %s", csv_file, str(e))


    @staticmethod
    def convert_headers_to_snake_case(headers: List[str]) -> List[str]:
        """Convert a list of headers to snake_case.

        Args:
            headers: List of header strings

        Returns:
            List of converted header strings
        """
        converted_headers = []

        for header in headers:
            header = header.strip().replace(' ', '_')  # replace spaces with underscores
            new_header = ""
            if "_" in header:  # assume it is already in snake_case
                new_header += header.lower()
            else:  # do reformat to snake_case
                for c in header:
                    if c.isupper():
                        new_header += "_" + c.lower()
                    else:
                        new_header += c
                new_header = new_header.lstrip('_')
            converted_headers.append(new_header)

        return converted_headers

    def _import_csv_file(self, file_path: str) -> None:
        """Import a single CSV file into a database table.
        
        Args:
            file_path: Path to the CSV file
        """
        log.info("sql-db: Starting import of CSV file: %s", file_path)
        
        # Extract table name from filename
        table_name = os.path.splitext(os.path.basename(file_path))[0].lower()
        table_name = re.sub(r'[^a-zA-Z0-9_]', '_', table_name)
        
        log.info("sql-db: Using table name: %s", table_name)

        # Check if table already exists
        if inspect(self.engine).has_table(table_name):
            log.info("sql-db: Table %s already exists, skipping import", table_name)
            return

        # Read CSV file and create table
        try:
            log.info("sql-db: Reading CSV file to determine structure...")
            with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                headers = next(reader)
                
                log.info("sql-db: CSV file has %d columns: %s", len(headers), headers)
                
                # Create table
                self._create_table_from_headers(table_name, headers)
                log.info("sql-db: Created table: %s", table_name)
                
                # Insert data
                row_count = 0
                chunk_size = 1000
                chunk = []
                
                for row in reader:
                    if len(row) != len(headers):
                        log.warning("sql-db: Skipping row with incorrect number of columns in %s", 
                                   file_path)
                        continue
                    
                    chunk.append(dict(zip(headers, row)))
                    row_count += 1
                    
                    if len(chunk) >= chunk_size:
                        self._insert_chunk(table_name, chunk)
                        log.debug("sql-db: Inserted chunk of %d rows (total: %d)", len(chunk), row_count)
                        chunk = []
                
                # Insert remaining rows
                if chunk:
                    self._insert_chunk(table_name, chunk)
                    log.debug("sql-db: Inserted final chunk of %d rows", len(chunk))
                
                log.info("sql-db: Successfully imported %d rows into table %s", row_count, table_name)

        except Exception as e:
            log.error("sql-db: Error processing CSV file %s: %s", file_path, str(e))
            raise

    def _create_table_from_headers(self, table_name: str, headers: List[str]) -> None:
        """Creates a table based on the headers of a CSV file.
        
        Args:
            table_name: Name of the table to create
            headers: List of column names from the CSV file
        """
        try:
            # Convert headers to snake_case
            headers = self.convert_headers_to_snake_case(headers)

            # Create table
            columns = []
            has_id = False
            
            # Check if there's an id column
            for header in headers:
                if header.lower() == 'id':
                    has_id = True
                    break

            # Add id column if none exists
            if not has_id:
                columns.append(sa.Column('id', sa.Integer, primary_key=True))

            # Add columns for CSV fields
            for header in headers:
                # If this is an id column, make it text type
                if header.lower() == 'id':
                    columns.append(sa.Column(header, sa.Text))
                else:
                    columns.append(sa.Column(header, sa.Text))

            # Create table
            metadata = sa.MetaData()
            table = sa.Table(table_name, metadata, *columns)
            metadata.create_all(self.engine)
        except Exception as e:
            log.error("sql-db: Error creating table %s from headers: %s", table_name, str(e))
            raise

    def _insert_chunk(self, table_name: str, chunk: List[Dict[str, Any]]) -> None:
        """Insert a chunk of rows into a table.
        
        Args:
            table_name: Name of the target table
            chunk: List of row dictionaries to insert
        """
        if not chunk:
            return

        try:
            with self.engine.begin() as conn:
                # Build insert statement
                insert_stmt = text(
                    f"INSERT INTO {table_name} ({', '.join(chunk[0].keys())}) "
                    f"VALUES ({', '.join([':' + k for k in chunk[0].keys()])})"
                )
                
                # Execute with chunk of rows
                conn.execute(insert_stmt, chunk)
        except Exception as e:
            log.error("sql-db: Error inserting chunk into %s: %s", table_name, str(e))
            raise
