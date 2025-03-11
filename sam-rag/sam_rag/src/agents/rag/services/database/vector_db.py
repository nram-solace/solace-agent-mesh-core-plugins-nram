"""
Vector database implementations.
"""

import os
import uuid
import json
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
from .vector_db_base import VectorDBBase


class ChromaDB(VectorDBBase):
    """
    Vector database using ChromaDB.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the ChromaDB vector database.

        Args:
            config: A dictionary containing configuration parameters.
                - persist_directory: The directory to persist the database to (default: "./chroma_db").
                - collection_name: The name of the collection to use (default: "documents").
                - embedding_dimension: The dimension of the embeddings (default: 768).
        """
        super().__init__(config)
        self.persist_directory = self.config.get("persist_directory", "./chroma_db")
        self.collection_name = self.config.get("collection_name", "documents")
        self.embedding_dimension = self.config.get("embedding_dimension", 768)
        self.client = None
        self.collection = None
        self._setup_client()

    def _setup_client(self) -> None:
        """
        Set up the ChromaDB client.
        """
        try:
            import chromadb
            from chromadb.config import Settings

            # Create the client
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )

            # Get or create the collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name, metadata={"hnsw:space": "cosine"}
            )
        except ImportError:
            raise ImportError(
                "The chromadb package is required for ChromaDB. "
                "Please install it with `pip install chromadb`."
            )

    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add documents to the vector database.

        Args:
            documents: The documents to add.
            embeddings: The embeddings of the documents.
            metadatas: Optional metadata for each document.
            ids: Optional IDs for each document.

        Returns:
            The IDs of the added documents.
        """
        if not documents or not embeddings:
            return []

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(documents))]

        # Create default metadata if not provided
        if metadatas is None:
            metadatas = [{"source": "unknown"} for _ in range(len(documents))]

        # Add the documents to the collection
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

        return ids

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query embedding.

        Args:
            query_embedding: The query embedding.
            top_k: The number of results to return.
            filter: Optional filter to apply to the search.

        Returns:
            A list of dictionaries containing the search results.
        """
        # Search the collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter,
        )

        # Format the results
        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append(
                {
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                }
            )

        return formatted_results

    def delete(self, ids: List[str]) -> None:
        """
        Delete documents from the vector database.

        Args:
            ids: The IDs of the documents to delete.
        """
        self.collection.delete(ids=ids)

    def get(self, ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get documents from the vector database.

        Args:
            ids: The IDs of the documents to get.

        Returns:
            A list of dictionaries containing the documents.
        """
        # Get the documents
        results = self.collection.get(ids=ids)

        # Format the results
        formatted_results = []
        for i in range(len(results["ids"])):
            formatted_results.append(
                {
                    "id": results["ids"][i],
                    "document": results["documents"][i],
                    "metadata": results["metadatas"][i],
                    "embedding": results["embeddings"][i],
                }
            )

        return formatted_results

    def update(
        self,
        ids: List[str],
        documents: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Update documents in the vector database.

        Args:
            ids: The IDs of the documents to update.
            documents: Optional new document contents.
            embeddings: Optional new embeddings.
            metadatas: Optional new metadata.
        """
        self.collection.update(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def count(self) -> int:
        """
        Get the number of documents in the vector database.

        Returns:
            The number of documents.
        """
        return self.collection.count()

    def clear(self) -> None:
        """
        Clear all documents from the vector database.
        """
        self.collection.delete(where={})


class FAISS(VectorDBBase):
    """
    Vector database using FAISS.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the FAISS vector database.

        Args:
            config: A dictionary containing configuration parameters.
                - index_path: The path to save the index to (default: "./faiss_index").
                - embedding_dimension: The dimension of the embeddings (default: 768).
                - index_type: The type of index to use (default: "Flat").
        """
        super().__init__(config)
        self.index_path = self.config.get("index_path", "./faiss_index")
        self.embedding_dimension = self.config.get("embedding_dimension", 768)
        self.index_type = self.config.get("index_type", "Flat")
        self.index = None
        self.documents = {}  # Map from ID to document and metadata
        self._setup_index()

    def _setup_index(self) -> None:
        """
        Set up the FAISS index.
        """
        try:
            import faiss

            # Create the index
            if self.index_type == "Flat":
                self.index = faiss.IndexFlatIP(self.embedding_dimension)
            elif self.index_type == "IVF":
                quantizer = faiss.IndexFlatIP(self.embedding_dimension)
                self.index = faiss.IndexIVFFlat(
                    quantizer, self.embedding_dimension, 100
                )
                self.index.train(
                    np.random.random((100, self.embedding_dimension)).astype(np.float32)
                )
            elif self.index_type == "HNSW":
                self.index = faiss.IndexHNSWFlat(self.embedding_dimension, 32)
            else:
                # Default to Flat
                self.index = faiss.IndexFlatIP(self.embedding_dimension)

            # Load the index if it exists
            if os.path.exists(self.index_path + ".index"):
                self.index = faiss.read_index(self.index_path + ".index")

            # Load the documents if they exist
            if os.path.exists(self.index_path + ".json"):
                with open(self.index_path + ".json", "r") as f:
                    self.documents = json.load(f)
        except ImportError:
            raise ImportError(
                "The faiss-cpu package is required for FAISS. "
                "Please install it with `pip install faiss-cpu` or `pip install faiss-gpu`."
            )

    def _save_index(self) -> None:
        """
        Save the index and documents to disk.
        """
        import faiss

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

        # Save the index
        faiss.write_index(self.index, self.index_path + ".index")

        # Save the documents
        with open(self.index_path + ".json", "w") as f:
            json.dump(self.documents, f)

    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add documents to the vector database.

        Args:
            documents: The documents to add.
            embeddings: The embeddings of the documents.
            metadatas: Optional metadata for each document.
            ids: Optional IDs for each document.

        Returns:
            The IDs of the added documents.
        """
        if not documents or not embeddings:
            return []

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(documents))]

        # Create default metadata if not provided
        if metadatas is None:
            metadatas = [{"source": "unknown"} for _ in range(len(documents))]

        # Add the documents to the index
        embeddings_array = np.array(embeddings).astype(np.float32)
        self.index.add(embeddings_array)

        # Store the documents and metadata
        for i, doc_id in enumerate(ids):
            self.documents[doc_id] = {
                "document": documents[i],
                "metadata": metadatas[i],
                "index": self.index.ntotal - len(documents) + i,
            }

        # Save the index
        self._save_index()

        return ids

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query embedding.

        Args:
            query_embedding: The query embedding.
            top_k: The number of results to return.
            filter: Optional filter to apply to the search.

        Returns:
            A list of dictionaries containing the search results.
        """
        # Convert the query embedding to a numpy array
        query_embedding_array = np.array([query_embedding]).astype(np.float32)

        # Search the index
        distances, indices = self.index.search(query_embedding_array, top_k)

        # Format the results
        formatted_results = []
        for i in range(len(indices[0])):
            index = indices[0][i]
            distance = distances[0][i]

            # Find the document ID for this index
            doc_id = None
            for id, doc in self.documents.items():
                if doc["index"] == index:
                    doc_id = id
                    break

            if doc_id is not None:
                # Apply filter if provided
                if filter is not None:
                    metadata = self.documents[doc_id]["metadata"]
                    if not all(metadata.get(k) == v for k, v in filter.items()):
                        continue

                formatted_results.append(
                    {
                        "id": doc_id,
                        "document": self.documents[doc_id]["document"],
                        "metadata": self.documents[doc_id]["metadata"],
                        "distance": float(distance),
                    }
                )

        return formatted_results

    def delete(self, ids: List[str]) -> None:
        """
        Delete documents from the vector database.

        Args:
            ids: The IDs of the documents to delete.
        """
        # FAISS doesn't support deletion directly, so we need to rebuild the index
        # Get all documents except the ones to delete
        remaining_docs = []
        remaining_embeddings = []
        remaining_metadatas = []
        remaining_ids = []

        for doc_id, doc in self.documents.items():
            if doc_id not in ids:
                remaining_ids.append(doc_id)
                remaining_docs.append(doc["document"])
                remaining_metadatas.append(doc["metadata"])

                # Get the embedding for this document
                index = doc["index"]
                embedding = self.index.reconstruct(index)
                remaining_embeddings.append(embedding)

        # Clear the index and documents
        self.index.reset()
        self.documents = {}

        # Add the remaining documents back
        if remaining_docs:
            self.add_documents(
                documents=remaining_docs,
                embeddings=remaining_embeddings,
                metadatas=remaining_metadatas,
                ids=remaining_ids,
            )

    def get(self, ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get documents from the vector database.

        Args:
            ids: The IDs of the documents to get.

        Returns:
            A list of dictionaries containing the documents.
        """
        # Get the documents
        formatted_results = []
        for doc_id in ids:
            if doc_id in self.documents:
                doc = self.documents[doc_id]
                index = doc["index"]
                embedding = self.index.reconstruct(index)

                formatted_results.append(
                    {
                        "id": doc_id,
                        "document": doc["document"],
                        "metadata": doc["metadata"],
                        "embedding": embedding.tolist(),
                    }
                )

        return formatted_results

    def update(
        self,
        ids: List[str],
        documents: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Update documents in the vector database.

        Args:
            ids: The IDs of the documents to update.
            documents: Optional new document contents.
            embeddings: Optional new embeddings.
            metadatas: Optional new metadata.
        """
        # FAISS doesn't support updates directly, so we need to delete and re-add
        # Get the current documents
        current_docs = self.get(ids)

        # Delete the documents
        self.delete(ids)

        # Prepare the updated documents
        updated_docs = []
        updated_embeddings = []
        updated_metadatas = []

        for i, doc_id in enumerate(ids):
            if i < len(current_docs):
                current_doc = current_docs[i]

                # Update the document content if provided
                doc_content = (
                    documents[i]
                    if documents and i < len(documents)
                    else current_doc["document"]
                )

                # Update the embedding if provided
                embedding = (
                    embeddings[i]
                    if embeddings and i < len(embeddings)
                    else current_doc["embedding"]
                )

                # Update the metadata if provided
                metadata = current_doc["metadata"]
                if metadatas and i < len(metadatas):
                    metadata.update(metadatas[i])

                updated_docs.append(doc_content)
                updated_embeddings.append(embedding)
                updated_metadatas.append(metadata)

        # Add the updated documents
        if updated_docs:
            self.add_documents(
                documents=updated_docs,
                embeddings=updated_embeddings,
                metadatas=updated_metadatas,
                ids=ids,
            )

    def count(self) -> int:
        """
        Get the number of documents in the vector database.

        Returns:
            The number of documents.
        """
        return self.index.ntotal

    def clear(self) -> None:
        """
        Clear all documents from the vector database.
        """
        self.index.reset()
        self.documents = {}
        self._save_index()


class Milvus(VectorDBBase):
    """
    Vector database using Milvus.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Milvus vector database.

        Args:
            config: A dictionary containing configuration parameters.
                - host: The host of the Milvus server (default: "localhost").
                - port: The port of the Milvus server (default: 19530).
                - collection_name: The name of the collection to use (default: "documents").
                - embedding_dimension: The dimension of the embeddings (default: 768).
        """
        super().__init__(config)
        self.host = self.config.get("host", "localhost")
        self.port = self.config.get("port", 19530)
        self.collection_name = self.config.get("collection_name", "documents")
        self.embedding_dimension = self.config.get("embedding_dimension", 768)
        self.client = None
        self.collection = None
        self._setup_client()

    def _setup_client(self) -> None:
        """
        Set up the Milvus client.
        """
        try:
            from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

            # Connect to Milvus
            connections.connect(host=self.host, port=self.port)

            # Check if the collection exists
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                self.collection.load()
            else:
                # Define the collection schema
                fields = [
                    FieldSchema(
                        name="id",
                        dtype=DataType.VARCHAR,
                        is_primary=True,
                        max_length=36,
                    ),
                    FieldSchema(
                        name="document", dtype=DataType.VARCHAR, max_length=65535
                    ),
                    FieldSchema(
                        name="embedding",
                        dtype=DataType.FLOAT_VECTOR,
                        dim=self.embedding_dimension,
                    ),
                    # Metadata fields can be added as needed
                    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=255),
                ]
                schema = CollectionSchema(fields=fields)

                # Create the collection
                self.collection = Collection(self.collection_name, schema=schema)

                # Create an index
                index_params = {
                    "metric_type": "IP",
                    "index_type": "HNSW",
                    "params": {"M": 8, "efConstruction": 64},
                }
                self.collection.create_index(
                    field_name="embedding", index_params=index_params
                )
                self.collection.load()
        except ImportError:
            raise ImportError(
                "The pymilvus package is required for Milvus. "
                "Please install it with `pip install pymilvus`."
            )

    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add documents to the vector database.

        Args:
            documents: The documents to add.
            embeddings: The embeddings of the documents.
            metadatas: Optional metadata for each document.
            ids: Optional IDs for each document.

        Returns:
            The IDs of the added documents.
        """
        if not documents or not embeddings:
            return []

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(documents))]

        # Create default metadata if not provided
        if metadatas is None:
            metadatas = [{"source": "unknown"} for _ in range(len(documents))]

        # Prepare the data to insert
        data = [
            ids,
            documents,
            embeddings,
            [m.get("source", "unknown") for m in metadatas],
        ]

        # Insert the data
        self.collection.insert(data)

        return ids

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query embedding.

        Args:
            query_embedding: The query embedding.
            top_k: The number of results to return.
            filter: Optional filter to apply to the search.

        Returns:
            A list of dictionaries containing the search results.
        """
        # Prepare the search parameters
        search_params = {
            "metric_type": "IP",
            "params": {"ef": 64},
        }

        # Prepare the filter expression
        expr = None
        if filter:
            conditions = []
            for key, value in filter.items():
                if isinstance(value, str):
                    conditions.append(f'{key} == "{value}"')
                else:
                    conditions.append(f"{key} == {value}")
            expr = " && ".join(conditions)

        # Search the collection
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["document", "source"],
        )

        # Format the results
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append(
                    {
                        "id": hit.id,
                        "document": hit.entity.get("document"),
                        "metadata": {"source": hit.entity.get("source")},
                        "distance": hit.distance,
                    }
                )

        return formatted_results

    def delete(self, ids: List[str]) -> None:
        """
        Delete documents from the vector database.

        Args:
            ids: The IDs of the documents to delete.
        """
        # Prepare the filter expression
        expr = f'id in ["{ids[0]}"'
        for id in ids[1:]:
            expr += f', "{id}"'
        expr += "]"

        # Delete the documents
        self.collection.delete(expr)

    def get(self, ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get documents from the vector database.

        Args:
            ids: The IDs of the documents to get.

        Returns:
            A list of dictionaries containing the documents.
        """
        # Prepare the filter expression
        expr = f'id in ["{ids[0]}"'
        for id in ids[1:]:
            expr += f', "{id}"'
        expr += "]"

        # Query the collection
        results = self.collection.query(
            expr=expr,
            output_fields=["document", "embedding", "source"],
        )

        # Format the results
        formatted_results = []
        for result in results:
            formatted_results.append(
                {
                    "id": result["id"],
                    "document": result["document"],
                    "metadata": {"source": result["source"]},
                    "embedding": result["embedding"],
                }
            )

        return formatted_results

    def update(
        self,
        ids: List[str],
        documents: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Update documents in the vector database.

        Args:
            ids: The IDs of the documents to update.
            documents: Optional new document contents.
            embeddings: Optional new embeddings.
            metadatas: Optional new metadata.
        """
        # Milvus doesn't support partial updates, so we need to delete and re-add
        # Get the current documents
        current_docs = self.get(ids)

        # Delete the documents
        self.delete(ids)

        # Prepare the updated documents
        updated_docs = []
        updated_embeddings = []
        updated_metadatas = []

        for i, doc_id in enumerate(ids):
            if i < len(current_docs):
                current_doc = current_docs[i]

                # Update the document content if provided
                doc_content = (
                    documents[i]
                    if documents and i < len(documents)
                    else current_doc["document"]
                )

                # Update the embedding if provided
                embedding = (
                    embeddings[i]
                    if embeddings and i < len(embeddings)
                    else current_doc["embedding"]
                )

                # Update the metadata if provided
                metadata = current_doc["metadata"]
                if metadatas and i < len(metadatas):
                    metadata.update(metadatas[i])

                updated_docs.append(doc_content)
                updated_embeddings.append(embedding)
                updated_metadatas.append(metadata)

        # Add the updated documents
        if updated_docs:
            self.add_documents(
                documents=updated_docs,
                embeddings=updated_embeddings,
                metadatas=updated_metadatas,
                ids=ids,
            )

    def count(self) -> int:
        """
        Get the number of documents in the vector database.

        Returns:
            The number of documents.
        """
        return self.collection.num_entities

    def clear(self) -> None:
        """
        Clear all documents from the vector database.
        """
        self.collection.drop()
        self._setup_client()
