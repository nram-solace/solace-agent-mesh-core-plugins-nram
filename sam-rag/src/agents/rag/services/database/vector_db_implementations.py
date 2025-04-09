"""
Additional vector database implementations.

This module contains implementations for various vector databases:
- Pinecone
- Weaviate
- Qdrant
- Redis
- Elasticsearch
- pgvector (PostgreSQL)
- Vespa
- Vald
- Zilliz
"""

import os
import uuid
import json
from typing import Dict, Any, List, Optional
import numpy as np

from .vector_db_base import VectorDBBase


class PineconeDB(VectorDBBase):
    """
    Vector database using Pinecone.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Pinecone vector database.

        Args:
            config: A dictionary containing configuration parameters.
                - api_key: The Pinecone API key (required).
                - environment: The Pinecone environment (required).
                - index_name: The name of the index to use (required).
                - namespace: The namespace to use (default: "default").
                - embedding_dimension: The dimension of the embeddings (default: 768).
        """
        super().__init__(config)
        self.api_key = self.config.get("api_key")
        if not self.api_key:
            raise ValueError("Pinecone API key is required")

        self.environment = self.config.get("environment")
        if not self.environment:
            raise ValueError("Pinecone environment is required")

        self.index_name = self.config.get("index_name")
        if not self.index_name:
            raise ValueError("Pinecone index name is required")

        self.namespace = self.config.get("namespace", "default")
        self.embedding_dimension = self.config.get("embedding_dimension", 768)
        self.index = None
        self._setup_client()

    def _setup_client(self) -> None:
        """
        Set up the Pinecone client.
        """
        try:
            import pinecone

            # Initialize Pinecone
            pinecone.init(api_key=self.api_key, environment=self.environment)

            # Check if the index exists
            if self.index_name not in pinecone.list_indexes():
                # Create the index
                pinecone.create_index(
                    name=self.index_name,
                    dimension=self.embedding_dimension,
                    metric="cosine",
                )

            # Connect to the index
            self.index = pinecone.Index(self.index_name)
        except ImportError:
            raise ImportError(
                "The pinecone-client package is required for PineconeDB. "
                "Please install it with `pip install pinecone-client`."
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

        # Add document text to metadata
        for i, doc in enumerate(documents):
            metadatas[i]["text"] = doc

        # Prepare the vectors to upsert
        vectors = []
        for i in range(len(documents)):
            vectors.append((ids[i], embeddings[i], metadatas[i]))

        # Upsert the vectors in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            self.index.upsert(vectors=batch, namespace=self.namespace)

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
        # Search the index
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=self.namespace,
            filter=filter,
        )

        # Format the results
        formatted_results = []
        for match in results.matches:
            formatted_results.append(
                {
                    "id": match.id,
                    "text": match.metadata.get("text", ""),
                    "metadata": {
                        k: v for k, v in match.metadata.items() if k != "text"
                    },
                    "distance": match.score,
                }
            )

        return formatted_results

    def delete(self, ids: List[str]) -> None:
        """
        Delete documents from the vector database.

        Args:
            ids: The IDs of the documents to delete.
        """
        self.index.delete(ids=ids, namespace=self.namespace)

    def get(self, ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get documents from the vector database.

        Args:
            ids: The IDs of the documents to get.

        Returns:
            A list of dictionaries containing the documents.
        """
        # Fetch the vectors
        results = self.index.fetch(ids=ids, namespace=self.namespace)

        # Format the results
        formatted_results = []
        for id, vector in results.vectors.items():
            formatted_results.append(
                {
                    "id": id,
                    "text": vector.metadata.get("text", ""),
                    "metadata": {
                        k: v for k, v in vector.metadata.items() if k != "text"
                    },
                    "embedding": vector.values,
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
        # Get the current documents
        current_docs = self.get(ids)

        # Prepare the vectors to upsert
        vectors = []
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

                # Add document text to metadata
                metadata["text"] = doc_content

                vectors.append((doc_id, embedding, metadata))

        # Upsert the vectors in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            self.index.upsert(vectors=batch, namespace=self.namespace)

    def count(self) -> int:
        """
        Get the number of documents in the vector database.

        Returns:
            The number of documents.
        """
        stats = self.index.describe_index_stats()
        return stats.namespaces.get(self.namespace, {}).get("vector_count", 0)

    def clear(self) -> None:
        """
        Clear all documents from the vector database.
        """
        self.index.delete(delete_all=True, namespace=self.namespace)


class WeaviateDB(VectorDBBase):
    """
    Vector database using Weaviate.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Weaviate vector database.

        Args:
            config: A dictionary containing configuration parameters.
                - url: The Weaviate URL (default: "http://localhost:8080").
                - api_key: The Weaviate API key (optional).
                - class_name: The name of the class to use (default: "Document").
                - batch_size: The batch size to use (default: 100).
        """
        super().__init__(config)
        self.url = self.config.get("url", "http://localhost:8080")
        self.api_key = self.config.get("api_key")
        self.class_name = self.config.get("class_name", "Document")
        self.batch_size = self.config.get("batch_size", 100)
        self.client = None
        self._setup_client()

    def _setup_client(self) -> None:
        """
        Set up the Weaviate client.
        """
        try:
            import weaviate
            from weaviate.auth import AuthApiKey

            # Create the client
            auth = AuthApiKey(self.api_key) if self.api_key else None
            self.client = weaviate.Client(url=self.url, auth_client_secret=auth)

            # Check if the class exists
            if not self.client.schema.exists(self.class_name):
                # Create the class
                class_obj = {
                    "class": self.class_name,
                    "vectorizer": "none",  # We provide our own vectors
                    "properties": [
                        {
                            "name": "text",
                            "dataType": ["text"],
                        },
                        {
                            "name": "source",
                            "dataType": ["string"],
                        },
                    ],
                }
                self.client.schema.create_class(class_obj)
        except ImportError:
            raise ImportError(
                "The weaviate-client package is required for WeaviateDB. "
                "Please install it with `pip install weaviate-client`."
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

        # Add the documents in batches
        with self.client.batch as batch:
            batch.batch_size = self.batch_size

            for i in range(len(documents)):
                # Prepare the properties
                properties = {"text": documents[i]}

                # Add metadata properties
                for key, value in metadatas[i].items():
                    properties[key] = value

                # Add the object
                batch.add_data_object(
                    data_object=properties,
                    class_name=self.class_name,
                    uuid=ids[i],
                    vector=embeddings[i],
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
        # Prepare the filter
        where_filter = None
        if filter:
            where_filter = {"operator": "And", "operands": []}
            for key, value in filter.items():
                where_filter["operands"].append(
                    {
                        "path": [key],
                        "operator": "Equal",
                        "valueString": value,
                    }
                )

        # Search the index
        results = (
            self.client.query.get(self.class_name, ["text", "source"])
            .with_near_vector({"vector": query_embedding})
            .with_limit(top_k)
        )
        if where_filter:
            results = results.with_where(where_filter)

        results = results.do()

        # Format the results
        formatted_results = []
        if "data" in results and "Get" in results["data"]:
            for item in results["data"]["Get"][self.class_name]:
                # Extract the document text
                document = item.get("text", "")

                # Extract the metadata
                metadata = {k: v for k, v in item.items() if k != "text"}

                # Extract the distance (certainty is between 0 and 1, convert to cosine distance)
                distance = 1 - item.get("_additional", {}).get("certainty", 0)

                formatted_results.append(
                    {
                        "id": item.get("_additional", {}).get("id", ""),
                        "text": document,
                        "metadata": metadata,
                        "distance": distance,
                    }
                )

        return formatted_results

    def delete(self, ids: List[str]) -> None:
        """
        Delete documents from the vector database.

        Args:
            ids: The IDs of the documents to delete.
        """
        for doc_id in ids:
            self.client.data_object.delete(doc_id, self.class_name)

    def get(self, ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get documents from the vector database.

        Args:
            ids: The IDs of the documents to get.

        Returns:
            A list of dictionaries containing the documents.
        """
        formatted_results = []
        for doc_id in ids:
            try:
                # Get the object
                result = self.client.data_object.get_by_id(
                    doc_id, self.class_name, with_vector=True
                )

                # Extract the document text
                document = result.get("properties", {}).get("text", "")

                # Extract the metadata
                metadata = {
                    k: v for k, v in result.get("properties", {}).items() if k != "text"
                }

                # Extract the embedding
                embedding = result.get("vector", [])

                formatted_results.append(
                    {
                        "id": doc_id,
                        "text": document,
                        "metadata": metadata,
                        "embedding": embedding,
                    }
                )
            except Exception:
                # Skip if the object doesn't exist
                continue

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
        for i, doc_id in enumerate(ids):
            # Prepare the updates
            properties = {}

            # Update the document content if provided
            if documents and i < len(documents):
                properties["text"] = documents[i]

            # Update the metadata if provided
            if metadatas and i < len(metadatas):
                for key, value in metadatas[i].items():
                    properties[key] = value

            # Update the object
            if properties:
                self.client.data_object.update(properties, self.class_name, doc_id)

            # Update the embedding if provided
            if embeddings and i < len(embeddings):
                self.client.data_object.update_vector(
                    doc_id, self.class_name, embeddings[i]
                )

    def count(self) -> int:
        """
        Get the number of documents in the vector database.

        Returns:
            The number of documents.
        """
        result = self.client.query.aggregate(self.class_name).with_meta_count().do()
        return (
            result.get("data", {})
            .get("Aggregate", {})
            .get(self.class_name, [{}])[0]
            .get("meta", {})
            .get("count", 0)
        )

    def clear(self) -> None:
        """
        Clear all documents from the vector database.
        """
        self.client.schema.delete_class(self.class_name)
        self._setup_client()


class QdrantDB(VectorDBBase):
    """
    Vector database using Qdrant.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Qdrant vector database.

        Args:
            config: A dictionary containing configuration parameters.
                - url: The Qdrant URL (default: "http://localhost:6333").
                - api_key: The Qdrant API key (optional).
                - collection_name: The name of the collection to use (default: "documents").
                - embedding_dimension: The dimension of the embeddings (default: 768).
        """
        super().__init__(config)
        self.url = self.config.get("url", "http://localhost:6333")
        self.api_key = self.config.get("api_key")
        self.collection_name = self.config.get("collection_name", "documents")
        self.embedding_dimension = self.config.get("embedding_dimension", 768)
        self.client = None
        self._setup_client()

    def _setup_client(self) -> None:
        """
        Set up the Qdrant client.
        """
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models

            # Create the client
            self.client = QdrantClient(url=self.url, api_key=self.api_key)

            # Check if the collection exists
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]

            if self.collection_name not in collection_names:
                # Create the collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.embedding_dimension,
                        distance=models.Distance.COSINE,
                    ),
                )
        except ImportError:
            raise ImportError(
                "The qdrant-client package is required for QdrantDB. "
                "Please install it with `pip install qdrant-client`."
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

        from qdrant_client.http import models

        # Prepare the points to upsert
        points = []
        for i in range(len(documents)):
            # Prepare the payload
            payload = {"text": documents[i]}

            # Add metadata to payload
            for key, value in metadatas[i].items():
                payload[key] = value

            # Create the point
            points.append(
                models.PointStruct(
                    id=ids[i],
                    vector=embeddings[i],
                    payload=payload,
                )
            )

        # Upsert the points
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
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
        from qdrant_client.http import models

        # Prepare the filter
        search_filter = None
        if filter:
            conditions = []
            for key, value in filter.items():
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value),
                    )
                )
            search_filter = models.Filter(
                must=conditions,
            )

        # Search the collection
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            query_filter=search_filter,
        )

        # Format the results
        formatted_results = []
        for point in results.points:
            # Extract the document text
            text = point.payload.get("text", "")

            # Extract the metadata
            metadata = {k: v for k, v in point.payload.items() if k != "text"}

            formatted_results.append(
                {
                    "id": point.id,
                    "text": text,
                    "metadata": metadata,
                    "distance": 1 - point.score,  # Convert similarity to distance
                    "version": point.version,  # Add version from ScoredPoint
                }
            )

        return formatted_results

    def delete(self, ids: List[str]) -> None:
        """
        Delete documents from the vector database.

        Args:
            ids: The IDs of the documents to delete.
        """
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=ids,
        )

    def get(self, ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get documents from the vector database.

        Args:
            ids: The IDs of the documents to get.

        Returns:
            A list of dictionaries containing the documents.
        """
        # Get the points
        results = self.client.retrieve(
            collection_name=self.collection_name,
            ids=ids,
            with_vectors=True,
        )

        # Format the results
        formatted_results = []
        for result in results:
            # Extract the document text
            document = result.payload.get("text", "")

            # Extract the metadata
            metadata = {k: v for k, v in result.payload.items() if k != "text"}

            formatted_results.append(
                {
                    "id": result.id,
                    "text": document,
                    "metadata": metadata,
                    "embedding": result.vector,
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
        from qdrant_client.http import models

        for i, doc_id in enumerate(ids):
            # Update the payload if needed
            if (documents and i < len(documents)) or (metadatas and i < len(metadatas)):
                # Get the current document
                current_docs = self.get([doc_id])
                if not current_docs:
                    continue

                current_doc = current_docs[0]

                # Prepare the payload
                payload = {}

                # Update the document content if provided
                if documents and i < len(documents):
                    payload["text"] = documents[i]
                else:
                    payload["text"] = current_doc["document"]

                # Update the metadata if provided
                current_metadata = current_doc["metadata"]
                if metadatas and i < len(metadatas):
                    current_metadata.update(metadatas[i])

                # Add metadata to payload
                for key, value in current_metadata.items():
                    payload[key] = value

                # Update the payload
                self.client.set_payload(
                    collection_name=self.collection_name,
                    payload=payload,
                    points=[doc_id],
                )

            # Update the vector if provided
            if embeddings and i < len(embeddings):
                self.client.update_vectors(
                    collection_name=self.collection_name,
                    points=[
                        models.PointVectors(
                            id=doc_id,
                            vector=embeddings[i],
                        )
                    ],
                )

    def count(self) -> int:
        """
        Get the number of documents in the vector database.

        Returns:
            The number of documents.
        """
        collection_info = self.client.get_collection(self.collection_name)
        return collection_info.vectors_count

    def clear(self) -> None:
        """
        Clear all documents from the vector database.
        """
        self.client.delete_collection(self.collection_name)
        self._setup_client()


class RedisDB(VectorDBBase):
    """
    Vector database using Redis with vector search.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Redis vector database.

        Args:
            config: A dictionary containing configuration parameters.
                - host: The Redis host (default: "localhost").
                - port: The Redis port (default: 6379).
                - password: The Redis password (optional).
                - index_name: The name of the index to use (default: "documents").
                - prefix: The prefix to use for keys (default: "doc:").
                - embedding_dimension: The dimension of the embeddings (default: 768).
        """
        super().__init__(config)
        self.host = self.config.get("host", "localhost")
        self.port = self.config.get("port", 6379)
        self.password = self.config.get("password")
        self.index_name = self.config.get("index_name", "documents")
        self.prefix = self.config.get("prefix", "doc:")
        self.embedding_dimension = self.config.get("embedding_dimension", 768)
        self.client = None
        self._setup_client()

    def _setup_client(self) -> None:
        """
        Set up the Redis client.
        """
        try:
            import redis
            from redis.commands.search.field import TextField, VectorField
            from redis.commands.search.indexDefinition import IndexDefinition, IndexType

            # Create the client
            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                password=self.password,
                decode_responses=True,
            )

            # Check if the index exists
            try:
                self.client.ft(self.index_name).info()
            except:
                # Create the index
                schema = (
                    TextField("text"),
                    TextField("source"),
                    VectorField(
                        "embedding",
                        "FLAT",
                        {
                            "TYPE": "FLOAT32",
                            "DIM": self.embedding_dimension,
                            "DISTANCE_METRIC": "COSINE",
                        },
                    ),
                )

                self.client.ft(self.index_name).create_index(
                    schema,
                    definition=IndexDefinition(
                        prefix=[self.prefix], index_type=IndexType.HASH
                    ),
                )
        except ImportError:
            raise ImportError(
                "The redis package is required for RedisDB. "
                "Please install it with `pip install redis`."
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

        import numpy as np

        # Add the documents
        pipeline = self.client.pipeline()
        for i in range(len(documents)):
            # Convert embedding to bytes
            embedding_bytes = np.array(embeddings[i], dtype=np.float32).tobytes()

            # Prepare the hash fields
            hash_fields = {
                "text": documents[i],
                "embedding": embedding_bytes,
            }

            # Add metadata fields
            for key, value in metadatas[i].items():
                hash_fields[key] = value

            # Add the hash
            key = f"{self.prefix}{ids[i]}"
            pipeline.hset(key, mapping=hash_fields)

        # Execute the pipeline
        pipeline.execute()

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
        import numpy as np

        # Prepare the query
        query = f"*=>[KNN {top_k} @embedding $embedding AS distance]"

        # Add filter if provided
        if filter:
            filter_conditions = []
            for key, value in filter.items():
                filter_conditions.append(f"@{key}:{{{value}}}")

            if filter_conditions:
                filter_str = " ".join(filter_conditions)
                query = f"{filter_str}=>[KNN {top_k} @embedding $embedding AS distance]"

        # Convert embedding to bytes
        embedding_bytes = np.array(query_embedding, dtype=np.float32).tobytes()

        # Execute the search
        results = self.client.ft(self.index_name).search(
            query,
            {"embedding": embedding_bytes},
        )

        # Format the results
        formatted_results = []
        for result in results.docs:
            # Extract the document ID
            doc_id = result.id.replace(self.prefix, "")

            # Extract the document text
            document = result.text

            # Extract the metadata
            metadata = {
                k: v
                for k, v in result.__dict__.items()
                if k not in ["id", "text", "embedding", "distance"]
            }

            # Extract the distance
            distance = float(result.distance)

            formatted_results.append(
                {
                    "id": doc_id,
                    "text": document,
                    "metadata": metadata,
                    "distance": distance,
                }
            )

        return formatted_results

    def delete(self, ids: List[str]) -> None:
        """
        Delete documents from the vector database.

        Args:
            ids: The IDs of the documents to delete.
        """
        pipeline = self.client.pipeline()
        for doc_id in ids:
            key = f"{self.prefix}{doc_id}"
            pipeline.delete(key)

        pipeline.execute()

    def get(self, ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get documents from the vector database.

        Args:
            ids: The IDs of the documents to get.

        Returns:
            A list of dictionaries containing the documents.
        """
        import numpy as np

        formatted_results = []
        for doc_id in ids:
            key = f"{self.prefix}{doc_id}"
            result = self.client.hgetall(key)

            if result:
                # Extract the document text
                document = result.get("text", "")

                # Extract the metadata
                metadata = {
                    k: v for k, v in result.items() if k not in ["text", "embedding"]
                }

                # Extract the embedding
                embedding_bytes = result.get("embedding", b"")
                embedding = (
                    np.frombuffer(embedding_bytes, dtype=np.float32).tolist()
                    if embedding_bytes
                    else []
                )

                formatted_results.append(
                    {
                        "id": doc_id,
                        "text": document,
                        "metadata": metadata,
                        "embedding": embedding,
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
        import numpy as np

        pipeline = self.client.pipeline()
        for i, doc_id in enumerate(ids):
            key = f"{self.prefix}{doc_id}"

            # Get the current document
            current = self.client.hgetall(key)
            if not current:
                continue

            # Prepare the updates
            updates = {}

            # Update the document content if provided
            if documents and i < len(documents):
                updates["text"] = documents[i]

            # Update the embedding if provided
            if embeddings and i < len(embeddings):
                embedding_bytes = np.array(embeddings[i], dtype=np.float32).tobytes()
                updates["embedding"] = embedding_bytes

            # Update the metadata if provided
            if metadatas and i < len(metadatas):
                for key, value in metadatas[i].items():
                    updates[key] = value

            # Update the hash
            if updates:
                pipeline.hset(key, mapping=updates)

        # Execute the pipeline
        pipeline.execute()

    def count(self) -> int:
        """
        Get the number of documents in the vector database.

        Returns:
            The number of documents.
        """
        # Get the index info
        info = self.client.ft(self.index_name).info()

        # Extract the number of documents
        return info["num_docs"]

    def clear(self) -> None:
        """
        Clear all documents from the vector database.
        """
        # Get all keys with the prefix
        keys = self.client.keys(f"{self.prefix}*")

        # Delete all keys
        if keys:
            self.client.delete(*keys)


class ElasticsearchDB(VectorDBBase):
    """
    Vector database using Elasticsearch with vector search.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Elasticsearch vector database.

        Args:
            config: A dictionary containing configuration parameters.
                - hosts: The Elasticsearch hosts (default: ["http://localhost:9200"]).
                - api_key: The Elasticsearch API key (optional).
                - username: The Elasticsearch username (optional).
                - password: The Elasticsearch password (optional).
                - index_name: The name of the index to use (default: "documents").
                - embedding_dimension: The dimension of the embeddings (default: 768).
        """
        super().__init__(config)
        self.hosts = self.config.get("hosts", ["http://localhost:9200"])
        self.api_key = self.config.get("api_key")
        self.username = self.config.get("username")
        self.password = self.config.get("password")
        self.index_name = self.config.get("index_name", "documents")
        self.embedding_dimension = self.config.get("embedding_dimension", 768)
        self.client = None
        self._setup_client()

    def _setup_client(self) -> None:
        """
        Set up the Elasticsearch client.
        """
        try:
            from elasticsearch import Elasticsearch

            # Create the client
            auth = {}
            if self.api_key:
                auth["api_key"] = self.api_key
            elif self.username and self.password:
                auth["basic_auth"] = (self.username, self.password)

            self.client = Elasticsearch(
                hosts=self.hosts,
                **auth,
            )

            # Check if the index exists
            if not self.client.indices.exists(index=self.index_name):
                # Create the index
                self.client.indices.create(
                    index=self.index_name,
                    body={
                        "mappings": {
                            "properties": {
                                "text": {"type": "text"},
                                "source": {"type": "keyword"},
                                "embedding": {
                                    "type": "dense_vector",
                                    "dims": self.embedding_dimension,
                                    "index": True,
                                    "similarity": "cosine",
                                },
                            }
                        }
                    },
                )
        except ImportError:
            raise ImportError(
                "The elasticsearch package is required for ElasticsearchDB. "
                "Please install it with `pip install elasticsearch`."
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

        # Add the documents
        operations = []
        for i in range(len(documents)):
            # Prepare the document
            doc = {
                "text": documents[i],
                "embedding": embeddings[i],
            }

            # Add metadata fields
            for key, value in metadatas[i].items():
                doc[key] = value

            # Add the index operation
            operations.append({"index": {"_index": self.index_name, "_id": ids[i]}})
            operations.append(doc)

        # Execute the bulk operation
        if operations:
            self.client.bulk(operations=operations, refresh=True)

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
        # Prepare the query
        query = {
            "knn": {
                "field": "embedding",
                "query_vector": query_embedding,
                "k": top_k,
                "num_candidates": top_k * 2,
            }
        }

        # Add filter if provided
        if filter:
            filter_conditions = []
            for key, value in filter.items():
                filter_conditions.append({"term": {key: value}})

            if filter_conditions:
                query = {
                    "bool": {
                        "must": query,
                        "filter": filter_conditions,
                    }
                }

        # Execute the search
        results = self.client.search(
            index=self.index_name,
            query=query,
            size=top_k,
        )

        # Format the results
        formatted_results = []
        for hit in results["hits"]["hits"]:
            # Extract the document ID
            doc_id = hit["_id"]

            # Extract the document text
            document = hit["_source"].get("text", "")

            # Extract the metadata
            metadata = {
                k: v
                for k, v in hit["_source"].items()
                if k not in ["text", "embedding"]
            }

            # Extract the distance (score is similarity, convert to distance)
            distance = 1 - hit["_score"]

            formatted_results.append(
                {
                    "id": doc_id,
                    "text": document,
                    "metadata": metadata,
                    "distance": distance,
                }
            )

        return formatted_results

    def delete(self, ids: List[str]) -> None:
        """
        Delete documents from the vector database.

        Args:
            ids: The IDs of the documents to delete.
        """
        operations = []
        for doc_id in ids:
            operations.append({"delete": {"_index": self.index_name, "_id": doc_id}})

        if operations:
            self.client.bulk(operations=operations, refresh=True)

    def get(self, ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get documents from the vector database.

        Args:
            ids: The IDs of the documents to get.

        Returns:
            A list of dictionaries containing the documents.
        """
        # Prepare the query
        query = {
            "ids": {
                "values": ids,
            }
        }

        # Execute the search
        results = self.client.search(
            index=self.index_name,
            query=query,
            size=len(ids),
        )

        # Format the results
        formatted_results = []
        for hit in results["hits"]["hits"]:
            # Extract the document ID
            doc_id = hit["_id"]

            # Extract the document text
            document = hit["_source"].get("text", "")

            # Extract the metadata
            metadata = {
                k: v
                for k, v in hit["_source"].items()
                if k not in ["text", "embedding"]
            }

            # Extract the embedding
            embedding = hit["_source"].get("embedding", [])

            formatted_results.append(
                {
                    "id": doc_id,
                    "text": document,
                    "metadata": metadata,
                    "embedding": embedding,
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
        operations = []
        for i, doc_id in enumerate(ids):
            # Prepare the updates
            doc = {}

            # Update the document content if provided
            if documents and i < len(documents):
                doc["text"] = documents[i]

            # Update the embedding if provided
            if embeddings and i < len(embeddings):
                doc["embedding"] = embeddings[i]

            # Update the metadata if provided
            if metadatas and i < len(metadatas):
                for key, value in metadatas[i].items():
                    doc[key] = value

            # Add the update operation
            if doc:
                operations.append(
                    {"update": {"_index": self.index_name, "_id": doc_id}}
                )
                operations.append({"doc": doc})

        # Execute the bulk operation
        if operations:
            self.client.bulk(operations=operations, refresh=True)

    def count(self) -> int:
        """
        Get the number of documents in the vector database.

        Returns:
            The number of documents.
        """
        # Get the index stats
        stats = self.client.count(index=self.index_name)

        # Extract the number of documents
        return stats["count"]

    def clear(self) -> None:
        """
        Clear all documents from the vector database.
        """
        # Delete the index
        self.client.indices.delete(index=self.index_name)

        # Recreate the index
        self._setup_client()


class PgVectorDB(VectorDBBase):
    """
    Vector database using PostgreSQL with pgvector extension.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the PostgreSQL vector database.

        Args:
            config: A dictionary containing configuration parameters.
                - connection_string: The PostgreSQL connection string (required).
                - table_name: The name of the table to use (default: "documents").
                - embedding_dimension: The dimension of the embeddings (default: 768).
        """
        super().__init__(config)
        self.connection_string = self.config.get("connection_string")
        if not self.connection_string:
            raise ValueError("PostgreSQL connection string is required")

        self.table_name = self.config.get("table_name", "documents")
        self.embedding_dimension = self.config.get("embedding_dimension", 768)
        self.conn = None
        self._setup_client()

    def _setup_client(self) -> None:
        """
        Set up the PostgreSQL client.
        """
        try:
            import psycopg2

            # Connect to the database
            self.conn = psycopg2.connect(self.connection_string)

            # Create the pgvector extension if it doesn't exist
            with self.conn.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

                # Create the table if it doesn't exist
                cursor.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id TEXT PRIMARY KEY,
                        text TEXT NOT NULL,
                        embedding vector({self.embedding_dimension}) NOT NULL,
                        metadata JSONB
                    );
                """
                )

                # Create an index for vector similarity search
                cursor.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx
                    ON {self.table_name}
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);
                """
                )

                self.conn.commit()
        except ImportError:
            raise ImportError(
                "The psycopg2 package is required for PgVectorDB. "
                "Please install it with `pip install psycopg2-binary`."
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

        import json

        # Add the documents
        with self.conn.cursor() as cursor:
            for i in range(len(documents)):
                # Convert metadata to JSON
                metadata_json = json.dumps(metadatas[i])

                # Insert the document
                cursor.execute(
                    f"""
                    INSERT INTO {self.table_name} (id, text, embedding, metadata)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE
                    SET text = EXCLUDED.text,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata;
                    """,
                    (ids[i], documents[i], embeddings[i], metadata_json),
                )

            self.conn.commit()

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
        # Prepare the query
        query = f"""
            SELECT id, text, metadata, 1 - (embedding <=> %s) as similarity
            FROM {self.table_name}
        """

        # Add filter if provided
        params = [query_embedding]
        if filter:
            filter_conditions = []
            for key, value in filter.items():
                filter_conditions.append(f"metadata->>'%s' = %s")
                params.extend([key, value])

            if filter_conditions:
                filter_str = " AND ".join(filter_conditions)
                query += f" WHERE {filter_str}"

        # Add order by and limit
        query += f" ORDER BY similarity DESC LIMIT {top_k}"

        # Execute the search
        with self.conn.cursor() as cursor:
            cursor.execute(query, params)
            results = cursor.fetchall()

        # Format the results
        formatted_results = []
        for result in results:
            doc_id, text, metadata_json, similarity = result

            # Parse the metadata
            metadata = json.loads(metadata_json) if metadata_json else {}

            formatted_results.append(
                {
                    "id": doc_id,
                    "text": text,
                    "metadata": metadata,
                    "distance": 1 - similarity,
                }
            )

        return formatted_results

    def delete(self, ids: List[str]) -> None:
        """
        Delete documents from the vector database.

        Args:
            ids: The IDs of the documents to delete.
        """
        # Delete the documents
        with self.conn.cursor() as cursor:
            cursor.execute(
                f"DELETE FROM {self.table_name} WHERE id = ANY(%s)",
                (ids,),
            )

            self.conn.commit()

    def get(self, ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get documents from the vector database.

        Args:
            ids: The IDs of the documents to get.

        Returns:
            A list of dictionaries containing the documents.
        """
        # Get the documents
        with self.conn.cursor() as cursor:
            cursor.execute(
                f"SELECT id, text, embedding, metadata FROM {self.table_name} WHERE id = ANY(%s)",
                (ids,),
            )
            results = cursor.fetchall()

        # Format the results
        formatted_results = []
        for result in results:
            doc_id, text, embedding, metadata_json = result

            # Parse the metadata
            metadata = json.loads(metadata_json) if metadata_json else {}

            formatted_results.append(
                {
                    "id": doc_id,
                    "text": text,
                    "metadata": metadata,
                    "embedding": embedding,
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
        # Get the current documents
        current_docs = self.get(ids)

        # Update the documents
        with self.conn.cursor() as cursor:
            for i, doc_id in enumerate(ids):
                if i < len(current_docs):
                    current_doc = current_docs[i]

                    # Update the document content if provided
                    text = (
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

                    # Convert metadata to JSON
                    metadata_json = json.dumps(metadata)

                    # Update the document
                    cursor.execute(
                        f"""
                        UPDATE {self.table_name}
                        SET text = %s, embedding = %s, metadata = %s
                        WHERE id = %s
                        """,
                        (text, embedding, metadata_json, doc_id),
                    )

            self.conn.commit()

    def count(self) -> int:
        """
        Get the number of documents in the vector database.

        Returns:
            The number of documents.
        """
        # Count the documents
        with self.conn.cursor() as cursor:
            cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
            result = cursor.fetchone()

        return result[0] if result else 0

    def clear(self) -> None:
        """
        Clear all documents from the vector database.
        """
        # Delete all documents
        with self.conn.cursor() as cursor:
            cursor.execute(f"DELETE FROM {self.table_name}")
            self.conn.commit()


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
                    "text": results["documents"][0][i],
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
                    "text": results["documents"][i],
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
                "text": documents[i],
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
                        "text": self.documents[doc_id]["document"],
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
                        "text": doc["document"],
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
                        "text": hit.entity.get("document"),
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
                    "text": result["document"],
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
