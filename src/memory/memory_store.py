"""MemoryStore - Vector DB Wrapper for storing and retrieving memories.

This module provides a ChromaDB-based vector store for the RAG-based
stateless memory architecture.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import time

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False


class MemoryType(Enum):
    """Types of memories stored in the database."""
    OBSERVATION = "observation"  # Keyframe observations (agent position, room visited)
    OBJECT = "object"            # Object states (location, state like open/closed)
    RELATION = "relation"        # Relationships between objects (Pot ON StoveBurner)
    ACTION = "action"            # Action history (for failed action tracking)


@dataclass
class MemoryDocument:
    """A document to be stored in the memory store."""
    content: str                           # The text content of the memory
    memory_type: MemoryType                # Type of memory
    step: int                              # The step number when this memory was created
    timestamp: float = field(default_factory=time.time)  # Unix timestamp
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    doc_id: Optional[str] = None           # Unique document ID (auto-generated if None)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "content": self.content,
            "memory_type": self.memory_type.value,
            "step": self.step,
            "timestamp": self.timestamp,
            **self.metadata
        }


class MemoryStore:
    """Vector database wrapper for storing and retrieving memories.

    Uses ChromaDB as the underlying vector store with embedding-based
    similarity search for relevant memory retrieval.
    """

    def __init__(self, collection_name: str = "agent_memory",
                 persist_directory: Optional[str] = None,
                 embedding_function: Optional[Any] = None):
        """Initialize the memory store.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory for persistent storage (None for in-memory)
            embedding_function: Custom embedding function (uses default if None)
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "chromadb is required for MemoryStore. "
                "Install it with: pip install chromadb"
            )

        # Initialize ChromaDB client
        if persist_directory:
            self._client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
        else:
            self._client = chromadb.Client(
                settings=Settings(anonymized_telemetry=False)
            )

        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )

        self._doc_counter = 0

    def add(self, document: MemoryDocument) -> str:
        """Add a document to the memory store.

        Args:
            document: The memory document to add

        Returns:
            The document ID
        """
        # Generate doc_id if not provided
        if document.doc_id is None:
            document.doc_id = f"{document.memory_type.value}_{self._doc_counter}"
            self._doc_counter += 1

        # Prepare metadata
        metadata = document.to_dict()
        del metadata["content"]  # Content is stored separately

        # Add to collection
        self._collection.add(
            ids=[document.doc_id],
            documents=[document.content],
            metadatas=[metadata]
        )

        return document.doc_id

    def add_batch(self, documents: List[MemoryDocument]) -> List[str]:
        """Add multiple documents to the memory store.

        Args:
            documents: List of memory documents to add

        Returns:
            List of document IDs
        """
        if not documents:
            return []

        ids = []
        contents = []
        metadatas = []

        for doc in documents:
            if doc.doc_id is None:
                doc.doc_id = f"{doc.memory_type.value}_{self._doc_counter}"
                self._doc_counter += 1

            ids.append(doc.doc_id)
            contents.append(doc.content)

            metadata = doc.to_dict()
            del metadata["content"]
            metadatas.append(metadata)

        self._collection.add(
            ids=ids,
            documents=contents,
            metadatas=metadatas
        )

        return ids

    def update(self, doc_id: str, document: MemoryDocument) -> None:
        """Update an existing document in the store.

        Args:
            doc_id: The document ID to update
            document: The new document content
        """
        metadata = document.to_dict()
        del metadata["content"]

        self._collection.update(
            ids=[doc_id],
            documents=[document.content],
            metadatas=[metadata]
        )

    def query(self,
              query_text: str,
              n_results: int = 5,
              memory_type: Optional[MemoryType] = None,
              min_step: Optional[int] = None,
              where_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Query the memory store for relevant documents.

        Args:
            query_text: The query text to search for
            n_results: Maximum number of results to return
            memory_type: Filter by memory type (optional)
            min_step: Only return memories from this step or later (optional)
            where_filter: Additional filter conditions (optional)

        Returns:
            List of matching documents with metadata and distances
        """
        # Build where filter
        where = {}
        if memory_type:
            where["memory_type"] = memory_type.value
        if min_step is not None:
            where["step"] = {"$gte": min_step}
        if where_filter:
            where.update(where_filter)

        # Query collection
        results = self._collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where if where else None,
            include=["documents", "metadatas", "distances"]
        )

        # Format results
        documents = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                documents.append({
                    "id": doc_id,
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if results["distances"] else None
                })

        return documents

    def query_by_type(self,
                      memory_type: MemoryType,
                      n_results: int = 10,
                      sort_by_recency: bool = True) -> List[Dict[str, Any]]:
        """Query all documents of a specific type.

        Args:
            memory_type: The type of memory to retrieve
            n_results: Maximum number of results
            sort_by_recency: If True, sort by step (newest first)

        Returns:
            List of matching documents
        """
        results = self._collection.get(
            where={"memory_type": memory_type.value},
            include=["documents", "metadatas"]
        )

        documents = []
        if results["ids"]:
            for i, doc_id in enumerate(results["ids"]):
                documents.append({
                    "id": doc_id,
                    "content": results["documents"][i],
                    "metadata": results["metadatas"][i]
                })

        # Sort by recency (step number, descending)
        if sort_by_recency:
            documents.sort(key=lambda x: x["metadata"].get("step", 0), reverse=True)

        return documents[:n_results]

    def get_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by its ID.

        Args:
            doc_id: The document ID

        Returns:
            The document dict or None if not found
        """
        results = self._collection.get(
            ids=[doc_id],
            include=["documents", "metadatas"]
        )

        if results["ids"]:
            return {
                "id": results["ids"][0],
                "content": results["documents"][0],
                "metadata": results["metadatas"][0]
            }
        return None

    def delete(self, doc_id: str) -> None:
        """Delete a document by ID.

        Args:
            doc_id: The document ID to delete
        """
        self._collection.delete(ids=[doc_id])

    def delete_by_filter(self, where_filter: Dict[str, Any]) -> None:
        """Delete documents matching a filter.

        Args:
            where_filter: Filter conditions for deletion
        """
        self._collection.delete(where=where_filter)

    def clear(self) -> None:
        """Clear all documents from the store."""
        # Get all IDs and delete them
        results = self._collection.get()
        if results["ids"]:
            self._collection.delete(ids=results["ids"])
        self._doc_counter = 0

    def count(self) -> int:
        """Get the total number of documents in the store.

        Returns:
            Number of documents
        """
        return self._collection.count()

    def exists(self, doc_id: str) -> bool:
        """Check if a document exists.

        Args:
            doc_id: The document ID to check

        Returns:
            True if document exists
        """
        results = self._collection.get(ids=[doc_id])
        return bool(results["ids"])
