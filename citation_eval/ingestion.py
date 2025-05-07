import os
import json
import pickle
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


@dataclass
class PDFDocument:
    """
    Dataclass to hold information about a single PDF document.
    This can store various types of extracted/processed data.
    """
    filename: str  # Without .pdf extension
    full_path: str | Path
    full_text: str = ""
    chunks: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Extensible storage for any additional fields
    custom_data: Dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, name):
        """Allow attribute-style access to custom_data fields."""
        if 'custom_data' in self.__dict__ and name in self.custom_data:
            return self.custom_data[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        """Set attributes normally but handle special case for dynamic fields."""
        # Normal attributes (defined in dataclass) are set normally
        if name in {'filename', 'full_path', 'full_text', 'chunks', 'metadata', 'custom_data'}:
            super().__setattr__(name, value)
        else:
            # Any other attribute goes into custom_data
            if 'custom_data' not in self.__dict__:
                # Initialize custom_data if it doesn't exist yet (during dataclass init)
                super().__setattr__('custom_data', {})
            self.custom_data[name] = value

    def __delattr__(self, name):
        """Allow deletion of custom_data fields."""
        if 'custom_data' in self.__dict__ and name in self.custom_data:
            del self.custom_data[name]
        else:
            super().__delattr__(name)

    def list_custom_fields(self) -> List[str]:
        """Return a list of all custom field names."""
        return list(self.custom_data.keys())

    def has_field(self, field_name: str) -> bool:
        """Check if a custom field exists."""
        return field_name in self.custom_data

    def __repr__(self):
        """Custom representation showing custom fields."""
        base_repr = f"PDFDocument(filename='{self.filename}', custom_fields={list(self.custom_data.keys())})"
        return base_repr


class PDFCollection:
    """
    A class to manage a collection of PDF documents and all associated data.
    This single class object holds information for all PDFs in a directory.
    """

    def __init__(self, pdf_directory: str | Path, cache_directory: str | Path = None):
        """
        Initialize the PDF collection.

        Args:
            pdf_directory: Directory containing PDF files
            cache_directory: Directory to save/load cached data (optional)
        """
        self.pdf_directory = pdf_directory
        self.cache_directory = cache_directory or os.path.join(pdf_directory, ".pdf_cache")

        # Core data storage
        self.documents: Dict[str, PDFDocument] = {}  # Key: filename without extension
        self.metadata: Dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "pdf_directory": pdf_directory,
        }

        # Vector database for search functionality
        self.vectorstore: Optional[FAISS] = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

        # Create cache directory
        os.makedirs(self.cache_directory, exist_ok=True)

        # Load cached data if available
        self._load_cache()

    def _load_cache(self) -> None:
        """Load cached data from disk if available."""
        cache_file = os.path.join(self.cache_directory, "collection_cache.pkl")

        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.documents = cached_data.get('documents', {})
                    self.metadata = cached_data.get('metadata', self.metadata)
                print(f"Loaded cached data for {len(self.documents)} documents")

                # Load vectorstore if exists
                vectorstore_path = os.path.join(self.cache_directory, "vectorstore")
                if os.path.exists(vectorstore_path):
                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )
                    self.vectorstore = FAISS.load_local(vectorstore_path, embeddings)
                    print("Loaded vector database")
            except Exception as e:
                print(f"Error loading cache: {e}")

    def _save_cache(self) -> None:
        """Save the collection data to disk."""
        cache_file = os.path.join(self.cache_directory, "collection_cache.pkl")

        cache_data = {
            'documents': self.documents,
            'metadata': self.metadata
        }

        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)

        # Save vectorstore if exists
        if self.vectorstore:
            vectorstore_path = os.path.join(self.cache_directory, "vectorstore")
            self.vectorstore.save_local(vectorstore_path)

        print("Cache saved successfully")

    def process_pdfs(self, force_reprocess: bool = False) -> None:
        """
        Process all PDFs in the directory and extract their content.

        Args:
            force_reprocess: If True, reprocess all PDFs even if cached
        """
        pdf_files = list(Path(self.pdf_directory).glob("*.pdf"))

        if not pdf_files:
            print(f"No PDF files found in {self.pdf_directory}")
            return

        print(f"Processing {len(pdf_files)} PDF files...")

        for pdf_path in pdf_files:
            filename = pdf_path.stem  # Without extension

            # Skip if already processed and not forcing reprocessing
            if filename in self.documents and not force_reprocess:
                continue

            print(f"Processing {filename}...")

            # Extract text from PDF
            text = self._extract_text_from_pdf(str(pdf_path))

            # Create chunks
            chunks = self.text_splitter.split_text(text)

            # Create or update document
            self.documents[filename] = PDFDocument(
                filename=filename,
                full_path=str(pdf_path),
                full_text=text,
                chunks=chunks,
                metadata={
                    "file_size": os.path.getsize(pdf_path),
                    "processed_at": datetime.now().isoformat(),
                    "num_chunks": len(chunks),
                }
            )

        # Update collection metadata
        self.metadata["last_processed"] = datetime.now().isoformat()
        self.metadata["total_documents"] = len(self.documents)

        # Create vector index
        self._create_vector_index()

        # Save cache
        self._save_cache()

        print(f"Processing complete. {len(self.documents)} documents in collection.")

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file."""
        doc = fitz.open(pdf_path)
        text = ""

        for page in doc:
            text += page.get_text()

        return text

    def _create_vector_index(self) -> None:
        """Create or update the vector index for all document chunks."""
        if not self.documents:
            print("No documents to index")
            return

        # Gather all chunks and metadata
        all_chunks = []
        all_metadata = []

        for doc in self.documents.values():
            for i, chunk in enumerate(doc.chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    "source": doc.filename,
                    "chunk_index": i,
                })

        # Create embeddings and vectorstore
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        if self.vectorstore:
            # Update existing vectorstore
            self.vectorstore.delete_collection() if hasattr(self.vectorstore, 'delete_collection') else None

        self.vectorstore = FAISS.from_texts(
            texts=all_chunks,
            embedding=embeddings,
            metadatas=all_metadata
        )

        print("Vector index created/updated")

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search across all documents in the collection.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of search results with document information
        """
        if not self.vectorstore:
            raise ValueError("Vector index not created. Run process_pdfs() first.")

        results = self.vectorstore.similarity_search_with_score(query, k=k)

        formatted_results = []
        for doc, score in results:
            source_doc = self.documents.get(doc.metadata["source"])
            result = {
                "content": doc.page_content,
                "score": score,
                "source": doc.metadata["source"],
                "chunk_index": doc.metadata["chunk_index"],
                "full_text": source_doc.full_text if source_doc else None,
                "metadata": source_doc.metadata if source_doc else {}
            }
            formatted_results.append(result)

        return formatted_results

    # Methods for adding custom data to documents
    def add_document_field(self, filename: str, field_name: str, value: Any) -> None:
        """
        Add a custom field to a specific document.

        Args:
            filename: Document filename (without extension)
            field_name: Name of the field to add
            value: Value to store
        """
        if filename not in self.documents:
            raise ValueError(f"Document '{filename}' not found")

        setattr(self.documents[filename], field_name, value)
        print(f"Added field '{field_name}' to document '{filename}'")

    def get_document(self, filename: str) -> Optional[PDFDocument]:
        """Get a specific document by filename."""
        return self.documents.get(filename)

    def get_all_filenames(self) -> List[str]:
        """Get a list of all document filenames."""
        return list(self.documents.keys())

    def get_field_from_all_documents(self, field_name: str) -> Dict[str, Any]:
        """
        Get a specific field from all documents.

        Args:
            field_name: Name of the field to retrieve

        Returns:
            Dictionary mapping filename to field value
        """
        result = {}
        for filename, doc in self.documents.items():
            if field_name in doc.custom_data:
                result[filename] = doc.custom_data[field_name]
        return result

    def export_to_json(self, output_path: str) -> None:
        """
        Export the entire collection to a JSON file.

        Args:
            output_path: Path where to save the JSON file
        """
        export_data = {
            "metadata": self.metadata,
            "documents": {}
        }

        for filename, doc in self.documents.items():
            export_data["documents"][filename] = {
                "filename": doc.filename,
                "full_path": doc.full_path,
                "full_text": doc.full_text,
                "metadata": doc.metadata,
                "custom_data": doc.custom_data,
                "num_chunks": len(doc.chunks)
            }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        print(f"Collection exported to {output_path}")

    def summarize(self) -> Dict[str, Any]:
        """Get a summary of the collection."""
        return {
            "total_documents": len(self.documents),
            "total_chunks": sum(len(doc.chunks) for doc in self.documents.values()),
            "average_document_length": sum(len(doc.full_text) for doc in self.documents.values()) / len(
                self.documents) if self.documents else 0,
            "metadata": self.metadata,
            "vector_index_exists": self.vectorstore is not None
        }


# Example usage
if __name__ == "__main__":
    # Create collection
    collection = PDFCollection(pdf_directory="pdfs")

    # Process all PDFs
    collection.process_pdfs()

    # Add custom fields
    # Example 1: Add a field to a specific document
    collection.add_document_field("document1", "category", "research")

    # Example 2: Add a field to all documents
    collection.add_all_documents_field(
        "word_count",
        lambda doc: len(doc.full_text.split())
    )

    # Example 3: Add computed fields
    collection.add_all_documents_field(
        "contains_data",
        lambda doc: "data" in doc.full_text.lower()
    )

    # Search across all documents
    results = collection.search("your search query")

    # Get summary
    summary = collection.summarize()
    print(json.dumps(summary, indent=2))

    # Export to JSON
    collection.export_to_json("collection_export.json")

    # Access specific document
    doc = collection.get_document("document1")
    if doc:
        print(f"Document '{doc.filename}' has {len(doc.chunks)} chunks")
        print(f"Custom data: {doc.custom_data}")