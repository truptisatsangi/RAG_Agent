from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from models.document_metadata import DocumentMetadata


class ChunkingService:
    """
    Responsible for splitting extracted text into semantically meaningful chunks.

    Responsibilities:
    - Apply chunking strategy.
    - Configure chunk size and overlap.
    - Preserve metadata.
    - Return chunks ready for embedding.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def create_chunks(
        self,
        text: str,
        metadata: DocumentMetadata,
    ) -> list[Document]:
        """Split text into chunks while preserving metadata."""

        chunks = self.text_splitter.create_documents(
            texts=[text],
            metadatas=[metadata.model_dump()],
        )

        return chunks