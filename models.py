"""Data models for PDF processing and chunking."""

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Optional, Dict, Any


@dataclass
class ImageAsset:
    """Represents an extracted image with metadata."""
    path: Path
    page: int
    width: int
    height: int
    description: Optional[str] = None
    entities: Optional[List[Dict[str, str]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['path'] = str(self.path)
        return data


@dataclass
class TableAsset:
    """Represents a reconstructed table."""
    page: int
    markdown_content: str
    source_type: str  # 'vision' (from image) or 'docling' (from native extraction)
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProcessingResult:
    """Results from PDF processing."""
    document_id: str
    total_pages: int
    markdown_path: Path
    images: List[ImageAsset]
    tables: List[TableAsset]
    stats: Dict[str, int]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'document_id': self.document_id,
            'total_pages': self.total_pages,
            'markdown_path': str(self.markdown_path),
            'images': [img.to_dict() for img in self.images],
            'tables': [tbl.to_dict() for tbl in self.tables],
            'stats': self.stats
        }


# ─── Chunk Data Models ───────────────────────────────────────────────────────


@dataclass
class TextChunk:
    """A text chunk extracted from a document section."""
    chunk_id: str                   # "{doc_id}_text_{section_id}_{idx}"
    document_id: str
    content: str                    # The chunk text
    section_path: List[str]         # Hierarchical header path: ["H1 title", "H2 title", ...]
    section_id: str                 # Slug of the deepest section header
    section_level: int              # Deepest header level (1-6)
    chunk_index: int                # Position within section
    char_count: int
    modality: str = "text"
    page_estimate: Optional[int] = None
    language: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TableChunk:
    """A table treated as a single atomic chunk."""
    chunk_id: str                   # "{doc_id}_table_{page}_{idx}"
    document_id: str
    content: str                    # Full markdown table
    description: str                # AI-generated description
    source_type: str                # "docling" or "vision"
    page: int
    row_count: int
    column_count: int
    char_count: int
    modality: str = "table"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ImageChunk:
    """An image represented as a chunk via its description + entities."""
    chunk_id: str                   # "{doc_id}_img_{page}_{idx}"
    document_id: str
    content: str                    # AI-generated description
    entities: List[Dict[str, str]]  # Extracted entities
    image_path: str                 # Path to saved PNG
    page: int
    width: int
    height: int
    char_count: int
    modality: str = "image"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ChunkingResult:
    """Aggregated chunking output for a document."""
    document_id: str
    text_chunks: List[TextChunk] = field(default_factory=list)
    table_chunks: List[TableChunk] = field(default_factory=list)
    image_chunks: List[ImageChunk] = field(default_factory=list)
    stats: Dict[str, int] = field(default_factory=dict)
    
    @property
    def all_chunks(self) -> List[Any]:
        """Return all chunks in a flat list, ordered: text → table → image."""
        return self.text_chunks + self.table_chunks + self.image_chunks
    
    @property
    def total_chunks(self) -> int:
        return len(self.text_chunks) + len(self.table_chunks) + len(self.image_chunks)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'document_id': self.document_id,
            'total_chunks': self.total_chunks,
            'text_chunks': [c.to_dict() for c in self.text_chunks],
            'table_chunks': [c.to_dict() for c in self.table_chunks],
            'image_chunks': [c.to_dict() for c in self.image_chunks],
            'stats': self.stats
        }
