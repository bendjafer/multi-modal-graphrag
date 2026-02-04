"""Data models for PDF processing."""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any


@dataclass
class ImageAsset:
    """Represents an extracted image with metadata."""
    path: Path
    page: int
    width: int
    height: int
    hash: str
    description: Optional[str] = None
    entities: Optional[List[Dict[str, str]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with Path objects as strings."""
        data = asdict(self)
        data['path'] = str(self.path)
        return data


@dataclass
class ProcessingResult:
    """Results from PDF processing."""
    document_id: str
    total_pages: int
    markdown_path: Path
    images: List[ImageAsset]
    stats: Dict[str, int]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'document_id': self.document_id,
            'total_pages': self.total_pages,
            'markdown_path': str(self.markdown_path),
            'images': [img.to_dict() for img in self.images],
            'stats': self.stats
        }
