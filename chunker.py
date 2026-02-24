"""Multimodal document chunking using LangChain text splitters.

Splits cleaned markdown into section-aware text chunks,
wraps tables and images as atomic chunks, and produces
a unified ChunkingResult for downstream processing.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE
from models import (
    ImageAsset, TableAsset,
    TextChunk, TableChunk, ImageChunk, ChunkingResult,
)

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Section-aware multimodal chunker using LangChain splitters.

    Pipeline:
        1. Strip section metadata comments (<!-- section:... -->)
        2. Remove inline markdown tables (already captured as TableChunks)
        3. Split markdown by headers using MarkdownHeaderTextSplitter
        4. Sub-split large sections using RecursiveCharacterTextSplitter
        5. Post-process: fix orphan fragments, merge small chunks
        6. Wrap each table as a single TableChunk
        7. Wrap each image as a single ImageChunk
        8. Return unified ChunkingResult
    """

    HEADERS_TO_SPLIT = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
        ("####", "h4"),
        ("#####", "h5"),
        ("######", "h6"),
    ]

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        min_chunk_size: int = MIN_CHUNK_SIZE,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

        # Header-based splitter (first pass)
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.HEADERS_TO_SPLIT,
            strip_headers=False,  # Keep headers in chunk content for retrieval context
        )

        # Recursive splitter (second pass for large sections)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
            is_separator_regex=False,
        )

    def chunk_document(
        self,
        document_id: str,
        markdown_content: str,
        tables: Optional[List[TableAsset]] = None,
        images: Optional[List[ImageAsset]] = None,
        language: str = "en",
    ) -> ChunkingResult:
        """Chunk a complete document into text, table, and image chunks."""
        tables = tables or []
        images = images or []

        text_chunks = self._chunk_text(
            document_id=document_id,
            text=markdown_content,
            language=language or "en",
        )

        table_chunks = self._chunk_tables(
            document_id=document_id,
            tables=tables,
        )

        image_chunks = self._chunk_images(
            document_id=document_id,
            images=images,
        )

        stats = {
            'text_chunks': len(text_chunks),
            'table_chunks': len(table_chunks),
            'image_chunks': len(image_chunks),
            'total_chunks': len(text_chunks) + len(table_chunks) + len(image_chunks),
            'avg_text_chunk_size': (
                sum(c.char_count for c in text_chunks) // max(len(text_chunks), 1)
            ),
            'min_text_chunk_size': (
                min((c.char_count for c in text_chunks), default=0)
            ),
            'max_text_chunk_size': (
                max((c.char_count for c in text_chunks), default=0)
            ),
        }

        logger.info(
            f"Chunked document {document_id}: "
            f"{stats['text_chunks']} text, "
            f"{stats['table_chunks']} table, "
            f"{stats['image_chunks']} image chunks "
            f"(avg text size: {stats['avg_text_chunk_size']} chars)"
        )

        return ChunkingResult(
            document_id=document_id,
            text_chunks=text_chunks,
            table_chunks=table_chunks,
            image_chunks=image_chunks,
            stats=stats,
        )

    def _clean_fragment_start(self, content: str) -> str:
        """Trim overlap fragments that start mid-sentence.

        When RecursiveCharacterTextSplitter splits mid-sentence, overlap
        chunks can start with '. ', ', ', or lowercase fragments that
        lost their preceding context.
        """
        original_length = len(content)

        content = re.sub(r'^[.,;:!?)\]\}]+\s*', '', content)

        if len(content) < 10:
            return ""

        if content and content[0].islower():
            # Some valid product/brand names legitimately start lowercase
            valid_lowercase_starts = [
                'iPhone', 'iOS', 'eBay', 'iPad', 'macOS', 'npm', 'etc',
                'eToro', 'eCommerce'
            ]
            starts_with_valid = any(content.startswith(term) for term in valid_lowercase_starts)

            if not starts_with_valid:
                sentence_boundary = re.search(r'[.!?]\s+[A-Z]', content)
                if sentence_boundary:
                    content = content[sentence_boundary.end() - 1:]
                elif len(content.split()) < 5:
                    logger.debug(f"Discarded short lowercase fragment: {content[:50]}...")
                    return ""

        content = content.strip()

        if original_length > 50 and len(content) / original_length < 0.5:
            logger.debug(f"Cleaned fragment removed {original_length - len(content)} chars")

        return content

    def _merge_small_chunks(
        self,
        chunks: List[TextChunk],
        document_id: str,
    ) -> List[TextChunk]:
        """Merge undersized chunks with their neighbors.

        Chunks below min_chunk_size are merged into the previous chunk
        (preferred when same section) or the next chunk. Eliminates
        micro-chunks that produce weak embeddings.
        """
        if not chunks:
            return chunks

        merged = []
        pending_small: List[TextChunk] = []
        merge_count = 0

        for chunk in chunks:
            if chunk.char_count < self.min_chunk_size:
                pending_small.append(chunk)
            else:
                if pending_small:
                    if merged and merged[-1].section_id == pending_small[0].section_id:
                        # Append small chunks to previous (same section — preserves context)
                        prev = merged[-1]
                        combined_content = prev.content
                        for small_chunk in pending_small:
                            combined_content += "\n\n" + small_chunk.content
                            merge_count += 1

                        merged[-1] = TextChunk(
                            chunk_id=prev.chunk_id,
                            document_id=prev.document_id,
                            content=combined_content,
                            section_path=prev.section_path,
                            section_id=prev.section_id,
                            section_level=prev.section_level,
                            chunk_index=prev.chunk_index,
                            char_count=len(combined_content),
                            language=prev.language,
                        )
                    else:
                        # Prepend small chunks to current (different section)
                        combined_content = ""
                        for small_chunk in pending_small:
                            combined_content += small_chunk.content + "\n\n"
                            merge_count += 1
                        combined_content += chunk.content

                        chunk = TextChunk(
                            chunk_id=chunk.chunk_id,
                            document_id=chunk.document_id,
                            content=combined_content,
                            section_path=chunk.section_path,
                            section_id=chunk.section_id,
                            section_level=chunk.section_level,
                            chunk_index=chunk.chunk_index,
                            char_count=len(combined_content),
                            language=chunk.language,
                        )

                    pending_small = []

                merged.append(chunk)

        # Handle trailing small chunks at end of document
        if pending_small:
            if merged:
                prev = merged[-1]
                combined_content = prev.content
                for small_chunk in pending_small:
                    combined_content += "\n\n" + small_chunk.content
                    merge_count += 1

                merged[-1] = TextChunk(
                    chunk_id=prev.chunk_id,
                    document_id=prev.document_id,
                    content=combined_content,
                    section_path=prev.section_path,
                    section_id=prev.section_id,
                    section_level=prev.section_level,
                    chunk_index=prev.chunk_index,
                    char_count=len(combined_content),
                    language=prev.language,
                )
            else:
                # Edge case: entire document has only small chunks — combine into one
                combined_content = "\n\n".join(s.content for s in pending_small)
                first = pending_small[0]
                merged.append(TextChunk(
                    chunk_id=first.chunk_id,
                    document_id=first.document_id,
                    content=combined_content,
                    section_path=first.section_path,
                    section_id=first.section_id,
                    section_level=first.section_level,
                    chunk_index=first.chunk_index,
                    char_count=len(combined_content),
                    language=first.language,
                ))
                merge_count += len(pending_small) - 1

        for chunk_idx, chunk in enumerate(merged):
            chunk.chunk_index = chunk_idx
            chunk.chunk_id = f"{document_id}_text_{chunk.section_id}_{chunk_idx}"

        if merge_count > 0:
            logger.info(f"Merged {merge_count} undersized chunks into {len(merged)} larger chunks")

        return merged

    def _chunk_text(
        self,
        document_id: str,
        text: str,
        language: Optional[str] = None,
    ) -> List[TextChunk]:
        """Split text into section-aware chunks using LangChain splitters."""
        if not text.strip():
            return []

        header_sections = self.header_splitter.split_text(text)

        pre_merge_chunks = []
        chunk_index_counter = 0

        for header_section in header_sections:
            content = header_section.page_content
            metadata = header_section.metadata  # {"h1": "Title", "h2": "Subtitle", ...}

            section_path = []
            section_level = 0
            for key in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                if key in metadata:
                    section_path.append(metadata[key])
                    section_level = int(key[1])

            section_id = self._slugify(section_path[-1]) if section_path else "root"

            sub_contents = (
                self.text_splitter.split_text(content)
                if len(content) > self.chunk_size
                else [content]
            )

            for sub_content in sub_contents:
                sub_content = self._clean_fragment_start(sub_content)

                if not sub_content:
                    continue

                chunk = TextChunk(
                    chunk_id=f"{document_id}_text_{section_id}_{chunk_index_counter}",
                    document_id=document_id,
                    content=sub_content,
                    section_path=section_path,
                    section_id=section_id,
                    section_level=section_level,
                    chunk_index=chunk_index_counter,
                    char_count=len(sub_content),
                    language=language,
                )
                pre_merge_chunks.append(chunk)
                chunk_index_counter += 1

        return self._merge_small_chunks(pre_merge_chunks, document_id)

    def _chunk_tables(
        self,
        document_id: str,
        tables: List[TableAsset],
    ) -> List[TableChunk]:
        """Wrap each table as a single atomic chunk."""
        table_chunks = []

        for table_idx, table in enumerate(tables):
            row_count, col_count = self._count_table_dimensions(table.markdown_content)

            table_chunks.append(TableChunk(
                chunk_id=f"{document_id}_table_{table.page:03d}_{table_idx}",
                document_id=document_id,
                content=table.markdown_content,
                description=table.description or "",
                source_type=table.source_type,
                page=table.page,
                row_count=row_count,
                column_count=col_count,
                char_count=len(table.markdown_content),
            ))

        return table_chunks

    def _chunk_images(
        self,
        document_id: str,
        images: List[ImageAsset],
    ) -> List[ImageChunk]:
        """Wrap each image as a single chunk with description + entities."""
        image_chunks = []

        for image_idx, image in enumerate(images):
            description = image.description or ""
            entities = image.entities or []

            # Build embeddable content: description + entity names for better retrieval
            embeddable_content = description
            if entities:
                entity_names = [e.get('name', '') for e in entities]
                embeddable_content += f"\n\nEntities: {', '.join(entity_names)}"

            if not embeddable_content.strip():
                logger.warning(f"Skipping image on page {image.page} (index {image_idx}): no description")
                continue

            image_chunks.append(ImageChunk(
                chunk_id=f"{document_id}_img_{image.page:03d}_{image_idx}",
                document_id=document_id,
                content=embeddable_content.strip(),
                entities=entities,
                image_path=str(image.path),
                page=image.page,
                width=image.width,
                height=image.height,
                char_count=len(embeddable_content.strip()),
            ))

        return image_chunks

    # ─── Utilities ────────────────────────────────────────────────────────────

    @staticmethod
    def _slugify(text: str) -> str:
        """Create URL-friendly slug from text."""
        slug = re.sub(r'[^\w\s-]', '', text.lower())
        slug = re.sub(r'[-\s]+', '-', slug)
        return slug[:50]

    @staticmethod
    def _count_table_dimensions(markdown_table: str) -> Tuple[int, int]:
        """Count rows and columns in a markdown table."""
        table_rows = [
            row.strip() for row in markdown_table.strip().split('\n')
            if row.strip() and row.strip().startswith('|')
        ]

        if not table_rows:
            return 0, 0

        col_count = max(table_rows[0].count('|') - 1, 0)

        non_separator_rows = [
            row for row in table_rows
            if not re.match(r'^\|[\s\-:|]+\|$', row)
        ]
        row_count = max(len(non_separator_rows) - 1, 0)  # Subtract header row

        return row_count, col_count

    def save_chunks(
        self,
        result: ChunkingResult,
        output_dir: Path,
    ) -> Path:
        """Save chunking result to JSON file. Returns the path to the saved file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{result.document_id}_chunks.json"

        output_path.write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
            encoding='utf-8',
        )

        logger.info(f"Saved {result.total_chunks} chunks to {output_path}")
        return output_path
