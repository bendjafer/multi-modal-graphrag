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


# Regex to detect markdown table rows: lines starting and ending with |
_TABLE_ROW_PATTERN = re.compile(r'^\|.*\|$')
_TABLE_SEPARATOR_PATTERN = re.compile(r'^\|[\s\-:|]+\|$')


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
    
    # Headers to split on — preserves full hierarchy
    HEADERS_TO_SPLIT = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
        ("####", "h4"),
        ("#####", "h5"),
        ("######", "h6"),
    ]
    
    # Pattern to strip section metadata comments injected by MarkdownProcessor
    SECTION_META_PATTERN = re.compile(
        r'<!-- section:id=\S+ level=\d+ num=\d+ -->\n?'
    )
    
    # Pattern to strip document language comment
    DOC_LANG_PATTERN = re.compile(r'<!-- document:lang=(\w+) -->\n?')
    
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
            strip_headers=False,  # Keep headers in chunk content for context
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
        document_id: str,   # Unique identifier for the document.
        markdown_content: str,  # Cleaned markdown text (from MarkdownProcessor).
        tables: Optional[List[TableAsset]] = None,  # List of extracted TableAsset objects.
        images: Optional[List[ImageAsset]] = None,  # List of extracted ImageAsset objects.
        language: Optional[str] = None,  # Detected language code (e.g., "en").
    ) -> ChunkingResult:
        """Chunk a complete document into text, table, and image chunks, 
        Returns:
            ChunkingResult with all chunks and statistics.
        """
        tables = tables or []
        images = images or []
        
        # --- Extract document language if present ---
        if not language:
            lang_match = self.DOC_LANG_PATTERN.search(markdown_content)
            if lang_match:
                language = lang_match.group(1)
        
        # --- Clean metadata comments before chunking ---
        clean_text = self._strip_metadata(markdown_content)
        
        # --- Remove the "Extracted Figures" appendix ---
        # This section is auto-appended by MarkdownProcessor and 
        # duplicates image data — images are already handled as ImageChunks
        clean_text = self._remove_extracted_figures_section(clean_text)
        
        # --- FIX #1: Strip inline markdown tables from text ---
        # Tables are already captured as TableChunks; leaving them in
        # the text creates duplicate embeddings
        clean_text = self._strip_inline_tables(clean_text)
        
        # --- Text chunking ---
        text_chunks = self._chunk_text(
            document_id=document_id,
            text=clean_text,
            language=language,
        )
        
        # --- Table chunking ---
        table_chunks = self._chunk_tables(
            document_id=document_id,
            tables=tables,
        )
        
        # --- Image chunking ---
        image_chunks = self._chunk_images(
            document_id=document_id,
            images=images,
        )
        
        # --- Stats ---
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
    
    def _strip_metadata(self, text: str) -> str:
        """Remove section/document metadata comments from markdown."""
        text = self.SECTION_META_PATTERN.sub('', text)
        text = self.DOC_LANG_PATTERN.sub('', text)
        return text.strip()
    
    def _remove_extracted_figures_section(self, text: str) -> str:
        """Remove the auto-appended 'Extracted Figures' section.
        
        This section is added by MarkdownProcessor._add_image_descriptions()
        and duplicates info that will be captured as ImageChunks.
        """
        # Find the separator + heading
        marker = "\n---\n\n## Extracted Figures"
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx].rstrip()
        return text
    
    def _strip_inline_tables(self, text: str) -> str:
        """Remove markdown table blocks from text content.

        FIX #1: Tables embedded inline in the markdown are already captured
        independently as TableChunks via the TableAsset pipeline. Leaving
        them in the text creates duplicate embeddings and wastes vector DB
        storage. This method strips contiguous blocks of | ... | rows.

        IMPROVEMENTS:
        - More robust detection of table boundaries
        - Better handling of tables immediately followed by content
        - Only add placeholder for substantial tables (3+ rows)
        - Cleaner whitespace handling around placeholders
        """
        lines = text.split('\n')
        result = []
        in_table = False
        table_lines_count = 0

        for i, line in enumerate(lines):
            stripped = line.strip()
            is_table_row = bool(
                stripped and _TABLE_ROW_PATTERN.match(stripped)
            )

            if is_table_row and not in_table:
                # Starting a table block
                in_table = True
                table_lines_count = 1
                # Don't append — skip table rows
            elif is_table_row and in_table:
                # Continuing a table block — skip
                table_lines_count += 1
            elif not is_table_row and in_table:
                # Table ended
                in_table = False
                # Only add placeholder if table had at least 3 rows (header + separator + data)
                if table_lines_count >= 3:
                    # Add placeholder on its own line, preserve spacing
                    if result and result[-1].strip():  # Previous line has content
                        result.append('')
                    result.append('[Table omitted — see table chunks]')
                    if stripped:  # Current line has content
                        result.append('')
                # Add the current non-table line
                if stripped or not in_table:  # Keep non-empty lines or spacing after tables
                    result.append(line)
                table_lines_count = 0
            else:
                # Regular content
                result.append(line)

        # Handle table at end of text
        if in_table and table_lines_count >= 3:
            if result and result[-1].strip():
                result.append('')
            result.append('[Table omitted — see table chunks]')

        # Clean up excessive blank lines around placeholders
        cleaned = '\n'.join(result)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)  # Max 2 consecutive newlines
        return cleaned


        # ============================================================================
        # IMPROVEMENT #2: Enhanced _clean_fragment_start()
        # ============================================================================

    def _clean_fragment_start(self, content: str) -> str:
        """FIX #2: Clean orphan overlap fragments.

        When RecursiveCharacterTextSplitter splits mid-sentence, overlap
        chunks can start with '. ', ', ', or lowercase fragments that
        lost their preceding context. This trims leading noise.

        IMPROVEMENTS:
        - Better detection of sentence fragments
        - Preserve valid lowercase starts (e.g., "iPhone", "eBay", proper nouns)
        - Handle multiple leading punctuation marks
        - Discard very short fragments (<5 words)
        - Log substantial content removal for debugging
        """
        original_length = len(content)

        # Strip leading sentence-ending punctuation + space (can be multiple)
        content = re.sub(r'^[.,;:!?)\]\}]+\s*', '', content)

        # If we removed something and now content is too short, it was probably junk
        if len(content) < 10:
            return ""

        # Check if starts with lowercase AND isn't a known valid lowercase start
        if content and content[0].islower():
            # Common valid lowercase starts to preserve
            valid_lowercase_starts = [
                'iPhone', 'iOS', 'eBay', 'iPad', 'macOS', 'npm', 'etc',
                'iPhone', 'eToro', 'eCommerce'
            ]
            starts_with_valid = any(content.startswith(term) for term in valid_lowercase_starts)

            if not starts_with_valid:
                # Find the first sentence boundary (. ! ? followed by capital)
                match = re.search(r'[.!?]\s+[A-Z]', content)
                if match:
                    # Start from the capital letter after the sentence boundary
                    content = content[match.end() - 1:]
                else:
                    # No clear sentence boundary found - check if it's a very short fragment
                    if len(content.split()) < 5:  # Less than 5 words
                        # Likely a fragment, discard it
                        logger.debug(f"Discarded short lowercase fragment: {content[:50]}...")
                        return ""

        # Final cleanup
        content = content.strip()

        # Log if we removed substantial content (might indicate a problem)
        if original_length > 50 and len(content) / original_length < 0.5:
            logger.debug(f"Cleaned fragment removed {original_length - len(content)} chars")

        return content
        
    # ============================================================================
    # IMPROVEMENT #3: Enhanced _merge_small_chunks()
    # ============================================================================

    def _merge_small_chunks(
        self,
        chunks: List[TextChunk],
        document_id: str,
    ) -> List[TextChunk]:
        """FIX #3: Merge undersized chunks with their neighbors.

        Chunks below min_chunk_size are merged into the previous chunk
        (preferred) or the next chunk. This eliminates noisy micro-chunks
        that produce weak embeddings.

        IMPROVEMENTS:
        - Better handling of multiple consecutive small chunks
        - Preserve section boundaries when merging (prefer same-section merges)
        - Queue small chunks and merge intelligently
        - Add statistics logging
        - Handle edge case of document with only small chunks
        """
        if not chunks:
            return chunks

        merged = []
        pending_small = []  # Queue of small chunks to merge
        merge_count = 0

        for chunk in chunks:
            if chunk.char_count < self.min_chunk_size:
                # This chunk is too small — queue it
                pending_small.append(chunk)
            else:
                # Normal-sized chunk
                if pending_small:
                    # Decide where to merge pending small chunks
                    # Prefer merging into previous chunk if same section
                    if merged and merged[-1].section_id == pending_small[0].section_id:
                        # Merge all pending into previous
                        prev = merged[-1]
                        combined_content = prev.content
                        for small in pending_small:
                            combined_content += "\n\n" + small.content
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
                        # Merge pending into current chunk (prepend)
                        combined_content = ""
                        for small in pending_small:
                            combined_content += small.content + "\n\n"
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

        # Handle trailing small chunks
        if pending_small:
            if merged:
                # Merge into last chunk
                prev = merged[-1]
                combined_content = prev.content
                for small in pending_small:
                    combined_content += "\n\n" + small.content
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
                # Only small chunks exist — combine them all into one
                if pending_small:
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

        # Re-index chunk_ids after merging
        for i, chunk in enumerate(merged):
            chunk.chunk_index = i
            chunk.chunk_id = f"{document_id}_text_{chunk.section_id}_{i}"

        if merge_count > 0:
            logger.info(f"✨ Merged {merge_count} undersized chunks into {len(merged)} larger chunks")

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
        
        # Pass 1: Split by markdown headers
        header_docs = self.header_splitter.split_text(text)
        
        # Pass 2: Sub-split large sections
        raw_chunks = []
        global_index = 0
        
        for doc in header_docs:
            content = doc.page_content
            metadata = doc.metadata  # {"h1": "Title", "h2": "Subtitle", ...}
            
            # Build section path from header hierarchy
            section_path = []
            section_level = 0
            for key in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                if key in metadata:
                    section_path.append(metadata[key])
                    section_level = int(key[1])
            
            # Derive section_id from deepest header
            section_id = self._slugify(section_path[-1]) if section_path else "root"
            
            # Sub-split if content exceeds chunk_size
            if len(content) > self.chunk_size:
                sub_docs = self.text_splitter.split_text(content)
            else:
                sub_docs = [content]
            
            for sub_idx, sub_content in enumerate(sub_docs):
                # FIX #2: Clean orphan fragment starts from overlap splits
                sub_content = self._clean_fragment_start(sub_content)
                
                if not sub_content:
                    continue
                
                chunk = TextChunk(
                    chunk_id=f"{document_id}_text_{section_id}_{global_index}",
                    document_id=document_id,
                    content=sub_content,
                    section_path=section_path,
                    section_id=section_id,
                    section_level=section_level,
                    chunk_index=global_index,
                    char_count=len(sub_content),
                    language=language,
                )
                raw_chunks.append(chunk)
                global_index += 1
        
        # FIX #3: Merge small chunks with neighbors
        merged_chunks = self._merge_small_chunks(raw_chunks, document_id)
        
        return merged_chunks
    
    def _chunk_tables(
        self,
        document_id: str,
        tables: List[TableAsset],
    ) -> List[TableChunk]:
        """Wrap each table as a single atomic chunk."""
        chunks = []
        
        for idx, table in enumerate(tables):
            # Count rows and columns from markdown
            row_count, col_count = self._count_table_dimensions(
                table.markdown_content
            )
            
            # Build embeddable content: description + table markdown
            content = table.markdown_content
            if table.description:
                content = f"{table.description}\n\n{content}"
            
            chunk = TableChunk(
                chunk_id=f"{document_id}_table_{table.page:03d}_{idx}",
                document_id=document_id,
                content=content,
                description=table.description or "",
                source_type=table.source_type,
                page=table.page,
                row_count=row_count,
                column_count=col_count,
                char_count=len(content),
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_images(
        self,
        document_id: str,
        images: List[ImageAsset],
    ) -> List[ImageChunk]:
        """Wrap each image as a single chunk with description + entities."""
        chunks = []
        
        for idx, image in enumerate(images):
            description = image.description or ""
            entities = image.entities or []
            
            # Build embeddable content: description + entity names
            content = description
            if entities:
                entity_names = [e.get('name', '') for e in entities]
                content += f"\n\nEntities: {', '.join(entity_names)}"
            
            # Skip images with no description
            if not content.strip():
                logger.warning(
                    f"Skipping image page {image.page} idx {idx}: no description"
                )
                continue
            
            chunk = ImageChunk(
                chunk_id=f"{document_id}_img_{image.page:03d}_{idx}",
                document_id=document_id,
                content=content.strip(),
                entities=entities,
                image_path=str(image.path),
                page=image.page,
                width=image.width,
                height=image.height,
                char_count=len(content.strip()),
            )
            chunks.append(chunk)
        
        return chunks
    
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
        lines = [
            l.strip() for l in markdown_table.strip().split('\n')
            if l.strip() and l.strip().startswith('|')
        ]
        
        if not lines:
            return 0, 0
        
        # Count columns from first row
        col_count = lines[0].count('|') - 1  # Leading/trailing pipes
        col_count = max(col_count, 0)
        
        # Count rows (exclude header separator line)
        data_rows = [
            l for l in lines
            if not re.match(r'^\|[\s\-:|]+\|$', l)
        ]
        row_count = max(len(data_rows) - 1, 0)  # Subtract header row
        
        return row_count, col_count
    
    def save_chunks(
        self,
        result: ChunkingResult,
        output_dir: Path,
    ) -> Path:
        """Save chunking result to JSON file.
        
        Returns path to saved file.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{result.document_id}_chunks.json"
        
        output_path.write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
            encoding='utf-8',
        )
        
        logger.info(f"Saved {result.total_chunks} chunks to {output_path}")
        return output_path
