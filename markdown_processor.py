"""Markdown preprocessing and cleaning for GraphRAG."""

import html
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import ftfy
from markdowncleaner import MarkdownCleaner, CleanerOptions

try:
    from langdetect import detect, LangDetectException
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False
    logging.warning("langdetect not installed. Language detection disabled. Install with: pip install langdetect")

from config import (
    MIN_LINE_LENGTH, MIN_SECTION_CONTENT_WORDS,
    REMOVE_CITATION_NUMBERS, REMOVE_PHOTO_CREDITS
)

logger = logging.getLogger(__name__)


class MarkdownProcessor:
    """Process and clean Docling markdown output for GraphRAG."""
    
    HEADER_PADDING = "___HEADER_PROTECTED___"
    HEADER_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$')
    
    # Citation patterns for academic references
    CITATION_PATTERNS = [
        re.compile(r'^\[\d+\]|\(\d+\)'),                                           # [1] or (1)
        re.compile(r'\b[A-Z][a-z]+ et al\., \d{4}|\(\w+, \d{4}\)'),               # Smith et al., 2021
        re.compile(r'\bdoi:\S+|http[s]?://\S+'),                                    # DOI/URL references
        re.compile(r'^\[\d+(,\s*\d+)*\]|\(\d+(,\s*\d+)*\)'),                      # [1,2,3] or (1,2,3)
        re.compile(r'[A-Za-z\s]+, \d{1,4}\([\d\-]+\):\d+\-\d+'),                  # Journal Vol(Issue):Pages
        re.compile(r'\[\d+(?:[-–]\d+)?\]'),                                        # [1-3] ranges
        re.compile(                                                                 # Footnote bibliography entries:
            r'^\d{1,3}\s+[A-Z][^.\n]+,[^.\n]+,.*?\d{4}\.?\s*$',                   # "18 Author, 'Title', Source, 2021."
            re.MULTILINE
        ),
    ]
    
    # Photo credit patterns  
    PHOTO_CREDIT_PATTERN = re.compile(
        r'^(?:[A-Z][a-z]+ [A-Z][a-z]+|[A-Z][a-z]+) - (?:Unsplash|[Ii]stockphoto|Bigstock).*$'
    )
    
    # Isolated artifacts (short standalone lines)
    ARTIFACT_PATTERN = re.compile(r'^[A-Za-z]{3,20}$')  # Single words like "Ashgabat", "Ankara"
    NUMBER_ARTIFACT_PATTERN = re.compile(r'^[\d,]+$')  # Standalone numbers
    
    def __init__(self, min_line_length: Optional[int] = None, 
                 min_section_words: Optional[int] = None,
                 remove_citations: Optional[bool] = None,
                 remove_photo_credits: Optional[bool] = None,
                 add_language_metadata: bool = True,
                 add_section_metadata: bool = True):
        """Initialize processor with configurable settings."""
        self.min_line_length = min_line_length or MIN_LINE_LENGTH
        self.min_section_words = min_section_words or MIN_SECTION_CONTENT_WORDS
        self.remove_citations = remove_citations if remove_citations is not None else REMOVE_CITATION_NUMBERS
        self.remove_photo_credits = remove_photo_credits if remove_photo_credits is not None else REMOVE_PHOTO_CREDITS
        self.add_language_metadata = add_language_metadata and HAS_LANGDETECT
        self.add_section_metadata = add_section_metadata
        
        self.options = CleanerOptions()
        self.options.remove_short_lines = True
        self.options.min_line_length = self.min_line_length
        self.options.remove_duplicate_headlines = True
        self.options.remove_footnotes_in_text = True
        self.options.contract_empty_lines = True
        self.options.remove_sections = False
        self.options.remove_whole_lines = True
        
        self.cleaner = MarkdownCleaner(options=self.options)
    
    def pipeline(self, raw_markdown_path: Path, image_metadata: Optional[List[Dict]] = None,
                 save_cleaned: bool = False) -> Tuple[str, Dict]:
        """Complete processing pipeline. Returns (cleaned_text, stats)."""
        if not raw_markdown_path.exists():
            raise FileNotFoundError(f"Markdown file not found: {raw_markdown_path}")
        
        stats = {
            'original_length': 0, 'cleaned_length': 0, 'chars_removed': 0, 
            'images_added': 0, 'headers_removed': 0, 'citations_removed': 0,
            'artifacts_removed': 0, 'language': None  # type: Optional[str]
        }
        
        raw_markdown = raw_markdown_path.read_text(encoding='utf-8')
        stats['original_length'] = len(raw_markdown)
        
        # Step 1: HTML entities + encoding fixes (BEFORE structural cleaning)
        text = html.unescape(ftfy.fix_text(raw_markdown))
        
        # Step 2: Remove HTML comments and convert TITLE markers
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
        text = re.sub(r'TITLE\s+([^\n]+)', r'## \1', text)
        
        # Step 3: Protect headers, run structural cleaning, restore
        text, protected = self._protect_headers(text)
        text = self.cleaner.clean_markdown_string(text)
        text = self._restore_headers(text, protected)
        
        # Step 4: Advanced cleaning (AFTER structural cleanup)
        if self.remove_citations:
            text, citations_count = self._remove_inline_citations(text)
            stats['citations_removed'] = citations_count
        
        if self.remove_photo_credits:
            text = self._remove_photo_credits(text)
        
        text, artifacts_count = self._remove_isolated_artifacts(text)
        stats['artifacts_removed'] = artifacts_count
        
        # Step 5: Remove empty sections
        text, headers_removed = self._remove_empty_sections(text)
        stats['headers_removed'] = headers_removed
        
        # Step 6: Normalize whitespace (AFTER all cleaning)
        text = self._normalize_whitespace(text)
        
        # Step 7: Detect language and add metadata
        language = None
        if self.add_language_metadata:
            language = self._detect_language(text)
            stats['language'] = language
        
        # Step 8: Add section metadata for better chunking
        if self.add_section_metadata:
            text = self._add_section_metadata(text)
        
        # Step 9: Add image descriptions
        if image_metadata:
            text, images_added = self._add_image_descriptions(text, image_metadata)
            stats['images_added'] = images_added
        
        # Step 10: Prepend document metadata
        if language:
            text = f"<!-- document:lang={language} -->\n\n{text}"
        
        cleaned = text.strip() + '\n'
        stats['cleaned_length'] = len(cleaned)
        stats['chars_removed'] = stats['original_length'] - stats['cleaned_length']
        
        if save_cleaned:
            cleaned_path = raw_markdown_path.parent / f"{raw_markdown_path.stem.replace('_raw', '_cleaned')}{raw_markdown_path.suffix}"
            cleaned_path.write_text(cleaned, encoding='utf-8')
            logger.info(f"Cleaned markdown saved: {cleaned_path}")
        
        logger.info(
            f"Processed: {stats['original_length']} → {stats['cleaned_length']} chars "
            f"(-{stats['chars_removed']}, +{stats['images_added']} images, "
            f"-{stats['headers_removed']} empty headers, "
            f"-{stats['citations_removed']} citations, "
            f"-{stats['artifacts_removed']} artifacts)"
        )
        
        return cleaned, stats
    
    def _protect_headers(self, text: str) -> tuple[str, dict]:
        """Add padding to headers to prevent removal as short lines."""
        protected = {}
        result = []
        
        for line in text.split('\n'):
            if match := self.HEADER_PATTERN.match(line):
                padded = f"{match.group(1)} {match.group(2)} {self.HEADER_PADDING}"
                protected[padded] = line
                result.append(padded)
            else:
                result.append(line)
        
        return '\n'.join(result), protected
    
    def _restore_headers(self, text: str, protected: dict) -> str:
        """Remove padding from protected headers."""
        for padded, original in protected.items():
            text = text.replace(padded, original)
        return text.replace(self.HEADER_PADDING, '')
    
    def _remove_empty_sections(self, text: str) -> Tuple[str, int]:
        """Remove headers with no meaningful content below them. Returns (text, count_removed)."""
        lines = text.split('\n')
        result = []
        headers_removed = 0
        i = 0
        
        while i < len(lines):
            line = lines[i]
            if match := self.HEADER_PATTERN.match(line):
                # Find next non-empty line
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                
                # Check if next content is another header or end of document
                if j >= len(lines) or self.HEADER_PATTERN.match(lines[j]):
                    headers_removed += 1
                    logger.debug(f"Removed empty header: {line}")
                else:
                    # Check if section has meaningful content (not just 1-2 words)
                    content_words = []
                    k = j
                    while k < len(lines) and not self.HEADER_PATTERN.match(lines[k]):
                        content_words.extend(lines[k].split())
                        k += 1
                    
                    if len(content_words) >= self.min_section_words:
                        result.append(line)
                    else:
                        headers_removed += 1
                        logger.debug(f"Removed header with minimal content: {line}")
            else:
                result.append(line)
            i += 1
        
        return '\n'.join(result), headers_removed
    
    def _add_image_descriptions(self, text: str, image_metadata: List[Dict]) -> Tuple[str, int]:
        """Add AI-generated image descriptions to markdown. Returns (text, count_added)."""
        if not image_metadata:
            return text, 0
        
        # Group by page
        page_images = {}
        for img in image_metadata:
            page_images.setdefault(img.get('page', 0), []).append(img)
        
        sections = []
        images_added = 0
        
        for page in sorted(page_images.keys()):
            for idx, img in enumerate(page_images[page], 1):
                if description := img.get('description', '').strip():
                    section = f"\n### Figure (Page {page}, Image {idx})\n\n{description}\n"
                    
                    if entities := img.get('entities', []):
                        section += "\n**Entities:**\n"
                        for entity in entities:
                            name = entity.get('name', 'unknown')
                            entity_type = entity.get('type', 'unknown')
                            section += f"- {name} ({entity_type})"
                            if props := entity.get('properties', ''):
                                section += f": {props}"
                            section += "\n"
                    
                    sections.append(section)
                    images_added += 1
        
        if sections:
            text += "\n\n---\n\n## Extracted Figures\n" + "".join(sections)
        
        return text, images_added
    
    def _remove_inline_citations(self, text: str) -> Tuple[str, int]:
        """Remove academic citation patterns. Returns (text, count_removed)."""
        citations_removed = 0
        for pattern in self.CITATION_PATTERNS:
            matches = list(pattern.finditer(text))
            # Replace from end to preserve positions
            for match in reversed(matches):
                text = text[:match.start()] + text[match.end():]
                citations_removed += 1
        return text, citations_removed
    
    def _remove_photo_credits(self, text: str) -> str:
        """Remove photo credit lines like 'Jana Shnipelson - Unsplash'."""
        lines = text.split('\n')
        cleaned_lines = [
            line for line in lines 
            if not self.PHOTO_CREDIT_PATTERN.match(line.strip())
        ]
        return '\n'.join(cleaned_lines)
    
    def _remove_isolated_artifacts(self, text: str) -> Tuple[str, int]:
        """Remove isolated short lines that are likely artifacts. Returns (text, count_removed)."""
        lines = text.split('\n')
        result = []
        artifacts_removed = 0
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Skip empty lines
            if not stripped:
                result.append(line)
                continue
            
            # Skip headers
            if self.HEADER_PATTERN.match(line):
                result.append(line)
                continue
            
            # Check for artifacts: single word (location names) or standalone numbers
            is_artifact = (
                (self.ARTIFACT_PATTERN.match(stripped) or 
                 self.NUMBER_ARTIFACT_PATTERN.match(stripped)) and
                len(stripped) < 30  # Safety: don't remove longer lines
            )
            
            # Only remove if surrounded by empty lines (truly isolated)
            prev_empty = (i == 0 or not lines[i-1].strip())
            next_empty = (i == len(lines)-1 or not lines[i+1].strip())
            
            if is_artifact and prev_empty and next_empty:
                artifacts_removed += 1
                logger.debug(f"Removed artifact: {stripped}")
            else:
                result.append(line)
        
        return '\n'.join(result), artifacts_removed
    
    def _detect_language(self, text: str) -> Optional[str]:
        """Detect document language using langdetect."""
        try:
            # Sample first 1000 chars for detection (faster, usually sufficient)
            sample = text[:1000].replace('\n', ' ').strip()
            if len(sample) < 50:
                # Not enough text for reliable detection
                return None
            lang = detect(sample)
            logger.debug(f"Detected language: {lang}")
            return lang
        except LangDetectException as e:
            logger.warning(f"Language detection failed: {e}")
            return None
    
    def _add_section_metadata(self, text: str) -> str:
        """Add metadata comments to section boundaries for better chunking."""
        lines = text.split('\n')
        result = []
        section_counter = 0
        current_level = 0
        
        for i, line in enumerate(lines):
            if match := self.HEADER_PATTERN.match(line):
                level = len(match.group(1))  # Number of # symbols
                title = match.group(2).strip()
                section_id = self._create_section_id(title)
                section_counter += 1
                
                # Add section metadata comment before header
                metadata = f"<!-- section:id={section_id} level={level} num={section_counter} -->"
                result.append(metadata)
                result.append(line)
                current_level = level
            else:
                result.append(line)
        
        return '\n'.join(result)
    
    def _create_section_id(self, title: str) -> str:
        """Create URL-friendly section ID from title."""
        # Remove special characters, lowercase, replace spaces with hyphens
        section_id = re.sub(r'[^\w\s-]', '', title.lower())
        section_id = re.sub(r'[-\s]+', '-', section_id)
        return section_id[:50]  # Limit length
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize all whitespace: multiple spaces, tabs, non-breaking spaces."""
        # Replace multiple spaces with single space
        text = re.sub(r' {2,}', ' ', text)
        # Replace tabs with space
        text = text.replace('\t', ' ')
        # Replace non-breaking spaces
        text = text.replace('\xa0', ' ')
        # Remove spaces before newlines
        text = re.sub(r' +\n', '\n', text)
        # Contract multiple blank lines to max 2
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text

