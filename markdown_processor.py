"""Markdown preprocessing and cleaning for GraphRAG."""

import html
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional

try:
    import ftfy
    HAS_FTFY = True
except ImportError:
    HAS_FTFY = False
    logging.warning("ftfy not installed. Install with: pip install ftfy")

from markdowncleaner import MarkdownCleaner, CleanerOptions

logger = logging.getLogger(__name__)


class MarkdownProcessor:
    """Process and clean Docling markdown output for GraphRAG."""
    
    MIN_LINE_LENGTH = 15
    HEADER_PADDING = "___HEADER_PROTECTED___"
    HEADER_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$')
    
    def __init__(self):
        self.stats = {'original_length': 0, 'cleaned_length': 0, 'chars_removed': 0, 
                      'images_added': 0, 'headers_removed': 0}
        
        self.options = CleanerOptions()
        self.options.remove_short_lines = True
        self.options.min_line_length = self.MIN_LINE_LENGTH
        self.options.remove_duplicate_headlines = True
        self.options.remove_footnotes_in_text = True
        self.options.contract_empty_lines = True
        self.options.remove_sections = False
        self.options.remove_whole_lines = True
        
        self.cleaner = MarkdownCleaner(options=self.options)
    
    def pipeline(self, raw_markdown_path: Path, image_metadata: Optional[List[Dict]] = None,
                 save_cleaned: bool = False) -> str:
        """Complete processing pipeline."""
        if not raw_markdown_path.exists():
            raise FileNotFoundError(f"Markdown file not found: {raw_markdown_path}")
        
        raw_markdown = raw_markdown_path.read_text(encoding='utf-8')
        self.stats['original_length'] = len(raw_markdown)
        
        # HTML entities + encoding fixes
        text = html.unescape(ftfy.fix_text(raw_markdown) if HAS_FTFY else raw_markdown)
        text = re.sub(r' {2,}', ' ', text).replace(' \n', '\n')
        
        # Remove HTML comments and convert TITLE markers
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
        text = re.sub(r'TITLE\s+([^\n]+)', r'## \1', text)
        
        # Protect headers, clean, restore, and remove empty sections
        text, protected = self._protect_headers(text)
        text = self.cleaner.clean_markdown_string(text)
        text = self._restore_headers(text, protected)
        text = self._remove_empty_sections(text)
        
        # Add images and normalize
        if image_metadata:
            text = self._add_image_descriptions(text, image_metadata)
        
        cleaned = re.sub(r'\n{3,}', '\n\n', text).strip() + '\n'
        
        self.stats['cleaned_length'] = len(cleaned)
        self.stats['chars_removed'] = self.stats['original_length'] - self.stats['cleaned_length']
        
        if save_cleaned:
            cleaned_path = raw_markdown_path.parent / f"{raw_markdown_path.stem.replace('_raw', '_cleaned')}{raw_markdown_path.suffix}"
            cleaned_path.write_text(cleaned, encoding='utf-8')
            logger.info(f"Cleaned markdown saved: {cleaned_path}")
        
        logger.info(f"Processed: {self.stats['original_length']} → {self.stats['cleaned_length']} chars "
                   f"(-{self.stats['chars_removed']}, +{self.stats['images_added']} images, "
                   f"-{self.stats['headers_removed']} empty headers)")
        
        return cleaned
    
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
    
    def _remove_empty_sections(self, text: str) -> str:
        """Remove headers with no content below them."""
        lines = text.split('\n')
        result = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            if self.HEADER_PATTERN.match(line):
                # Find next non-empty line
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                
                # Keep header if next non-empty line is not another header
                if j < len(lines) and lines[j].strip() and not self.HEADER_PATTERN.match(lines[j]):
                    result.append(line)
                else:
                    self.stats['headers_removed'] += 1
                    logger.debug(f"Removed empty header: {line}")
            else:
                result.append(line)
            i += 1
        
        return '\n'.join(result)
    
    def _add_image_descriptions(self, text: str, image_metadata: List[Dict]) -> str:
        """Add AI-generated image descriptions to markdown."""
        if not image_metadata:
            return text
        
        # Group by page
        page_images = {}
        for img in image_metadata:
            page_images.setdefault(img.get('page', 0), []).append(img)
        
        sections = []
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
                    self.stats['images_added'] += 1
        
        if sections:
            text += "\n\n---\n\n## Extracted Figures\n" + "".join(sections)
        
        return text
    
    def get_stats(self) -> Dict:
        """Return processing statistics."""
        return self.stats.copy()
