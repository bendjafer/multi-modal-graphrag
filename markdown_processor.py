"""Markdown preprocessing and cleaning for GraphRAG."""

import html
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import ftfy
from markdowncleaner import MarkdownCleaner, CleanerOptions

from config import (
    MIN_LINE_LENGTH, MIN_SECTION_CONTENT_WORDS,
    REMOVE_CITATION_NUMBERS, REMOVE_PHOTO_CREDITS
)

logger = logging.getLogger(__name__)


class MarkdownProcessor:
    """Process and clean Docling markdown output for GraphRAG."""

    HEADER_PADDING = "___HEADER_PROTECTED___"
    HEADER_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$')

    # Academic citation patterns — covers inline refs, DOIs, bibliography entries
    CITATION_PATTERNS = [
        re.compile(r'^\[\d+\]|\(\d+\)'),                                           # [1] or (1)
        re.compile(r'\b[A-Z][a-z]+ et al\., \d{4}|\(\w+, \d{4}\)'),               # Smith et al., 2021
        re.compile(r'\bdoi:\S+|http[s]?://\S+'),                                    # DOI/URL references
        re.compile(r'^\[\d+(,\s*\d+)*\]|\(\d+(,\s*\d+)*\)'),                      # [1,2,3] or (1,2,3)
        re.compile(r'[A-Za-z\s]+, \d{1,4}\([\d\-]+\):\d+\-\d+'),                  # Journal Vol(Issue):Pages
        re.compile(r'\[\d+(?:[-–]\d+)?\]'),                                        # [1-3] ranges
        re.compile(                                                                 # Footnote bibliography entries
            r'^\d{1,3}\s+[A-Z][^.\n]+,[^.\n]+,.*?\d{4}\.?\s*$',
            re.MULTILINE
        ),
    ]

    PHOTO_CREDIT_PATTERN = re.compile(
        r'^(?:[A-Z][a-z]+ [A-Z][a-z]+|[A-Z][a-z]+) - (?:Unsplash|[Ii]stockphoto|Bigstock).*$'
    )

    # Single words (e.g. city names) and bare numbers that appear as isolated lines
    ARTIFACT_PATTERN = re.compile(r'^[A-Za-z]{3,20}$')
    NUMBER_ARTIFACT_PATTERN = re.compile(r'^[\d,]+$')

    TABLE_ROW_PATTERN = re.compile(r'^\|.*\|$')

    def __init__(self, min_line_length: Optional[int] = None,
                 min_section_words: Optional[int] = None,
                 remove_citations: Optional[bool] = None,
                 remove_photo_credits: Optional[bool] = None):
        self.min_line_length = min_line_length or MIN_LINE_LENGTH
        self.min_section_words = min_section_words or MIN_SECTION_CONTENT_WORDS
        self.remove_citations = remove_citations if remove_citations is not None else REMOVE_CITATION_NUMBERS
        self.remove_photo_credits = remove_photo_credits if remove_photo_credits is not None else REMOVE_PHOTO_CREDITS

        self.options = CleanerOptions()
        self.options.remove_short_lines = True
        self.options.min_line_length = self.min_line_length
        self.options.remove_duplicate_headlines = True
        self.options.remove_footnotes_in_text = True
        self.options.contract_empty_lines = True
        self.options.remove_sections = False
        self.options.remove_whole_lines = True

        self.cleaner = MarkdownCleaner(options=self.options)

    def pipeline(self, raw_markdown_path: Optional[Path] = None,
                 raw_markdown_text: Optional[str] = None,
                 save_cleaned: bool = False) -> Tuple[str, Dict]:
        """Complete processing pipeline. Returns (cleaned_text, stats).

        Accepts raw text directly (raw_markdown_text) to avoid a disk roundtrip
        when the content is already in memory, or a file path for standalone use.
        """
        stats = {
            'original_length': 0, 'cleaned_length': 0, 'chars_removed': 0,
            'headers_removed': 0, 'citations_removed': 0,
            'artifacts_removed': 0, 'tables_removed': 0, 'language': 'en'
        }

        # Issue #3: Accept content directly from memory to avoid write-then-read roundtrip
        if raw_markdown_text is None:
            if raw_markdown_path is None or not raw_markdown_path.exists():
                raise FileNotFoundError(f"Markdown file not found: {raw_markdown_path}")
            raw_markdown_text = raw_markdown_path.read_text(encoding='utf-8')

        stats['original_length'] = len(raw_markdown_text)

        text = html.unescape(ftfy.fix_text(raw_markdown_text))

        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
        text = re.sub(r'TITLE\s+([^\n]+)', r'## \1', text)

        text, protected_headers = self._protect_headers(text)
        text = self.cleaner.clean_markdown_string(text)
        text = self._restore_headers(text, protected_headers)

        if self.remove_citations:
            text, citations_count = self._remove_inline_citations(text)
            stats['citations_removed'] = citations_count

        if self.remove_photo_credits:
            text = self._remove_photo_credits(text)

        text, tables_removed = self._strip_inline_tables(text)
        stats['tables_removed'] = tables_removed

        text, artifacts_count = self._remove_isolated_artifacts(text)
        stats['artifacts_removed'] = artifacts_count

        text, headers_removed = self._remove_empty_sections(text)
        stats['headers_removed'] = headers_removed

        text = self._normalize_whitespace(text)

        cleaned = text.strip() + '\n'
        stats['cleaned_length'] = len(cleaned)
        stats['chars_removed'] = stats['original_length'] - stats['cleaned_length']

        if save_cleaned:
            if raw_markdown_path is None:
                logger.warning("save_cleaned=True but no raw_markdown_path provided — skipping save")
            else:
                cleaned_path = raw_markdown_path.parent / f"{raw_markdown_path.stem.replace('_raw', '_cleaned')}{raw_markdown_path.suffix}"
                cleaned_path.write_text(cleaned, encoding='utf-8')
                logger.info(f"Cleaned markdown saved: {cleaned_path}")

        logger.info(
            f"Processed: {stats['original_length']} → {stats['cleaned_length']} chars "
            f"(-{stats['chars_removed']}, -{stats['headers_removed']} empty headers, "
            f"-{stats['citations_removed']} citations, "
            f"-{stats['tables_removed']} tables, "
            f"-{stats['artifacts_removed']} artifacts)"
        )

        return cleaned, stats

    def _protect_headers(self, text: str) -> tuple[str, dict]:
        """Add padding to headers to prevent removal as short lines."""
        protected_headers = {}
        result_lines = []

        for line in text.split('\n'):
            if match := self.HEADER_PATTERN.match(line):
                padded = f"{match.group(1)} {match.group(2)} {self.HEADER_PADDING}"
                protected_headers[padded] = line
                result_lines.append(padded)
            else:
                result_lines.append(line)

        return '\n'.join(result_lines), protected_headers

    def _restore_headers(self, text: str, protected_headers: dict) -> str:
        """Remove padding from protected headers."""
        for padded, original in protected_headers.items():
            text = text.replace(padded, original)
        return text.replace(self.HEADER_PADDING, '')

    def _remove_empty_sections(self, text: str) -> Tuple[str, int]:
        """Remove headers with no meaningful content below them. Returns (text, count_removed)."""
        all_lines = text.split('\n')
        result_lines = []
        headers_removed = 0
        i = 0

        while i < len(all_lines):
            line = all_lines[i]
            if match := self.HEADER_PATTERN.match(line):
                j = i + 1
                while j < len(all_lines) and not all_lines[j].strip():
                    j += 1

                if j >= len(all_lines) or self.HEADER_PATTERN.match(all_lines[j]):
                    headers_removed += 1
                    logger.debug(f"Removed empty header: {line}")
                else:
                    # Count words in section body to determine if it has meaningful content
                    section_words = []
                    k = j
                    while k < len(all_lines) and not self.HEADER_PATTERN.match(all_lines[k]):
                        section_words.extend(all_lines[k].split())
                        k += 1

                    if len(section_words) >= self.min_section_words:
                        result_lines.append(line)
                    else:
                        headers_removed += 1
                        logger.debug(f"Removed header with minimal content: {line}")
                        i = k - 1
            else:
                result_lines.append(line)
            i += 1

        return '\n'.join(result_lines), headers_removed

    def _remove_inline_citations(self, text: str) -> Tuple[str, int]:
        """Remove academic citation patterns. Returns (text, count_removed)."""
        citations_removed = 0
        for pattern in self.CITATION_PATTERNS:
            citation_matches = list(pattern.finditer(text))
            for match in reversed(citation_matches):
                text = text[:match.start()] + text[match.end():]
                citations_removed += 1
        return text, citations_removed

    def _remove_photo_credits(self, text: str) -> str:
        """Remove photo credit lines like 'Jana Shnipelson - Unsplash'."""
        all_lines = text.split('\n')
        filtered_lines = [
            line for line in all_lines
            if not self.PHOTO_CREDIT_PATTERN.match(line.strip())
        ]
        return '\n'.join(filtered_lines)

    def _remove_isolated_artifacts(self, text: str) -> Tuple[str, int]:
        """Remove isolated short lines that are likely OCR artifacts. Returns (text, count_removed)."""
        all_lines = text.split('\n')
        filtered_lines = []
        artifacts_removed = 0

        for i, line in enumerate(all_lines):
            stripped_line = line.strip()

            if not stripped_line:
                filtered_lines.append(line)
                continue

            if self.HEADER_PATTERN.match(line):
                filtered_lines.append(line)
                continue

            is_isolated_artifact = (
                (self.ARTIFACT_PATTERN.match(stripped_line) or
                 self.NUMBER_ARTIFACT_PATTERN.match(stripped_line)) and
                len(stripped_line) < 30
            )

            is_preceded_by_blank = (i == 0 or not all_lines[i - 1].strip())
            is_followed_by_blank = (i == len(all_lines) - 1 or not all_lines[i + 1].strip())

            if is_isolated_artifact and is_preceded_by_blank and is_followed_by_blank:
                artifacts_removed += 1
                logger.debug(f"Removed artifact: {stripped_line}")
            else:
                filtered_lines.append(line)

        return '\n'.join(filtered_lines), artifacts_removed

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize all whitespace: multiple spaces, tabs, non-breaking spaces."""
        text = re.sub(r' {2,}', ' ', text)
        text = text.replace('\t', ' ')
        text = text.replace('\xa0', ' ')
        text = re.sub(r' +\n', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text

    def _strip_inline_tables(self, text: str) -> Tuple[str, int]:
        """Remove markdown table blocks from text (they are captured as TableChunks)."""
        all_lines = text.split('\n')
        filtered_lines = []
        in_table = False
        tables_removed_count = 0
        current_table_block: List[str] = []

        for line in all_lines:
            stripped = line.strip()
            is_table_row = bool(stripped and self.TABLE_ROW_PATTERN.match(stripped))

            if is_table_row and not in_table:
                in_table = True
                current_table_block = [line]
            elif is_table_row and in_table:
                current_table_block.append(line)
            elif not is_table_row and in_table:
                in_table = False
                if len(current_table_block) >= 3:
                    filtered_lines.append('\n[Table omitted — captured separately]\n')
                    tables_removed_count += 1
                else:
                    filtered_lines.extend(current_table_block)
                current_table_block = []
                filtered_lines.append(line)
            else:
                filtered_lines.append(line)

        if in_table:
            if len(current_table_block) >= 3:
                filtered_lines.append('\n[Table omitted — captured separately]\n')
                tables_removed_count += 1
            else:
                filtered_lines.extend(current_table_block)

        return '\n'.join(filtered_lines), tables_removed_count
