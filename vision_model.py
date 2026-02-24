"""AI-powered image filtering with async batch processing."""

import asyncio
import base64
import hashlib
import json
import logging
import re
from typing import Dict, List, Optional
from io import BytesIO
from pathlib import Path

from openai import AsyncOpenAI, RateLimitError, APIConnectionError, APIStatusError
from PIL import Image

try:
    from diskcache import Cache
    HAS_DISKCACHE = True
except ImportError:
    HAS_DISKCACHE = False
    logging.warning("diskcache not installed. Vision API responses won't be cached. Install with: pip install diskcache")

from config import (
    OPENAI_API_KEY, VISION_MODEL, ENABLE_VISION_CACHE, VISION_CACHE_DIR,
    VISION_CACHE_TTL_DAYS, VISION_CACHE_SIZE_LIMIT_GB,
    VISION_MAX_TOKENS_FILTER, VISION_MAX_TOKENS_TABLE, VISION_TEMPERATURE,
)

logger = logging.getLogger(__name__)

# Cache TTL in seconds (0 = no expiry)
_CACHE_TTL_SECONDS = VISION_CACHE_TTL_DAYS * 86_400 if VISION_CACHE_TTL_DAYS > 0 else None
_CACHE_SIZE_LIMIT = VISION_CACHE_SIZE_LIMIT_GB * (2 ** 30)


class VisionModel:
    """Async vision processor with concurrent batch processing."""

    def __init__(self, model: str = None, api_key: Optional[str] = None,
                 max_retries: int = 5, base_delay: float = 1.0, max_concurrent: int = 10):
        self.model = model or VISION_MODEL
        self.client = AsyncOpenAI(api_key=api_key or OPENAI_API_KEY)
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_concurrent = max_concurrent
        # Issue #1: Semaphore created lazily inside async context, not at init time.
        # asyncio primitives should be created within a running event loop.
        self._semaphore: Optional[asyncio.Semaphore] = None

        self.cache_enabled = ENABLE_VISION_CACHE and HAS_DISKCACHE
        if self.cache_enabled:
            cache_path = Path(VISION_CACHE_DIR)
            cache_path.mkdir(parents=True, exist_ok=True)
            self.cache = Cache(str(cache_path), size_limit=_CACHE_SIZE_LIMIT)
            logger.info(f"Vision API caching enabled at {cache_path} (TTL: {VISION_CACHE_TTL_DAYS}d, limit: {VISION_CACHE_SIZE_LIMIT_GB}GB)")
        else:
            self.cache = None

    @property
    def semaphore(self) -> asyncio.Semaphore:
        """Lazily create semaphore within a running event loop (Issue #1 fix)."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent)
        return self._semaphore

    def _get_cache_key(self, pil_image: Image.Image, context: str) -> str:
        """Generate cache key from raw pixel data — faster than PNG encoding (Issue #4 fix)."""
        raw_bytes = pil_image.tobytes() + context.encode('utf-8')
        return hashlib.sha256(raw_bytes).hexdigest()

    async def analyze_images_batch_memory(
        self,
        pil_images: List[Image.Image],
        contexts: List[str]
    ) -> List[Dict]:
        """Analyze multiple PIL Images directly from memory (no disk I/O)."""
        if len(pil_images) != len(contexts):
            raise ValueError("pil_images and contexts must have same length")

        tasks = [
            self._analyze_single_memory(img, ctx)
            for img, ctx in zip(pil_images, contexts)
        ]

        raw_gather_results = await asyncio.gather(*tasks, return_exceptions=True)

        analysis_results = []
        for image_idx, gather_result in enumerate(raw_gather_results):
            if isinstance(gather_result, Exception):
                logger.error(f"Image {image_idx} analysis failed: {gather_result}")
                analysis_results.append({
                    'keep': False,
                    'reason': f"Analysis error: {str(gather_result)}",
                    'description': None,
                    'entities': None
                })
            else:
                analysis_results.append(gather_result)

        return analysis_results

    async def _analyze_single_memory(self, pil_image: Image.Image, context: str) -> Dict:
        """Analyze single PIL Image from memory with semaphore control."""
        async with self.semaphore:
            try:
                if self.cache_enabled:
                    cache_key = self._get_cache_key(pil_image, context)
                    cached_analysis = self.cache.get(cache_key)
                    if cached_analysis is not None:
                        return cached_analysis

                base64_image = self._encode_image_from_pil(pil_image)
                response_text = await self._call_api_with_retry(base64_image, context)
                analysis_result = self._parse_response(response_text)

                if self.cache_enabled:
                    self.cache.set(cache_key, analysis_result, expire=_CACHE_TTL_SECONDS)

                return analysis_result
            except Exception as e:
                logger.error(f"AI analysis failed for image: {e}")
                return {
                    "keep": False,
                    "reason": f"Error: {e}",
                    "description": None,
                    "entities": None
                }

    def _encode_image_from_pil(self, pil_image: Image.Image) -> str:
        """Encode PIL Image to base64 without saving to disk."""
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    async def _call_api_with_retry(self, base64_image: str, context: str) -> str:
        """Call API with exponential backoff on rate limits, connection errors, and 5xx responses (Issue #7 fix)."""
        for attempt in range(self.max_retries):
            try:
                return await self._call_api(base64_image, context)
            except RateLimitError as e:
                self._maybe_raise(e, attempt)
                await self._backoff(attempt, "Rate limit")
            except APIConnectionError as e:
                self._maybe_raise(e, attempt)
                await self._backoff(attempt, "Connection error")
            except APIStatusError as e:
                # 4xx errors are caller errors — not retryable (except 429 which becomes RateLimitError)
                if e.status_code < 500:
                    raise
                self._maybe_raise(e, attempt)
                await self._backoff(attempt, f"Server error {e.status_code}")

        raise RuntimeError("Max retries exceeded")

    def _maybe_raise(self, error: Exception, attempt: int) -> None:
        """Re-raise if this was the last allowed attempt."""
        if attempt == self.max_retries - 1:
            raise error

    async def _backoff(self, attempt: int, reason: str) -> None:
        """Wait with exponential backoff, then log."""
        wait_time = self.base_delay * (2 ** attempt)
        logger.warning(f"{reason}, waiting {wait_time:.1f}s (retry {attempt + 1}/{self.max_retries})")
        await asyncio.sleep(wait_time)

    async def _call_api(self, base64_image: str, context: str) -> str:
        """Call OpenAI API with 3-way classification prompt."""
        prompt = f"""Analyze this image for Knowledge Graph RAG extraction.

Context: {context[:300] if context else 'None'}

CRITICAL: Start with EXACTLY "0", "1", or "2":
- "1" = KEEP (diagrams, charts, technical schematics, flowcharts, graphs, infographics)
- "0" = REJECT (decorative: logos, icons, stock photos, backgrounds, page separators)
- "2" = TABLE (data tables, spreadsheets, tabular data with rows/columns)

If "1", continue with:
DESCRIPTION: [Detailed analysis - type, components, data, labels, relationships]
ENTITIES: [{{"type":"component|concept|metric|process", "name":"...", "properties":"..."}}]

If "0", continue with:
REASON: [Why rejected]

If "2", continue with:
REASON: Table detected - will be reconstructed separately.

Examples:
"1
DESCRIPTION: System architecture diagram showing 3-tier design: frontend (React), API layer (Node.js), database (PostgreSQL). Arrows indicate data flow.
ENTITIES: [{{"type":"component","name":"Frontend","properties":"React framework"}},{{"type":"component","name":"API Layer","properties":"Node.js with REST"}}]"

"0
REASON: Company logo in header, purely decorative."

"2
REASON: Data table with performance metrics across multiple categories."

Now analyze:"""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{base64_image}",
                        "detail": "auto"
                    }}
                ]
            }],
            max_tokens=VISION_MAX_TOKENS_FILTER,
            temperature=VISION_TEMPERATURE,
        )

        return response.choices[0].message.content.strip()

    def _parse_response(self, text: str) -> Dict:
        """Parse AI response into structured data (handles 0/1/2 classification codes)."""
        if not text:
            return {"keep": False, "is_table": False, "reason": "Empty response",
                    "description": None, "entities": None}

        classification_code = text[0]
        if classification_code not in ['0', '1', '2']:
            logger.warning(f"Unexpected classification code: '{classification_code}'")
            return {"keep": False, "is_table": False, "reason": "Invalid format",
                    "description": None, "entities": None}

        reason_regex_match = re.search(r'REASON:\s*(.+)', text, re.IGNORECASE | re.DOTALL)
        reason = reason_regex_match.group(1).strip() if reason_regex_match else text[1:].strip()

        if classification_code == '2':
            return {"keep": False, "is_table": True, "reason": reason,
                    "description": None, "entities": None}

        if classification_code == '0':
            return {"keep": False, "is_table": False, "reason": reason,
                    "description": None, "entities": None}

        # classification_code == '1': valuable figure, extract description and entities
        description_regex_match = re.search(
            r'DESCRIPTION:\s*(.+?)(?=ENTITIES:|$)', text, re.IGNORECASE | re.DOTALL
        )
        description = description_regex_match.group(1).strip() if description_regex_match else ""

        entities = []
        entities_regex_match = re.search(r'ENTITIES:\s*(\[.+?\])', text, re.IGNORECASE | re.DOTALL)
        if entities_regex_match:
            try:
                entities = json.loads(entities_regex_match.group(1))
            except json.JSONDecodeError:
                logger.warning("Failed to parse entities JSON from vision response")

        return {"keep": True, "is_table": False, "reason": "Valuable for knowledge graph",
                "description": description, "entities": entities}

    async def reconstruct_table(self, pil_image: Image.Image, context: str = "") -> Dict:
        """Send a table image to OpenAI for markdown reconstruction + GraphRAG description.

        Returns dict with 'markdown' and 'description' keys.
        """
        async with self.semaphore:
            base64_image = self._encode_image_from_pil(pil_image)

            prompt = f"""Reconstruct the table in this image as a clean Markdown table, then describe it.

Context: {context[:200] if context else 'None'}

Rules:
- Preserve all data values exactly as shown
- Use proper Markdown table syntax with | and ---
- Maintain column alignment
- If cells are merged, repeat the value
- If text is unclear, use [unclear] placeholder

Output format (follow EXACTLY):
TABLE:
| Column1 | Column2 |
| --- | --- |
| data | data |

DESCRIPTION: [1-2 sentences: what this table contains, key entities, metrics, and relationships useful for a knowledge graph]

Output now:"""

            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high"
                            }}
                        ]
                    }],
                    max_tokens=VISION_MAX_TOKENS_TABLE,
                    temperature=VISION_TEMPERATURE,
                )
                raw_response_text = response.choices[0].message.content.strip()
                return self._parse_table_response(raw_response_text)
            except Exception as e:
                logger.error(f"Table reconstruction failed: {e}")
                return {"markdown": "", "description": ""}

    async def reconstruct_table_from_html(self, html_content: str, context: str = "") -> Dict:
        """Reconstruct a table from Docling HTML output via OpenAI (text-only, no image).

        Returns dict with 'markdown' and 'description' keys.
        """
        async with self.semaphore:
            prompt = f"""Convert this HTML table into a clean Markdown table, then describe it.

Context: {context[:200] if context else 'None'}

HTML Table:
{html_content[:3000]}

Rules:
- Preserve all data values exactly
- Use proper Markdown table syntax with | and ---
- Maintain column alignment
- If cells are merged, repeat the value

Output format (follow EXACTLY):
TABLE:
| Column1 | Column2 |
| --- | --- |
| data | data |

DESCRIPTION: [1-2 sentences: what this table contains, key entities, metrics, and relationships useful for a knowledge graph]

Output now:"""

            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }],
                    max_tokens=VISION_MAX_TOKENS_TABLE,
                    temperature=VISION_TEMPERATURE,
                )
                raw_response_text = response.choices[0].message.content.strip()
                return self._parse_table_response(raw_response_text)
            except Exception as e:
                logger.error(f"Table reconstruction from HTML failed: {e}")
                return {"markdown": "", "description": ""}

    def _parse_table_response(self, text: str) -> Dict:
        """Parse table reconstruction response into markdown + description."""
        if not text:
            return {"markdown": "", "description": ""}

        description_match = re.search(r'DESCRIPTION:\s*(.+)', text, re.IGNORECASE | re.DOTALL)
        description = description_match.group(1).strip() if description_match else ""

        # Extract the markdown table block between TABLE: and DESCRIPTION:
        table_section_match = re.search(
            r'TABLE:\s*\n(.*?)(?=\nDESCRIPTION:|\Z)', text, re.IGNORECASE | re.DOTALL
        )
        if table_section_match:
            markdown = table_section_match.group(1).strip()
        else:
            # Fallback: everything before DESCRIPTION is the table
            markdown = text[:description_match.start()].strip() if description_match else text.strip()

        return {"markdown": markdown, "description": description}
