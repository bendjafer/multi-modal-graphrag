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

from openai import AsyncOpenAI, RateLimitError
from PIL import Image

try:
    from diskcache import Cache
    HAS_DISKCACHE = True
except ImportError:
    HAS_DISKCACHE = False
    logging.warning("diskcache not installed. Vision API responses won't be cached. Install with: pip install diskcache")

from config import OPENAI_API_KEY, VISION_MODEL, ENABLE_VISION_CACHE, VISION_CACHE_DIR

logger = logging.getLogger(__name__)


class VisionModel:
    """Async vision processor with concurrent batch processing."""
    
    def __init__(self, model: str = None, api_key: Optional[str] = None, 
                 max_retries: int = 5, base_delay: float = 1.0, max_concurrent: int = 10):
        self.model = model or VISION_MODEL
        self.client = AsyncOpenAI(api_key=api_key or OPENAI_API_KEY)
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # Initialize cache if enabled
        self.cache_enabled = ENABLE_VISION_CACHE and HAS_DISKCACHE
        if self.cache_enabled:
            cache_path = Path(VISION_CACHE_DIR)
            cache_path.mkdir(parents=True, exist_ok=True)
            self.cache = Cache(str(cache_path))
            logger.info(f"Vision API caching enabled at {cache_path}")
        else:
            self.cache = None
    
    def _get_cache_key(self, image_data: bytes, context: str) -> str:
        """Generate cache key from image data and context."""
        content = image_data + context.encode('utf-8')
        return hashlib.sha256(content).hexdigest()
    
    async def analyze_images_batch(self, image_paths: List[str], contexts: List[str]) -> List[Dict]:
        """Analyze multiple images concurrently from file paths."""
        tasks = [
            self._analyze_single(path, ctx) 
            for path, ctx in zip(image_paths, contexts)
        ]
        return await asyncio.gather(*tasks)
    
    async def analyze_images_batch_memory(
        self, 
        pil_images: List[Image.Image], 
        contexts: List[str]
    ) -> List[Dict]:
        """
        Analyze multiple PIL Images directly from memory (no disk I/O).
        
        Args:
            pil_images: List of PIL Image objects
            contexts: Context for each image
            
        Returns:
            List of analysis results
        """
        if len(pil_images) != len(contexts):
            raise ValueError("pil_images and contexts must have same length")
        
        tasks = [
            self._analyze_single_memory(img, ctx) 
            for img, ctx in zip(pil_images, contexts)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Image {i} analysis failed: {result}")
                processed.append({
                    'keep': False,
                    'reason': f"Analysis error: {str(result)}",
                    'description': None,
                    'entities': None
                })
            else:
                processed.append(result)
        
        return processed
    
    async def _analyze_single(self, image_path: str, context: str) -> Dict:
        """Analyze single image from file path with semaphore control."""
        async with self.semaphore:
            try:
                base64_image = self._encode_image_from_path(image_path)
                response_text = await self._call_api_with_retry(base64_image, context)
                return self._parse_response(response_text)
            except Exception as e:
                logger.error(f"AI analysis failed for {image_path}: {e}")
                return {
                    "keep": False, 
                    "reason": f"Error: {e}", 
                    "description": None, 
                    "entities": None
                }
    
    async def _analyze_single_memory(self, pil_image: Image.Image, context: str) -> Dict:
        """Analyze single PIL Image from memory with semaphore control."""
        async with self.semaphore:
            try:
                # Encode image for cache key and API
                base64_image = self._encode_image_from_pil(pil_image)
                
                # Check cache if enabled
                if self.cache_enabled:
                    cache_key = self._get_cache_key(base64.b64decode(base64_image), context)
                    cached_result = self.cache.get(cache_key)
                    if cached_result is not None:
                        logger.debug(f"Cache hit for image analysis")
                        return cached_result
                
                # Cache miss or disabled - call API
                response_text = await self._call_api_with_retry(base64_image, context)
                result = self._parse_response(response_text)
                
                # Store in cache if enabled
                if self.cache_enabled:
                    self.cache.set(cache_key, result)
                
                return result
            except Exception as e:
                logger.error(f"AI analysis failed for image: {e}")
                return {
                    "keep": False, 
                    "reason": f"Error: {e}", 
                    "description": None, 
                    "entities": None
                }
    
    def analyze_image(self, image_path: str, context: str = "") -> Dict:
        """Synchronous wrapper for single image (backwards compatible)."""
        return asyncio.run(self._analyze_single(image_path, context))
    
    def _encode_image_from_path(self, image_path: str) -> str:
        """Encode image file to base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def _encode_image_from_pil(self, pil_image: Image.Image) -> str:
        """Encode PIL Image to base64 without saving to disk."""
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    async def _call_api_with_retry(self, base64_image: str, context: str) -> str:
        """Call API with exponential backoff retry logic."""
        for attempt in range(self.max_retries):
            try:
                return await self._call_api(base64_image, context)
            except RateLimitError as e:
                if attempt == self.max_retries - 1:
                    raise
                
                wait_time = self.base_delay * (2 ** attempt)
                logger.warning(
                    f"Rate limit hit, waiting {wait_time:.1f}s "
                    f"before retry {attempt + 1}/{self.max_retries}"
                )
                await asyncio.sleep(wait_time)
            except Exception as e:
                raise
        
        raise Exception("Max retries exceeded")
    
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
            max_tokens=800,
            temperature=0.1
        )
        
        return response.choices[0].message.content.strip()
    
    def _parse_response(self, text: str) -> Dict:
        """Parse AI response into structured data (handles 0/1/2 decisions)."""
        if not text:
            return {"keep": False, "is_table": False, "reason": "Empty response",
                    "description": None, "entities": None}
        
        decision = text[0]
        if decision not in ['0', '1', '2']:
            logger.warning(f"Invalid decision character: {decision}")
            return {"keep": False, "is_table": False, "reason": "Invalid format",
                    "description": None, "entities": None}
        
        reason_match = re.search(r'REASON:\s*(.+)', text, re.IGNORECASE | re.DOTALL)
        reason = reason_match.group(1).strip() if reason_match else text[1:].strip()
        
        if decision == '2':
            return {"keep": False, "is_table": True, "reason": reason,
                    "description": None, "entities": None}
        
        if decision == '0':
            return {"keep": False, "is_table": False, "reason": reason,
                    "description": None, "entities": None}
        
        # decision == '1' — keep
        desc_match = re.search(
            r'DESCRIPTION:\s*(.+?)(?=ENTITIES:|$)', text, re.IGNORECASE | re.DOTALL
        )
        description = desc_match.group(1).strip() if desc_match else ""
        
        entities = []
        entities_match = re.search(r'ENTITIES:\s*(\[.+?\])', text, re.IGNORECASE | re.DOTALL)
        if entities_match:
            try:
                entities = json.loads(entities_match.group(1))
            except json.JSONDecodeError:
                logger.warning("Failed to parse entities JSON")
        
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
                    max_tokens=2000,
                    temperature=0.1
                )
                raw = response.choices[0].message.content.strip()
                return self._parse_table_response(raw)
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
                    max_tokens=2000,
                    temperature=0.1
                )
                raw = response.choices[0].message.content.strip()
                return self._parse_table_response(raw)
            except Exception as e:
                logger.error(f"Table reconstruction from HTML failed: {e}")
                return {"markdown": "", "description": ""}
    
    def _parse_table_response(self, text: str) -> Dict:
        """Parse table reconstruction response into markdown + description."""
        if not text:
            return {"markdown": "", "description": ""}
        
        desc_match = re.search(r'DESCRIPTION:\s*(.+)', text, re.IGNORECASE | re.DOTALL)
        description = desc_match.group(1).strip() if desc_match else ""
        
        # Extract markdown table: everything between TABLE: and DESCRIPTION:
        table_match = re.search(
            r'TABLE:\s*\n(.*?)(?=\nDESCRIPTION:|\Z)', text, re.IGNORECASE | re.DOTALL
        )
        if table_match:
            markdown = table_match.group(1).strip()
        else:
            # Fallback: take everything before DESCRIPTION as the table
            markdown = text[:desc_match.start()].strip() if desc_match else text.strip()
        
        return {"markdown": markdown, "description": description}
