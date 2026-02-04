"""AI-powered image filtering with async batch processing."""

import asyncio
import base64
import json
import logging
import os
import re
import time
from typing import Dict, List, Optional

from openai import AsyncOpenAI, RateLimitError

from config import OPENAI_API_KEY, VISION_MODEL

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
        
    async def analyze_images_batch(self, image_paths: List[str], contexts: List[str]) -> List[Dict]:
        """Analyze multiple images concurrently."""
        tasks = [
            self._analyze_single(path, ctx) 
            for path, ctx in zip(image_paths, contexts)
        ]
        return await asyncio.gather(*tasks)
    
    async def _analyze_single(self, image_path: str, context: str) -> Dict:
        """Analyze single image with semaphore control."""
        async with self.semaphore:
            try:
                base64_image = self._encode_image(image_path)
                response_text = await self._call_api_with_retry(base64_image, context)
                return self._parse_response(response_text)
            except Exception as e:
                logger.error(f"AI analysis failed for {image_path}: {e}")
                return {"keep": False, "reason": f"Error: {e}", "description": None, "entities": None}
    
    def analyze_image(self, image_path: str, context: str = "") -> Dict:
        """Synchronous wrapper for single image (backwards compatible)."""
        return asyncio.run(self._analyze_single(image_path, context))
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    async def _call_api_with_retry(self, base64_image: str, context: str) -> str:
        """Call API with exponential backoff retry logic."""
        for attempt in range(self.max_retries):
            try:
                return await self._call_api(base64_image, context)
            except RateLimitError as e:
                if attempt == self.max_retries - 1:
                    raise
                
                wait_time = self.base_delay * (2 ** attempt)
                logger.warning(f"Rate limit hit, waiting {wait_time:.1f}s before retry {attempt + 1}/{self.max_retries}")
                await asyncio.sleep(wait_time)
            except Exception as e:
                raise
        
        raise Exception("Max retries exceeded")
    
    async def _call_api(self, base64_image: str, context: str) -> str:
        """Call OpenAI API with optimized prompt."""
        prompt = f"""Analyze this image for Knowledge Graph RAG extraction.

Context: {context[:300] if context else 'None'}

CRITICAL: Start with EXACTLY "0" or "1":
- "1" = KEEP (valuable for knowledge graphs: diagrams, charts, technical schematics, data tables, flowcharts)
- "0" = REJECT (decorative: logos, icons, stock photos, backgrounds, page separators)

If "1", continue with:
DESCRIPTION: [Detailed analysis - type, components, data, labels, relationships]
ENTITIES: [{{"type":"component|concept|metric|process", "name":"...", "properties":"..."}}]

If "0", continue with:
REASON: [Why rejected]

Examples:
"1
DESCRIPTION: System architecture diagram showing 3-tier design: frontend (React), API layer (Node.js), database (PostgreSQL). Arrows indicate data flow. Labels show REST endpoints and authentication flow.
ENTITIES: [{{"type":"component","name":"Frontend","properties":"React framework"}},{{"type":"component","name":"API Layer","properties":"Node.js with REST"}},{{"type":"database","name":"PostgreSQL","properties":"Data storage"}}]"

"0
REASON: Company logo in header, purely decorative."

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
        """Parse AI response into structured data."""
        if not text:
            return {"keep": False, "reason": "Empty response", "description": None, "entities": None}
        
        decision = text[0]
        if decision not in ['0', '1']:
            logger.warning(f"Invalid decision character: {decision}")
            return {"keep": False, "reason": "Invalid format", "description": None, "entities": None}
        
        keep = (decision == '1')
        
        if not keep:
            reason_match = re.search(r'REASON:\s*(.+)', text, re.IGNORECASE | re.DOTALL)
            reason = reason_match.group(1).strip() if reason_match else text[1:].strip()
            return {"keep": False, "reason": reason, "description": None, "entities": None}
        
        desc_match = re.search(r'DESCRIPTION:\s*(.+?)(?=ENTITIES:|$)', text, re.IGNORECASE | re.DOTALL)
        description = desc_match.group(1).strip() if desc_match else ""
        
        entities = []
        entities_match = re.search(r'ENTITIES:\s*(\[.+?\])', text, re.IGNORECASE | re.DOTALL)
        if entities_match:
            try:
                entities = json.loads(entities_match.group(1))
            except json.JSONDecodeError:
                logger.warning("Failed to parse entities JSON")
        
        return {
            "keep": True,
            "reason": "Valuable for knowledge graph",
            "description": description,
            "entities": entities
        }
