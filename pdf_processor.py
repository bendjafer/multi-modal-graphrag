"""Core PDF processing with async batch image filtering."""

import asyncio
import hashlib
import json
import logging
import time
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from PIL import Image

from config import (
    BASE_OUTPUT_DIR, IMAGES_SUBDIR, MARKDOWN_SUBDIR,
    MIN_IMAGE_SIZE, MAX_IMAGE_SIZE,
    VISION_MODEL, OPENAI_API_KEY, MAX_CONCURRENT_REQUESTS
)
from models import ImageAsset, ProcessingResult
from vision_model import VisionModel

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Handles PDF processing with async AI-powered image filtering."""
    
    def __init__(self, output_base: Path = Path(BASE_OUTPUT_DIR)):
        self.output_base = output_base
        self.vision = VisionModel(
            model=VISION_MODEL, 
            api_key=OPENAI_API_KEY, 
            max_concurrent=MAX_CONCURRENT_REQUESTS
        )
        
    def process(self, pdf_path: Path, context: str = "") -> ProcessingResult:
        """Process a single PDF file."""
        start_time = time.time()
        
        # Reset stats for this document
        self.stats = {
            'found': 0,
            'filtered_size': 0,
            'filtered_duplicate': 0,
            'filtered_ai': 0,
            'kept': 0
        }
        self.seen_hashes = set()
        
        doc_id = pdf_path.stem
        output_dir = self.output_base / f"{doc_id}_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        images_dir = output_dir / IMAGES_SUBDIR
        images_dir.mkdir(exist_ok=True)
        
        converter = self._create_converter()
        result = converter.convert(pdf_path)
        document = result.document
        
        markdown_content = document.export_to_markdown()
        markdown_path = output_dir / MARKDOWN_SUBDIR / f"{doc_id}.md"
        markdown_path.parent.mkdir(exist_ok=True)
        markdown_path.write_text(markdown_content, encoding='utf-8')
        
        images = asyncio.run(self._extract_and_filter_images_async(
            document, 
            images_dir, 
            context
        ))
        
        result_obj = ProcessingResult(
            document_id=doc_id,
            total_pages=len(document.pages),
            markdown_path=markdown_path,
            images=images,
            stats=self.stats.copy()
        )
        
        metadata_path = output_dir / f"{doc_id}_metadata.json"
        metadata_path.write_text(json.dumps(result_obj.to_dict(), indent=2))
        
        elapsed = time.time() - start_time
        logger.info(f"Finished converting document {doc_id}.pdf in {elapsed:.2f} sec.")
        self._log_stats()
        
        return result_obj
    
    def _create_converter(self) -> DocumentConverter:
        """Initialize document converter with optimized OCR settings."""
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.generate_page_images = False
        pipeline_options.generate_picture_images = True
        
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
    
    async def _extract_and_filter_images_async(
        self, 
        document, 
        output_dir: Path,
        context: str
    ) -> List[ImageAsset]:
        """Extract figures and filter with AI in async batches."""
        
        image_candidates = []
        
        # Extract pictures from document
        try:
            pictures = list(document.pictures)
            logger.info(f"Found {len(pictures)} pictures in document")
            
            for pic_idx, picture in enumerate(pictures):
                self.stats['found'] += 1
                
                # Get page number
                page_label = 1
                if hasattr(picture, 'prov') and picture.prov and len(picture.prov) > 0:
                    page_label = picture.prov[0].page_no
                
                # Get image data
                try:
                    if hasattr(picture, 'image') and picture.image:
                        image_data = picture.image.pil_image
                    elif hasattr(picture, 'get_image'):
                        image_data = picture.get_image(document)
                    else:
                        continue
                except Exception as e:
                    logger.debug(f"Could not load picture {pic_idx}: {e}")
                    continue
                
                # Size filter
                if not self._check_image_size(image_data):
                    self.stats['filtered_size'] += 1
                    continue
                
                # Duplicate filter
                img_hash = self._compute_hash(image_data)
                if img_hash in self.seen_hashes:
                    self.stats['filtered_duplicate'] += 1
                    continue
                
                self.seen_hashes.add(img_hash)
                
                # Get caption
                caption = ""
                if hasattr(picture, 'caption'):
                    caption = str(picture.caption) if picture.caption else ""
                
                full_context = f"{context} | Page {page_label} | Caption: {caption}"
                
                # Save temporary image
                temp_path = output_dir / f"temp_p{page_label}_i{pic_idx}_{img_hash[:8]}.png"
                image_data.save(temp_path, 'PNG')
                
                image_candidates.append({
                    'path': temp_path,
                    'final_name': f"page_{page_label:03d}_fig_{pic_idx}_{img_hash[:16]}.png",
                    'context': full_context,
                    'hash': img_hash,
                    'page': page_label,
                    'image_data': image_data
                })
                
        except Exception as e:
            logger.error(f"Error extracting pictures: {e}", exc_info=True)
        
        if not image_candidates:
            logger.info("No images passed size/duplicate filters")
            return []
        
        # Batch AI analysis
        logger.info(f"Analyzing {len(image_candidates)} images with AI...")
        
        paths = [str(img['path']) for img in image_candidates]
        contexts = [img['context'] for img in image_candidates]
        
        ai_results = await self.vision.analyze_images_batch(paths, contexts)
        
        # Process results
        kept_images = []
        for candidate, ai_result in zip(image_candidates, ai_results):
            temp_path = candidate['path']
            
            if not ai_result['keep']:
                temp_path.unlink(missing_ok=True)
                self.stats['filtered_ai'] += 1
                logger.info(f"❌ {candidate['final_name']}: {ai_result['reason']}")
                continue
            
            final_path = output_dir / candidate['final_name']
            temp_path.rename(final_path)
            
            asset = ImageAsset(
                path=final_path,
                page=candidate['page'],
                width=candidate['image_data'].width,
                height=candidate['image_data'].height,
                hash=candidate['hash'],
                description=ai_result.get('description'),
                entities=ai_result.get('entities')
            )
            kept_images.append(asset)
            self.stats['kept'] += 1
            
            entity_count = len(ai_result.get('entities', []))
            logger.info(f"✅ {candidate['final_name']}: {entity_count} entities")
        
        return kept_images
    
    def _check_image_size(self, image: Image.Image) -> bool:
        """Check if image meets size requirements."""
        w, h = image.size
        min_w, min_h = MIN_IMAGE_SIZE
        max_w, max_h = MAX_IMAGE_SIZE
        return (min_w <= w <= max_w) and (min_h <= h <= max_h)
    
    def _compute_hash(self, image: Image.Image) -> str:
        """Compute SHA256 hash of image."""
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        return hashlib.sha256(buffer.getvalue()).hexdigest()
    
    def _log_stats(self):
        """Log processing statistics."""
        logger.info(f"\n{'='*50}")
        logger.info(f"Found: {self.stats['found']}")
        logger.info(f"Filtered (size): {self.stats['filtered_size']}")
        logger.info(f"Filtered (duplicate): {self.stats['filtered_duplicate']}")
        logger.info(f"Filtered (AI): {self.stats['filtered_ai']}")
        logger.info(f"✓ Kept: {self.stats['kept']}")
        logger.info(f"{'='*50}")
