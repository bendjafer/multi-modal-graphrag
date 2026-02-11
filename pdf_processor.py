"""Core PDF processing with async batch image filtering - MEMORY EFFICIENT."""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions, TableStructureOptions, TableFormerMode
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from PIL import Image

from config import (
    BASE_OUTPUT_DIR, IMAGES_SUBDIR, TABLES_SUBDIR, MARKDOWN_SUBDIR,
    CHUNKS_SUBDIR, MIN_IMAGE_SIZE, MAX_IMAGE_SIZE,
    VISION_MODEL, OPENAI_API_KEY, MAX_CONCURRENT_REQUESTS
)
from models import ImageAsset, TableAsset, ProcessingResult
from vision_model import VisionModel
from markdown_processor import MarkdownProcessor
from chunker import DocumentChunker

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
        self.markdown_processor = MarkdownProcessor()
        self.chunker = DocumentChunker()
        
    def process(self, pdf_path: Path, context: str = "") -> ProcessingResult:
        """Process a single PDF file."""
        start_time = time.time()
        
        self.stats = {
            'found': 0,
            'filtered_size': 0,
            'filtered_ai': 0,
            'tables_docling': 0,
            'tables_vision': 0,
            'tables_reconstructed': 0,
            'kept': 0
        }
        
        doc_id = pdf_path.stem
        output_dir = self.output_base / f"{doc_id}_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        images_dir = output_dir / IMAGES_SUBDIR
        images_dir.mkdir(exist_ok=True)
        
        tables_dir = output_dir / TABLES_SUBDIR
        tables_dir.mkdir(exist_ok=True)
        
        # Convert PDF with optimized Docling settings
        converter = self._create_converter()
        result = converter.convert(pdf_path)
        document = result.document
        
        # Export raw markdown
        raw_markdown_content = document.export_to_markdown()
        raw_markdown_path = output_dir / MARKDOWN_SUBDIR / f"{doc_id}_raw.md"
        raw_markdown_path.parent.mkdir(exist_ok=True)
        raw_markdown_path.write_text(raw_markdown_content, encoding='utf-8')
        
        # --- TABLES: Extract natively from Docling + reconstruct via OpenAI ---
        docling_tables = asyncio.run(
            self._extract_and_reconstruct_tables(document, tables_dir, context)
        )
        
        # --- IMAGES: Extract, filter with AI, route vision-detected tables ---
        images, vision_tables = asyncio.run(
            self._extract_and_filter_images_async(document, images_dir, tables_dir, context)
        )
        
        # Merge all tables
        all_tables = docling_tables + vision_tables
        
        # Save tables as individual markdown files
        for i, table in enumerate(all_tables):
            table_path = tables_dir / f"table_page{table.page:03d}_{i}.md"
            content = table.markdown_content
            if table.description:
                content = f"<!-- description: {table.description} -->\n\n{content}"
            table_path.write_text(content, encoding='utf-8')
        
        # Prepare image metadata for markdown injection
        image_metadata = [
            {'page': img.page, 'description': img.description, 'entities': img.entities} 
            for img in images
        ]
        
        # Process markdown (clean + add image descriptions)
        cleaned_markdown, markdown_stats = self.markdown_processor.pipeline(
            raw_markdown_path=raw_markdown_path,
            image_metadata=image_metadata,
            save_cleaned=False
        )
        
        # Save final markdown
        markdown_path = output_dir / MARKDOWN_SUBDIR / f"{doc_id}.md"
        markdown_path.write_text(cleaned_markdown, encoding='utf-8')
        
        # --- CHUNKING: Split into text, table, and image chunks ---
        chunks_dir = output_dir / CHUNKS_SUBDIR
        chunks_dir.mkdir(exist_ok=True)
        
        chunking_result = self.chunker.chunk_document(
            document_id=doc_id,
            markdown_content=cleaned_markdown,
            tables=all_tables,
            images=images,
            language=markdown_stats.get('language'),
        )
        self.chunker.save_chunks(chunking_result, chunks_dir)
        
        # Create result
        combined_stats = {**self.stats, **markdown_stats, **chunking_result.stats}
        result_obj = ProcessingResult(
            document_id=doc_id,
            total_pages=len(document.pages),
            markdown_path=markdown_path,
            images=images,
            tables=all_tables,
            stats=combined_stats
        )
        
        # Save metadata
        metadata_path = output_dir / f"{doc_id}_metadata.json"
        metadata_path.write_text(json.dumps(result_obj.to_dict(), indent=2))
        
        elapsed = time.time() - start_time
        logger.info(f"Finished converting document {doc_id}.pdf in {elapsed:.2f} sec.")
        self._log_stats()
        
        return result_obj
    
    def _create_converter(self) -> DocumentConverter:
        """Initialize document converter with optimal settings for GraphRAG."""
        pipeline_options = PdfPipelineOptions(
            # Table extraction — critical for this project
            do_table_structure=True,
            table_structure_options=TableStructureOptions(
                do_cell_matching=True,
                mode=TableFormerMode.ACCURATE
            ),
            generate_table_images=True,
            
            # Image extraction
            generate_picture_images=True,
            images_scale=2.0,             # 2x resolution for better AI classification
            generate_page_images=False,
            
            # OCR — disable unless processing scanned PDFs
            do_ocr=False,
        )
        
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
    
    async def _extract_and_reconstruct_tables(
        self, document, tables_dir: Path, context: str
    ) -> List[TableAsset]:
        """Extract tables natively from Docling and reconstruct via OpenAI."""
        tables = []
        
        try:
            doc_tables = list(document.tables)
            logger.info(f"Found {len(doc_tables)} tables via Docling")
            self.stats['tables_docling'] = len(doc_tables)
            
            if not doc_tables:
                return []
            
            # Build reconstruction tasks
            tasks = []
            table_meta = []
            
            for tbl_idx, table in enumerate(doc_tables):
                # Get page number
                page = 1
                if hasattr(table, 'prov') and table.prov and len(table.prov) > 0:
                    page = table.prov[0].page_no
                
                # Get caption
                caption = ""
                if hasattr(table, 'caption_text'):
                    caption = table.caption_text(document) or ""
                elif hasattr(table, 'captions') and table.captions:
                    caption = str(table.captions[0]) if table.captions else ""
                
                full_context = f"{context} | Page {page} | Table caption: {caption}"
                
                # Prefer image-based reconstruction if table image available
                if hasattr(table, 'image') and table.image:
                    try:
                        pil_img = table.image.pil_image
                        tasks.append(self.vision.reconstruct_table(pil_img, full_context))
                        table_meta.append({'page': page, 'source': 'docling_image'})
                        continue
                    except Exception:
                        pass
                
                # Fallback: export to HTML and reconstruct from text
                try:
                    html_content = table.export_to_html()
                    tasks.append(
                        self.vision.reconstruct_table_from_html(html_content, full_context)
                    )
                    table_meta.append({'page': page, 'source': 'docling_html'})
                except Exception as e:
                    logger.warning(f"Could not export table {tbl_idx} to HTML: {e}")
            
            if not tasks:
                return []
            
            # Run all reconstructions
            logger.info(f"Reconstructing {len(tasks)} Docling tables via OpenAI...")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for meta, result in zip(table_meta, results):
                if isinstance(result, Exception):
                    logger.error(f"Table reconstruction failed: {result}")
                    continue
                if result and result.get('markdown'):
                    tables.append(TableAsset(
                        page=meta['page'],
                        markdown_content=result['markdown'],
                        source_type=meta['source'],
                        description=result.get('description', '')
                    ))
                    self.stats['tables_reconstructed'] += 1
                    logger.info(
                        f"📊 Reconstructed table page {meta['page']} "
                        f"via {meta['source']}"
                    )
                    
        except Exception as e:
            logger.error(f"Error extracting Docling tables: {e}", exc_info=True)
        
        return tables
    
    async def _extract_and_filter_images_async(
        self, document, output_dir: Path, tables_dir: Path, context: str
    ) -> Tuple[List[ImageAsset], List[TableAsset]]:
        """Extract figures, filter with AI, route tables separately."""
        
        min_w, min_h = MIN_IMAGE_SIZE
        max_w, max_h = MAX_IMAGE_SIZE
        image_candidates = []
        
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
                
                # Inline size filter
                w, h = image_data.size
                if not (min_w <= w <= max_w and min_h <= h <= max_h):
                    self.stats['filtered_size'] += 1
                    continue
                
                # Get caption
                caption = ""
                if hasattr(picture, 'caption'):
                    caption = str(picture.caption) if picture.caption else ""
                
                full_context = f"{context} | Page {page_label} | Caption: {caption}"
                
                image_candidates.append({
                    'image_data': image_data,
                    'final_name': f"page_{page_label:03d}_fig_{pic_idx}.png",
                    'context': full_context,
                    'page': page_label,
                    'pic_idx': pic_idx
                })
                
        except Exception as e:
            logger.error(f"Error extracting pictures: {e}", exc_info=True)
        
        if not image_candidates:
            logger.info("No images passed size filter")
            return [], []
        
        # Batch AI analysis
        logger.info(f"Analyzing {len(image_candidates)} images with AI...")
        
        pil_images = [c['image_data'] for c in image_candidates]
        contexts = [c['context'] for c in image_candidates]
        ai_results = await self.vision.analyze_images_batch_memory(pil_images, contexts)
        
        # Route results: keep / reject / table
        kept_images = []
        table_candidates = []
        
        for candidate, ai_result in zip(image_candidates, ai_results):
            if ai_result.get('is_table'):
                # Table detected via vision — queue for reconstruction
                self.stats['tables_vision'] += 1
                table_candidates.append(candidate)
                logger.info(f"📊 {candidate['final_name']}: table detected by AI")
                continue
            
            if not ai_result['keep']:
                self.stats['filtered_ai'] += 1
                logger.info(f"❌ {candidate['final_name']}: {ai_result['reason']}")
                continue
            
            # Save approved image to disk
            final_path = output_dir / candidate['final_name']
            candidate['image_data'].save(final_path, 'PNG')
            
            asset = ImageAsset(
                path=final_path,
                page=candidate['page'],
                width=candidate['image_data'].width,
                height=candidate['image_data'].height,
                description=ai_result.get('description'),
                entities=ai_result.get('entities')
            )
            kept_images.append(asset)
            self.stats['kept'] += 1
            
            entity_count = len(ai_result.get('entities', []))
            logger.info(f"✅ {candidate['final_name']}: {entity_count} entities")
        
        # Reconstruct vision-detected tables via OpenAI
        tables = []
        if table_candidates:
            logger.info(f"Reconstructing {len(table_candidates)} vision-detected tables...")
            table_tasks = [
                self.vision.reconstruct_table(c['image_data'], c['context'])
                for c in table_candidates
            ]
            table_results = await asyncio.gather(*table_tasks, return_exceptions=True)
            
            for candidate, result in zip(table_candidates, table_results):
                if isinstance(result, Exception):
                    logger.error(f"Table reconstruction failed: {result}")
                    continue
                if result and result.get('markdown'):
                    tables.append(TableAsset(
                        page=candidate['page'],
                        markdown_content=result['markdown'],
                        source_type='vision',
                        description=result.get('description', '')
                    ))
                    self.stats['tables_reconstructed'] += 1
                    logger.info(f"📊 Reconstructed table from {candidate['final_name']}")
        
        return kept_images, tables
    
    def _log_stats(self):
        """Log processing statistics."""
        logger.info(f"\n{'='*50}")
        logger.info(f"Images found: {self.stats['found']}")
        logger.info(f"Filtered (size): {self.stats['filtered_size']}")
        logger.info(f"Filtered (AI): {self.stats['filtered_ai']}")
        logger.info(f"Tables (Docling native): {self.stats['tables_docling']}")
        logger.info(f"Tables (vision-detected): {self.stats['tables_vision']}")
        logger.info(f"Tables reconstructed: {self.stats['tables_reconstructed']}")
        logger.info(f"✓ Images kept: {self.stats['kept']}")
        logger.info(f"{'='*50}")
