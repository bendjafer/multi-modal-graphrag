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
from docling_core.types.doc.labels import DocItemLabel
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
        self.converter = self._create_converter()  # initialized once, reused per PDF
        self.stats: Dict = {}

    def process(self, pdf_path: Path, context: str = "") -> ProcessingResult:
        """Process a single PDF file."""
        start_time = time.monotonic()  # Issue #11: monotonic clock is immune to wall-clock adjustments

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

        conversion_result = self.converter.convert(pdf_path)
        document = conversion_result.document

        # Export raw markdown (references excluded — they fragment text for RAG)
        excluded_labels = {DocItemLabel.REFERENCE}
        markdown_labels = {l for l in DocItemLabel if l not in excluded_labels}
        raw_markdown_content = document.export_to_markdown(labels=markdown_labels)

        # Write raw markdown for archival — kept for debugging/reprocessing
        raw_markdown_path = output_dir / MARKDOWN_SUBDIR / f"{doc_id}_raw.md"
        raw_markdown_path.parent.mkdir(exist_ok=True)
        raw_markdown_path.write_text(raw_markdown_content, encoding='utf-8')

        # Issue #2: Run table extraction and image filtering concurrently in one event loop
        docling_tables, extracted_images, vision_tables = asyncio.run(
            self._run_extraction(document, images_dir, tables_dir, context)
        )

        all_tables = docling_tables + vision_tables

        for table_idx, table in enumerate(all_tables):
            table_path = tables_dir / f"table_page{table.page:03d}_{table_idx}.md"
            table_content = table.markdown_content
            if table.description:
                table_content = f"<!-- description: {table.description} -->\n\n{table_content}"
            table_path.write_text(table_content, encoding='utf-8')

        for figure_idx, image_asset in enumerate(extracted_images):
            figure_path = images_dir / f"figure_page{image_asset.page:03d}_{figure_idx}.md"
            figure_content = f"![Figure]({image_asset.path.name})\n"

            if image_asset.description:
                figure_content = f"\n**Description:** {image_asset.description}\n"

            if image_asset.entities:
                figure_content += "\n**Entities:**\n"
                for entity in image_asset.entities:
                    name = entity.get('name', 'unknown')
                    entity_type = entity.get('type', 'unknown')
                    figure_content += f"- {name} ({entity_type})\n"

            figure_path.write_text(figure_content, encoding='utf-8')

        # Issue #3: Pass raw text directly — avoids writing then immediately re-reading
        cleaned_markdown, markdown_stats = self.markdown_processor.pipeline(
            raw_markdown_path=raw_markdown_path,
            raw_markdown_text=raw_markdown_content,
            save_cleaned=False
        )

        markdown_path = output_dir / MARKDOWN_SUBDIR / f"{doc_id}.md"
        markdown_path.write_text(cleaned_markdown, encoding='utf-8')

        chunks_dir = output_dir / CHUNKS_SUBDIR
        chunks_dir.mkdir(exist_ok=True)

        chunking_result = self.chunker.chunk_document(
            document_id=doc_id,
            markdown_content=cleaned_markdown,
            tables=all_tables,
            images=extracted_images,
            language="en",
        )
        self.chunker.save_chunks(chunking_result, chunks_dir)

        combined_stats = {**self.stats, **markdown_stats, **chunking_result.stats}
        processing_result = ProcessingResult(
            document_id=doc_id,
            total_pages=len(document.pages),
            markdown_path=markdown_path,
            images=extracted_images,
            tables=all_tables,
            stats=combined_stats
        )

        metadata_path = output_dir / f"{doc_id}_metadata.json"
        metadata_path.write_text(json.dumps(processing_result.to_dict(), indent=2))

        elapsed = time.monotonic() - start_time
        logger.info(f"Finished processing {doc_id}.pdf in {elapsed:.2f}s")
        self._log_stats()

        return processing_result

    async def _run_extraction(
        self, document, images_dir: Path, tables_dir: Path, context: str
    ) -> Tuple[List[TableAsset], List[ImageAsset], List[TableAsset]]:
        """Run table and image extraction concurrently within a single event loop (Issue #2 fix)."""
        docling_tables, (extracted_images, vision_tables) = await asyncio.gather(
            self._extract_and_reconstruct_tables(document, tables_dir, context),
            self._extract_and_filter_images(document, images_dir, tables_dir, context),
        )
        return docling_tables, extracted_images, vision_tables

    def _create_converter(self) -> DocumentConverter:
        pipeline_options = PdfPipelineOptions(
            do_table_structure=True,
            table_structure_options=TableStructureOptions(
                do_cell_matching=True,
                mode=TableFormerMode.ACCURATE
            ),
            generate_table_images=True,
            generate_picture_images=True,
            images_scale=2.0,  # 2x resolution for better AI classification
            generate_page_images=False,
            do_ocr=False,  # documents are not scanned
        )

        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    @staticmethod
    def _unwrap_gather_results(
        metadata_list: List[Dict],
        gather_results: List,
        required_key: str,
    ) -> List[Tuple[Dict, Dict]]:
        """Filter asyncio.gather results: log exceptions, skip empty, return (metadata, result) pairs (Issue #12 fix)."""
        valid_pairs = []
        for metadata, result in zip(metadata_list, gather_results):
            if isinstance(result, Exception):
                logger.error(f"Async task failed: {result}")
                continue
            if result and result.get(required_key):
                valid_pairs.append((metadata, result))
        return valid_pairs

    async def _extract_and_reconstruct_tables(
        self, document, tables_dir: Path, context: str
    ) -> List[TableAsset]:
        """Reconstruct extracted tables using the Vision model."""
        reconstructed_tables = []

        try:
            docling_tables = list(document.tables)
            logger.info(f"Found {len(docling_tables)} tables via Docling")
            self.stats['tables_docling'] = len(docling_tables)

            if not docling_tables:
                return []

            reconstruction_tasks = []
            table_info_list = []

            for table_idx, table in enumerate(docling_tables):
                page = 1
                if hasattr(table, 'prov') and table.prov and len(table.prov) > 0:
                    page = table.prov[0].page_no

                caption = ""
                if hasattr(table, 'caption_text'):
                    caption = table.caption_text(document) or ""
                elif hasattr(table, 'captions') and table.captions:
                    caption = str(table.captions[0]) if table.captions else ""

                full_context = f"{context} | Page {page} | Table caption: {caption}"

                # Prefer image-based reconstruction when available — it preserves
                # visual layout that HTML export sometimes loses
                if hasattr(table, 'image') and table.image:
                    try:
                        pil_image = table.image.pil_image
                        reconstruction_tasks.append(self.vision.reconstruct_table(pil_image, full_context))
                        table_info_list.append({'page': page, 'source': 'docling_image'})
                        continue
                    except Exception:
                        pass

                # Fallback: reconstruct from HTML export
                try:
                    html_content = table.export_to_html()
                    reconstruction_tasks.append(
                        self.vision.reconstruct_table_from_html(html_content, full_context)
                    )
                    table_info_list.append({'page': page, 'source': 'docling_html'})
                except Exception as e:
                    logger.warning(f"Could not export table {table_idx} to HTML: {e}")

            if not reconstruction_tasks:
                return []

            logger.info(f"Reconstructing {len(reconstruction_tasks)} Docling tables via OpenAI...")
            reconstruction_results = await asyncio.gather(*reconstruction_tasks, return_exceptions=True)

            # Issue #12: centralized gather result filtering
            for table_info, reconstruction_result in self._unwrap_gather_results(
                table_info_list, reconstruction_results, 'markdown'
            ):
                reconstructed_tables.append(TableAsset(
                    page=table_info['page'],
                    markdown_content=reconstruction_result['markdown'],
                    source_type=table_info['source'],
                    description=reconstruction_result.get('description', '')
                ))
                self.stats['tables_reconstructed'] += 1
                logger.info(
                    f"Reconstructed table on page {table_info['page']} "
                    f"via {table_info['source']}"
                )

        except Exception as e:
            logger.error(f"Error extracting Docling tables: {e}", exc_info=True)

        return reconstructed_tables

    async def _extract_and_filter_images(
        self, document, output_dir: Path, tables_dir: Path, context: str
    ) -> Tuple[List[ImageAsset], List[TableAsset]]:
        """Extract figures, filter with AI, and route vision-detected tables separately."""

        min_w, min_h = MIN_IMAGE_SIZE
        max_w, max_h = MAX_IMAGE_SIZE
        image_candidates = []

        try:
            pictures = list(document.pictures)
            logger.info(f"Found {len(pictures)} pictures in document")

            for pic_idx, picture in enumerate(pictures):
                self.stats['found'] += 1

                page_number = 1
                if hasattr(picture, 'prov') and picture.prov and len(picture.prov) > 0:
                    page_number = picture.prov[0].page_no

                try:
                    if hasattr(picture, 'image') and picture.image:
                        pil_image = picture.image.pil_image
                    elif hasattr(picture, 'get_image'):
                        pil_image = picture.get_image(document)
                    else:
                        continue
                except Exception as e:
                    logger.debug(f"Could not load picture {pic_idx}: {e}")
                    continue

                width, height = pil_image.size
                if not (min_w <= width <= max_w and min_h <= height <= max_h):
                    self.stats['filtered_size'] += 1
                    # Issue #10: size-rejected images released immediately
                    del pil_image
                    continue

                caption = ""
                if hasattr(picture, 'caption'):
                    caption = str(picture.caption) if picture.caption else ""

                full_context = f"{context} | Page {page_number} | Caption: {caption}"

                image_candidates.append({
                    'pil_image': pil_image,
                    'filename': f"page_{page_number:03d}_fig_{pic_idx}.png",
                    'context': full_context,
                    'page': page_number,
                })

        except Exception as e:
            logger.error(f"Error extracting pictures: {e}", exc_info=True)

        if not image_candidates:
            logger.info("No images passed size filter")
            return [], []

        logger.info(f"Analyzing {len(image_candidates)} images with AI...")

        pil_images = [c['pil_image'] for c in image_candidates]
        contexts = [c['context'] for c in image_candidates]

        # Surviving candidates are analyzed by the vision model to determine if they are worth keeping
        vision_filter_results = await self.vision.analyze_images_batch_memory(pil_images, contexts)

        # Free the extracted list — individual candidates still hold references
        del pil_images

        kept_images = []
        table_candidates = []

        for candidate, vision_filter_result in zip(image_candidates, vision_filter_results):
            if vision_filter_result.get('is_table'):
                self.stats['tables_vision'] += 1
                table_candidates.append(candidate)
                logger.info(f"{candidate['filename']}: classified as table by vision model")
                continue

            if not vision_filter_result['keep']:
                self.stats['filtered_ai'] += 1
                logger.info(f"{candidate['filename']}: rejected — {vision_filter_result['reason']}")
                # Issue #10: free rejected images promptly
                del candidate['pil_image']
                continue

            final_path = output_dir / candidate['filename']
            pil_to_save = candidate.pop('pil_image')
            pil_to_save.save(final_path, 'PNG')
            img_width, img_height = pil_to_save.size
            # Issue #10: release the PIL image immediately after saving
            del pil_to_save

            image_asset = ImageAsset(
                path=final_path,
                page=candidate['page'],
                width=img_width,
                height=img_height,
                description=vision_filter_result.get('description'),
                entities=vision_filter_result.get('entities')
            )
            kept_images.append(image_asset)
            self.stats['kept'] += 1

            entity_count = len(vision_filter_result.get('entities', []))
            logger.info(f"{candidate['filename']}: kept ({entity_count} entities)")

        # Reconstruct the tables detected by the vision model
        vision_reconstructed_tables = []
        if table_candidates:
            logger.info(f"Reconstructing {len(table_candidates)} vision-detected tables...")
            table_reconstruction_tasks = [
                self.vision.reconstruct_table(c['pil_image'], c['context'])
                for c in table_candidates
            ]
            # Issue #10: free table PIL images after task creation
            for tc in table_candidates:
                del tc['pil_image']

            table_reconstruction_results = await asyncio.gather(*table_reconstruction_tasks, return_exceptions=True)

            # Issue #12: centralized gather result filtering
            table_meta = [{'page': c['page'], 'filename': c['filename']} for c in table_candidates]
            for table_info, reconstruction_result in self._unwrap_gather_results(
                table_meta, table_reconstruction_results, 'markdown'
            ):
                vision_reconstructed_tables.append(TableAsset(
                    page=table_info['page'],
                    markdown_content=reconstruction_result['markdown'],
                    source_type='vision',
                    description=reconstruction_result.get('description', '')
                ))
                self.stats['tables_reconstructed'] += 1
                logger.info(f"Reconstructed table from {table_info['filename']}")

        return kept_images, vision_reconstructed_tables

    def _log_stats(self):
        logger.info(f"\n{'='*50}")
        logger.info(f"  Images found:              {self.stats['found']}")
        logger.info(f"  Images kept:               {self.stats['kept']}")
        logger.info(f"    Filtered (size):         {self.stats['filtered_size']}")
        logger.info(f"    Filtered (AI):           {self.stats['filtered_ai']}")
        logger.info(f"  Tables (Docling native):   {self.stats['tables_docling']}")
        logger.info(f"  Tables (vision-detected):  {self.stats['tables_vision']}")
        logger.info(f"    Tables reconstructed:    {self.stats['tables_reconstructed']}")
        logger.info(f"{'='*50}")
