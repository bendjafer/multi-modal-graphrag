"""Main script for batch PDF processing."""

import logging
import warnings
from pathlib import Path

from tqdm import tqdm

from config import LOG_FORMAT, LOG_LEVEL, INPUT_PDF_FOLDER, BASE_OUTPUT_DIR
from pdf_processor import PDFProcessor

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def process_pdf_folder(input_folder: Path, output_folder: Path):
    """Process all PDFs in a folder using a single shared processor instance."""
    pdf_files = list(input_folder.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in {input_folder}")
        return

    logger.info(f"Found {len(pdf_files)} PDF files to process")

    processor = PDFProcessor(output_base=output_folder)

    for pdf_path in tqdm(pdf_files, desc="Processing PDFs", unit="pdf"):
        try:
            logger.info(f"\n{'='*70}")
            logger.info(f"Processing: {pdf_path.name}")
            logger.info(f"{'='*70}")

            processing_result = processor.process(
                pdf_path=pdf_path,
                context=f"Document: {pdf_path.stem}"
            )

            logger.info(f"  Processed {processing_result.total_pages} pages")
            logger.info(f"  Kept {len(processing_result.images)} images")
            logger.info(f"  Saved to {processing_result.markdown_path.parent.parent}")

        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {e}", exc_info=True)
            continue

    logger.info(f"\n{'='*70}")
    logger.info("Batch processing complete!")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    INPUT_FOLDER = Path(INPUT_PDF_FOLDER)
    OUTPUT_FOLDER = Path(BASE_OUTPUT_DIR)

    OUTPUT_FOLDER.mkdir(exist_ok=True)

    process_pdf_folder(INPUT_FOLDER, OUTPUT_FOLDER)
