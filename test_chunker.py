"""Verification test for the improved chunker against real Docling output."""

import sys, json
sys.path.insert(0, '.')

from pathlib import Path
from chunker import DocumentChunker
from models import ImageAsset, TableAsset

# Load real markdown
md_path = Path('results/docling_output/markdown/docling.md')
markdown = md_path.read_text()
print(f'Loaded markdown: {len(markdown)} chars, {len(markdown.splitlines())} lines')

# Real tables from the pipeline
tables = [
    TableAsset(
        page=5,
        markdown_content="""| CPU | Thread budget | native backend | native backend | native backend | pypdfium backend | pypdfium backend | pypdfium backend |
|-------------------------|-----------------|------------------|------------------|------------------|--------------------|--------------------|-------------------|
| | | TTS | Pages/s | Mem | TTS | Pages/s | Mem |
| Apple M3 Max (16 cores) | 4 16 | 177 s 167 s | 1. 1. | 6. GB | 103 s 92 s | 2. 2. | 2. GB |
| Intel(R) Xeon E5-2690 | 4 16 | 375 s 244 s | 0. 0. | 6. GB | 239 s 143 s | 0. 1. | 2. GB |""",
        source_type='docling',
        description='Runtime characteristics of Docling on two CPU systems.'
    ),
    TableAsset(
        page=8,
        markdown_content="""| | human | MRCNN R50 R101 | FRCNN R101 | YOLO v5x6 |
|---|---|---|---|---|
| Caption | 84-89 | 68.4 71.3 | 70.1 | 77.2 |
| All | 82-83 | 72.4 73.4 | 73.4 | 76.8 |""",
        source_type='docling_image',
        description='Prediction performance of object detection networks on DocLayNet.'
    ),
]

# Real images
images = [
    ImageAsset(
        path=Path('results/docling_output/images/page_003_fig_1.png'),
        page=3, width=786, height=278,
        description='Flowchart illustrating a model pipeline for processing PDF documents.',
        entities=[{'type': 'process', 'name': 'Parse PDF pages'}, {'type': 'process', 'name': 'OCR'}]
    ),
]

# Run chunker with default config (MIN_CHUNK_SIZE=200 now)
chunker = DocumentChunker()
result = chunker.chunk_document(
    document_id='docling',
    markdown_content=markdown,
    tables=tables,
    images=images,
)

print(f'\n=== RESULTS ===')
print(f'Total chunks: {result.total_chunks}')
print(f'Text: {len(result.text_chunks)}, Tables: {len(result.table_chunks)}, Images: {len(result.image_chunks)}')
print(f'Stats: {json.dumps(result.stats, indent=2)}')

# Check for quality issues
issues = []

for i, tc in enumerate(result.text_chunks):
    content = tc.content
    # Issue: starts with period
    if content.startswith('.') or content.startswith(','):
        issues.append(f'  FRAGMENT START chunk {i} [{tc.chunk_id}]: starts with "{content[:40]}..."')
    # Issue: too small
    if tc.char_count < 200:
        issues.append(f'  SMALL CHUNK {i} [{tc.chunk_id}]: only {tc.char_count} chars')
    # Issue: contains inline table rows
    table_lines = [l for l in content.split('\n') if l.strip().startswith('|') and l.strip().endswith('|')]
    if table_lines:
        issues.append(f'  INLINE TABLE chunk {i} [{tc.chunk_id}]: {len(table_lines)} table rows found')

print(f'\n=== QUALITY CHECK ===')
if issues:
    print(f'Found {len(issues)} issues:')
    for issue in issues:
        print(issue)
else:
    print('No quality issues found!')

print(f'\n=== CHUNK DETAILS ===')
for i, tc in enumerate(result.text_chunks):
    print(f'\n--- Chunk {i} [{tc.chunk_id}] ---')
    print(f'  Section: {" > ".join(tc.section_path)} (level={tc.section_level})')
    print(f'  Chars: {tc.char_count}')
    first_line = tc.content.split('\n')[0][:100]
    last_line = tc.content.split('\n')[-1][:100]
    print(f'  Start: {first_line}...')
    print(f'  End:   ...{last_line}')

# Save to verify JSON
import tempfile
with tempfile.TemporaryDirectory() as tmpdir:
    path = chunker.save_chunks(result, Path(tmpdir))
    saved_data = json.loads(path.read_text())
    print(f'\nSerialization: OK ({len(json.dumps(saved_data))} bytes)')

print('\n=== DONE ===')
