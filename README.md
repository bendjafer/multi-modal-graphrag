# Multi-Modal GraphRAG

PDF → Clean Markdown + AI-filtered images + reconstructed tables — ready for GraphRAG pipelines.

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure
cp .env.example .env   # or edit .env directly
# Set OPENAI_API_KEY=sk-your-key

# 3. Run
python process.py
```

PDFs go in `dummy_pdfs/` → output lands in `results/{doc_id}_output/`.

## How It Works

```
PDF ──→ Docling ──→ raw markdown + images
                         │            │
                         │      size filter → AI classifier
                         │            │          │         │
                         │         KEEP(1)   TABLE(2)   REJECT(0)
                         │            │          │
                         │         save PNG   reconstruct
                         │            │       as markdown
                         ▼            ▼          ▼
                    10-step clean ←── image descriptions
                         │
                    final .md + metadata.json + tables/
```

**AI classifies each image into 3 categories:**

| Decision | Action | Output |
|----------|--------|--------|
| `1` KEEP | Generate description + entities | `images/*.png` |
| `2` TABLE | Reconstruct via OpenAI | `tables/table_*.md` |
| `0` REJECT | Discard | — |

## Files

| File | Role |
|------|------|
| `process.py` | Entry point — batch processes PDFs |
| `pdf_processor.py` | Orchestrator — conversion, filtering, output |
| `vision_model.py` | OpenAI vision API — classify images, reconstruct tables |
| `markdown_processor.py` | 10-step markdown cleaning pipeline |
| `models.py` | Data classes: `ImageAsset`, `TableAsset`, `ProcessingResult` |
| `config.py` | All settings via env vars / `.env` |

## Output Structure

```
results/{doc_id}_output/
├── markdown/
│   ├── {doc_id}_raw.md      # Docling's raw output
│   └── {doc_id}.md          # Cleaned + enriched
├── images/                  # AI-approved PNGs only
├── tables/                  # Reconstructed table markdown
└── {doc_id}_metadata.json   # Full processing stats
```

## Configuration

All settings live in `.env` (see `.env` for defaults):

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENAI_API_KEY` | *required* | OpenAI API access |
| `VISION_MODEL` | `gpt-4o-mini` | Vision classification model |
| `INPUT_PDF_FOLDER` | `dummy_pdfs` | Source PDF directory |
| `BASE_OUTPUT_DIR` | `results` | Output root |
| `MAX_CONCURRENT_REQUESTS` | `10` | Parallel API calls |
| `ENABLE_VISION_CACHE` | `true` | Cache API responses (needs `diskcache`) |
| `MIN_IMAGE_WIDTH/HEIGHT` | `100` | Minimum image dimensions |
| `REMOVE_CITATION_NUMBERS` | `true` | Strip academic citations |

**Unused but configured** (for future pipeline stages):

| Variable | Model | Planned Use |
|----------|-------|-------------|
| `ENTITY_EXTRACTION_MODEL` | `sciphi/triplex` | Text entity extraction |
| `SUMMARY_MODEL` | `qwen2.5:14b` | Document summarization |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Vector embeddings |
| `OLLAMA_HOST` | `localhost:11434` | Local model backend |

## Development Workflow

### Adding a new pipeline stage

1. Create module (e.g. `chunker.py`)
2. Add config vars to `config.py`
3. Call from `pdf_processor.py` after `markdown_processor.pipeline()`
4. Add output data class to `models.py` if needed

### Modifying the vision prompt

> ⚠️ **High risk** — the prompt format and `_parse_response()` regex parser are tightly coupled. If you change the prompt output format, update the parser.

### Markdown cleaning

`MarkdownProcessor.pipeline()` runs 10 steps in order. To add a new cleaning step, insert it in the `pipeline()` method at the appropriate position — earlier = pre-structural, later = post-structural.

## Not Yet Implemented

- [ ] Text chunking (section-aware markers already injected: `<!-- section:id=... -->`)
- [ ] Entity extraction from text
- [ ] Knowledge graph construction
- [ ] Embedding generation
- [ ] Vector store + retrieval
- [ ] Query / generation loop

## Dependencies

**Core:** `docling`, `openai`, `Pillow`, `ftfy`, `markdowncleaner`, `tqdm`, `python-dotenv`
**Optional:** `diskcache` (API caching), `langdetect` (language detection), `imagehash` (perceptual hashing)
**Unused:** `easyocr`, `pydantic`, `aiohttp`, `mdformat`
