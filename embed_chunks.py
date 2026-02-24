"""Generate embeddings for all chunks.

Usage:
    python embed_chunks.py [chunks_json_path]

    If no path provided, processes all chunks files in results/
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Optional
import voyageai
from dotenv import load_dotenv

load_dotenv()

from config import VOYAGE_API_KEY, VOYAGE_MODEL, BASE_OUTPUT_DIR, VOYAGE_BATCH_SIZE


def validate_api_key(api_key: Optional[str]) -> voyageai.Client:
    """Validate API key and return an initialized Voyage client."""
    if not api_key:
        print("Error: VOYAGE_API_KEY not found in .env file")
        print("Get your key at: https://www.voyageai.com/")
        sys.exit(1)

    try:
        return voyageai.Client(api_key=api_key)
    except Exception as e:
        print(f"Error initializing Voyage AI client: {e}")
        sys.exit(1)


def load_chunks(chunks_path: Path) -> tuple:
    """Load and validate chunks from a JSON file. Returns (document_id, flat_chunks)."""
    if not chunks_path.exists():
        print(f"Error: File not found: {chunks_path}")
        sys.exit(1)

    try:
        with open(chunks_path) as f:
            raw_chunks_json = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file: {e}")
        sys.exit(1)

    if 'document_id' not in raw_chunks_json:
        print("Error: Missing 'document_id' in chunks file")
        sys.exit(1)

    document_id = raw_chunks_json['document_id']

    flat_chunks = (
        raw_chunks_json.get('text_chunks', []) +
        raw_chunks_json.get('table_chunks', []) +
        raw_chunks_json.get('image_chunks', [])
    )

    if not flat_chunks:
        print("Error: No chunks found in file")
        sys.exit(1)

    return document_id, flat_chunks


def generate_embeddings_batch(
    client: voyageai.Client,
    texts: List[str],
    chunk_ids: List[str],
    model: str,
    batch_size: int = VOYAGE_BATCH_SIZE
) -> List[Dict]:
    """Generate embeddings in batches. Returns list of {chunk_id, embedding} dicts."""
    all_embeddings = []
    total_batches = (len(texts) - 1) // batch_size + 1

    for batch_start in range(0, len(texts), batch_size):
        batch_num = batch_start // batch_size + 1
        text_batch = texts[batch_start:batch_start + batch_size]
        id_batch = chunk_ids[batch_start:batch_start + batch_size]

        try:
            voyage_response = client.embed(
                text_batch,
                model=model,
                input_type="document"
            )

            batch_embedding_vectors = voyage_response.embeddings

            for chunk_id, embedding_vector in zip(id_batch, batch_embedding_vectors):
                all_embeddings.append({
                    'chunk_id': chunk_id,
                    'embedding': embedding_vector
                })

            print(f"    Batch {batch_num}/{total_batches}: {len(text_batch)} chunks")

        except Exception as e:
            print(f"    Batch {batch_num} failed: {e}")
            continue

    return all_embeddings


def save_embeddings(
    embeddings: List[Dict],
    output_path: Path,
    document_id: str,
    model: str
) -> None:
    """Save embeddings to JSON for GraphRAG consumption."""
    if not embeddings:
        print("  No embeddings generated")
        return

    embeddings_payload = {
        'document_id': document_id,
        'embedding_model': model,
        'embedding_provider': 'voyage_ai',
        'total_chunks': len(embeddings),
        'embeddings': embeddings
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(embeddings_payload, f, indent=2)

    print(f"  Saved: {output_path}")
    print(f"  Total: {len(embeddings)} embeddings")


def process_chunks_file(chunks_path: Path, client: voyageai.Client) -> None:
    """Embed all chunks from a single chunks JSON file."""
    print(f"\nProcessing: {chunks_path}")

    document_id, flat_chunks = load_chunks(chunks_path)

    print(f"  Chunks: {len(flat_chunks)}")
    print(f"  Model: {VOYAGE_MODEL}")

    texts = [chunk['content'] for chunk in flat_chunks]
    chunk_ids = [chunk['chunk_id'] for chunk in flat_chunks]

    all_embeddings = generate_embeddings_batch(
        client=client,
        texts=texts,
        chunk_ids=chunk_ids,
        model=VOYAGE_MODEL,
        batch_size=128
    )

    output_path = chunks_path.parent / f"{document_id}_embeddings_voyage.json"
    save_embeddings(
        embeddings=all_embeddings,
        output_path=output_path,
        document_id=document_id,
        model=VOYAGE_MODEL
    )


def find_all_chunks_files(base_dir: str) -> List[Path]:
    """Find all chunks JSON files in the results directory."""
    results_dir = Path(base_dir)
    if not results_dir.exists():
        return []

    discovered_chunk_files = list(results_dir.glob("*/chunks/*_chunks.json"))
    return sorted(discovered_chunk_files)


def main():
    """CLI entry point."""
    client = validate_api_key(VOYAGE_API_KEY)

    if len(sys.argv) > 1:
        chunks_path = Path(sys.argv[1])
        process_chunks_file(chunks_path, client)
    else:
        discovered_chunk_files = find_all_chunks_files(BASE_OUTPUT_DIR)

        if not discovered_chunk_files:
            print(f"Error: No chunks files found in {BASE_OUTPUT_DIR}/*/chunks/")
            print("\nRun chunking first or specify path:")
            print("  python embed_chunks.py path/to/chunks.json")
            sys.exit(1)

        print(f"Found {len(discovered_chunk_files)} chunks file(s)")
        print("=" * 70)

        for chunks_path in discovered_chunk_files:
            process_chunks_file(chunks_path, client)

        print("\n" + "=" * 70)
        print(f"Completed: {len(discovered_chunk_files)} document(s) embedded")


if __name__ == "__main__":
    main()
