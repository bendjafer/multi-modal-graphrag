"""Generate embeddings for all chunks.
Processes all chunks files in results directory.
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

from config import VOYAGE_API_KEY, VOYAGE_MODEL, BASE_OUTPUT_DIR


def validate_api_key(api_key: Optional[str]) -> voyageai.Client:
    """Validate API key and return client."""
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
    """Load and validate chunks from JSON file."""
    if not chunks_path.exists():
        print(f"Error: File not found: {chunks_path}")
        sys.exit(1)

    try:
        with open(chunks_path) as f:
            chunks_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file: {e}")
        sys.exit(1)

    if 'document_id' not in chunks_data:
        print("Error: Missing 'document_id' in chunks file")
        sys.exit(1)

    document_id = chunks_data['document_id']

    all_chunks = (
        chunks_data.get('text_chunks', []) + 
        chunks_data.get('table_chunks', []) + 
        chunks_data.get('image_chunks', [])
    )

    if not all_chunks:
        print("Error: No chunks found in file")
        sys.exit(1)

    return document_id, all_chunks


def generate_embeddings_batch(
    client: voyageai.Client,
    texts: List[str],
    chunk_ids: List[str],
    model: str,
    batch_size: int = 128
) -> List[Dict]:
    """Generate embeddings in batches."""
    embeddings = []
    total_batches = (len(texts) - 1) // batch_size + 1

    for i in range(0, len(texts), batch_size):
        batch_num = i // batch_size + 1
        batch_texts = texts[i:i + batch_size]
        batch_ids = chunk_ids[i:i + batch_size]

        try:
            result = client.embed(
                batch_texts,
                model=model,
                input_type="document"
            )

            batch_embeddings = result.embeddings

            for chunk_id, embedding in zip(batch_ids, batch_embeddings):
                embeddings.append({
                    'chunk_id': chunk_id,
                    'embedding': embedding
                })

            print(f"    Batch {batch_num}/{total_batches}: {len(batch_texts)} chunks")

        except Exception as e:
            print(f"    Batch {batch_num} failed: {e}")
            continue

    return embeddings


def save_embeddings(
    embeddings: List[Dict],
    output_path: Path,
    document_id: str,
    model: str
) -> None:
    """Save embeddings for GraphRAG."""
    if not embeddings:
        print("  No embeddings generated")
        return

    output_data = {
        'document_id': document_id,
        'embedding_model': model,
        'embedding_provider': 'voyage_ai',
        'total_chunks': len(embeddings),
        'embeddings': embeddings
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"  Saved: {output_path}")
    print(f"  Total: {len(embeddings)} embeddings")


def process_chunks_file(chunks_path: Path, client: voyageai.Client) -> None:
    """Process a single chunks file."""
    print(f"\nProcessing: {chunks_path}")

    document_id, all_chunks = load_chunks(chunks_path)

    print(f"  Chunks: {len(all_chunks)}")
    print(f"  Model: {VOYAGE_MODEL}")

    texts = [chunk['content'] for chunk in all_chunks]
    chunk_ids = [chunk['chunk_id'] for chunk in all_chunks]

    embeddings = generate_embeddings_batch(
        client=client,
        texts=texts,
        chunk_ids=chunk_ids,
        model=VOYAGE_MODEL,
        batch_size=128
    )

    output_path = chunks_path.parent / f"{document_id}_embeddings_voyage.json"
    save_embeddings(
        embeddings=embeddings,
        output_path=output_path,
        document_id=document_id,
        model=VOYAGE_MODEL
    )


def find_all_chunks_files(base_dir: str) -> List[Path]:
    """Find all chunks files in results directory."""
    results_path = Path(base_dir)
    if not results_path.exists():
        return []

    chunks_files = list(results_path.glob("*/chunks/*_chunks.json"))
    return sorted(chunks_files)


def main():
    """CLI entry point."""
    client = validate_api_key(VOYAGE_API_KEY)

    # Use command-line argument if provided (The argument is the path to the chunks file)

    if len(sys.argv) > 1:
        chunks_path = Path(sys.argv[1])
        process_chunks_file(chunks_path, client)
    else:
        # Process all chunks files in results directory
        chunks_files = find_all_chunks_files(BASE_OUTPUT_DIR)

        if not chunks_files:
            print(f"Error: No chunks files found in {BASE_OUTPUT_DIR}/*/chunks/")
            print("\nRun chunking first or specify path:")
            print("  python embed_chunks.py path/to/chunks.json")
            sys.exit(1)

        print(f"Found {len(chunks_files)} chunks file(s)")
        print("="*70)

        for chunks_path in chunks_files:
            process_chunks_file(chunks_path, client)

        print("\n" + "="*70)
        print(f"Completed: {len(chunks_files)} document(s) embedded")


if __name__ == "__main__":
    main()
