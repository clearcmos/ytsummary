#!/usr/bin/env python3
import subprocess
import sys
import re
import json
import requests
import glob
import os
import argparse
import numpy as np
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live

def check_human_subs(url):
    """Check if human English subtitles exist."""
    result = subprocess.run(
        ['yt-dlp', '--list-subs', url],
        capture_output=True,
        text=True
    )

    lines = result.stdout.split('\n')
    in_subs = False

    for line in lines:
        if 'Available subtitles' in line:
            in_subs = True
            continue
        if 'Available automatic captions' in line:
            in_subs = False
            break

        if in_subs and re.match(r'^en(-|$)', line.strip().split()[0] if line.strip().split() else ''):
            return True

    return False

def clean_srt_text(srt_path):
    """Extract clean text from SRT file, removing timestamps and numbering."""
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Remove subtitle numbers and timestamps
    lines = content.split('\n')
    text_lines = []
    for line in lines:
        line = line.strip()
        # Skip empty lines, numbers, and timestamp lines
        if line and not line.isdigit() and '-->' not in line:
            text_lines.append(line)

    return ' '.join(text_lines)

def chunk_text_semantic(text, chunk_size=500, overlap=100):
    """
    Chunk text into overlapping segments for better context preservation.
    Uses sentence boundaries for semantic coherence.

    Args:
        text: Full transcript text
        chunk_size: Target tokens per chunk (approximate)
        overlap: Number of tokens to overlap between chunks

    Returns:
        List of text chunks with metadata
    """
    # Split into sentences (rough approximation)
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = []
    current_length = 0
    chunk_start_idx = 0

    for i, sentence in enumerate(sentences):
        sentence_length = len(sentence.split())  # Approximate token count

        # If adding this sentence exceeds chunk_size, save current chunk
        if current_length + sentence_length > chunk_size and current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'start_idx': chunk_start_idx,
                'end_idx': i - 1,
                'token_count': current_length
            })

            # Create overlap by keeping last sentences
            overlap_sentences = []
            overlap_length = 0
            for sent in reversed(current_chunk):
                sent_len = len(sent.split())
                if overlap_length + sent_len <= overlap:
                    overlap_sentences.insert(0, sent)
                    overlap_length += sent_len
                else:
                    break

            current_chunk = overlap_sentences
            current_length = overlap_length
            chunk_start_idx = i - len(overlap_sentences)

        current_chunk.append(sentence)
        current_length += sentence_length

    # Add final chunk
    if current_chunk:
        chunks.append({
            'text': ' '.join(current_chunk),
            'start_idx': chunk_start_idx,
            'end_idx': len(sentences) - 1,
            'token_count': current_length
        })

    return chunks

# Global embedding model (lazy loaded)
_embedding_model = None

def get_embedding_model():
    """Lazy load the sentence transformer model."""
    global _embedding_model
    if _embedding_model is None:
        print("Loading embedding model (first time only)...")
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, lightweight model
    return _embedding_model

def retrieve_relevant_chunks(query, chunks, top_k=3):
    """
    Retrieve most relevant chunks for a query using semantic similarity.

    Args:
        query: User's question
        chunks: List of chunk dictionaries with 'text' and metadata
        top_k: Number of top chunks to return

    Returns:
        List of most relevant chunk texts
    """
    model = get_embedding_model()

    # Embed query and chunks
    query_embedding = model.encode([query])
    chunk_texts = [chunk['text'] for chunk in chunks]
    chunk_embeddings = model.encode(chunk_texts)

    # Calculate cosine similarity
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]

    # Get top-k indices
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    # Return top chunks with similarity scores (for debugging)
    relevant_chunks = []
    for idx in top_indices:
        relevant_chunks.append({
            'text': chunks[idx]['text'],
            'similarity': similarities[idx],
            'chunk_idx': idx
        })

    return relevant_chunks

def ask_ollama(prompt, context=None, temperature=0.3, stream=False):
    """Send a prompt to Ollama API with optional context and streaming support."""
    try:
        full_prompt = prompt
        if context:
            full_prompt = f"{context}\n\n{prompt}"

        # Get model and host from environment or use defaults
        model = os.environ.get('YTSUMMARY_MODEL', 'qwen2.5:7b-instruct')
        ollama_host = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')

        payload = {
            'model': model,
            'prompt': full_prompt,
            'stream': stream,
            'options': {
                'temperature': temperature,  # Low temp for factual accuracy
                'top_p': 0.4,  # Narrow probability distribution
                'top_k': 10,  # Limit token choices for consistency
                'repeat_penalty': 1.1  # Prevent repetition
            }
        }

        if stream:
            # Streaming mode
            response = requests.post(
                f'{ollama_host}/api/generate',
                json=payload,
                stream=True,
                timeout=120
            )
            response.raise_for_status()

            full_content = ""
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode('utf-8'))
                        if 'response' in chunk:
                            content_piece = chunk['response']
                            full_content += content_piece
                            yield content_piece, full_content
                        if chunk.get('done', False):
                            break
                    except json.JSONDecodeError:
                        continue
        else:
            # Non-streaming mode (backward compatible)
            response = requests.post(
                f'{ollama_host}/api/generate',
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            return result.get('response', 'No response generated')

    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Ollama: {e}")
        if stream:
            yield None, None
        else:
            return None

def summarize_with_ollama(text, title, chunks):
    """Send text to Ollama API for summarization using multi-stage RAG with streaming."""
    console = Console()
    console.print("\n[cyan]Generating summary with Ollama (streaming)...[/cyan]\n")

    # Stage 1: Use full text for short videos, or sample chunks for long ones
    if len(text.split()) < 2000:
        # Short video - use full text
        context_text = text
    else:
        # Long video - use strategic sampling
        # Get first, middle, and last chunks for overview
        context_chunks = [
            chunks[0],  # Introduction
            chunks[len(chunks)//2],  # Middle
            chunks[-1]  # Conclusion
        ]
        context_text = "\n\n---SECTION BREAK---\n\n".join([c['text'] for c in context_chunks])

    prompt = f"""Create an accurate summary of the YouTube video titled "{title}" based ONLY on the provided transcript sections below.

CRITICAL RULES:
1. Use ONLY information explicitly stated in the transcript sections
2. Pay special attention to negative statements ("not for", "don't recommend", "avoid", "not recommended")
3. Include specific numbers, prices, and technical details EXACTLY as mentioned
4. If something is unclear in the transcript, state that it's unclear
5. Do NOT add information from general knowledge about this topic

TRANSCRIPT SECTIONS:
{context_text}

Write a concise summary covering:
- Main topic and purpose
- Key points with specific details from the transcript
- Important recommendations, warnings, or conclusions (especially negative statements)
- Any caveats or limitations the speaker mentions

Base your entire summary on the transcript sections above. Do not add external information."""

    # Stream and render markdown in real-time
    full_response = ""
    with Live(Markdown(""), console=console, refresh_per_second=10) as live:
        for chunk, accumulated in ask_ollama(prompt, temperature=0.2, stream=True):
            if chunk is None:
                return None
            full_response = accumulated
            # Update live display with rendered markdown
            live.update(Markdown(accumulated))
            time.sleep(0.01)  # Small delay for smooth rendering

    console.print()  # Newline after streaming completes
    return full_response

def interactive_qa(subtitle_text, video_title, summary, chunks):
    """Interactive Q&A session about the video using RAG retrieval."""
    console = Console()
    console.print("\n" + "="*60)
    console.print("[cyan]Q&A MODE - Ask questions about the video (type 'quit' to exit)[/cyan]")
    console.print("="*60 + "\n")

    conversation_history = []

    while True:
        try:
            question = input("\nYour question: ").strip()

            if question.lower() in ['quit', 'exit', 'q']:
                console.print("[yellow]Exiting Q&A mode.[/yellow]")
                break

            if not question:
                continue

            # Stage 1: Retrieve relevant chunks using semantic search
            if len(chunks) > 5:
                # Use RAG for long videos
                relevant_chunks = retrieve_relevant_chunks(question, chunks, top_k=3)
                retrieved_text = "\n\n---RELEVANT SECTION---\n\n".join([c['text'] for c in relevant_chunks])

                # Debug: Show which chunks were retrieved
                console.print(f"[dim][Retrieved {len(relevant_chunks)} relevant sections from {len(chunks)} total chunks][/dim]")
            else:
                # Short video - use full text
                retrieved_text = subtitle_text

            # Build context with retrieved information
            context = f"""Answer questions about the YouTube video "{video_title}" using ONLY the retrieved transcript sections below.

CRITICAL RULES:
1. Answer ONLY from the retrieved transcript sections below
2. If the answer isn't in these sections, say "This information is not in the video sections I can access"
3. Pay attention to exact wording, especially negative statements ("not for", "not recommended", "avoid")
4. Quote or paraphrase specific details from the transcript
5. Do NOT use general knowledge about this topic

RETRIEVED TRANSCRIPT SECTIONS:
{retrieved_text}

SUMMARY (for reference):
{summary}

PREVIOUS CONVERSATION:
{chr(10).join(conversation_history) if conversation_history else 'None'}

Answer the question using only the transcript sections above:"""

            # Stream the answer with rendered markdown
            console.print("\n[green]Answer:[/green] ", end='')
            full_answer = ""
            with Live(Markdown(""), console=console, refresh_per_second=10, auto_refresh=True) as live:
                for chunk, accumulated in ask_ollama(question, context, stream=True):
                    if chunk is None:
                        console.print("\n[red]Failed to get an answer. Please try again.[/red]")
                        break
                    full_answer = accumulated
                    # Update live display with rendered markdown
                    live.update(Markdown(accumulated))
                    time.sleep(0.01)

            if full_answer:
                console.print()  # Newline after streaming
                # Store in history for context
                conversation_history.append(f"Q: {question}")
                conversation_history.append(f"A: {full_answer}")
                # Keep only last 4 exchanges to avoid token overflow
                if len(conversation_history) > 8:
                    conversation_history = conversation_history[-8:]

        except KeyboardInterrupt:
            console.print("\n\n[yellow]Exiting Q&A mode.[/yellow]")
            break
        except EOFError:
            console.print("\n\n[yellow]Exiting Q&A mode.[/yellow]")
            break

def download_subtitles(url):
    """Download subtitles - prefer human, fallback to auto-generated."""
    # Get video title first
    title_result = subprocess.run(
        ['yt-dlp', '--get-title', url],
        capture_output=True,
        text=True
    )
    video_title = title_result.stdout.strip()

    if check_human_subs(url):
        # Human English exists - download original
        print("Human English subtitles found, downloading...")
        subprocess.run([
            'yt-dlp',
            '--skip-download',
            '--write-subs',
            '--sub-langs', 'en-orig',
            '--convert-subs', 'srt',
            '-o', '%(title)s.%(ext)s',
            url
        ])
    else:
        # No human English - fallback to auto-generated
        print("No human English subtitles, downloading auto-generated...")
        subprocess.run([
            'yt-dlp',
            '--skip-download',
            '--write-auto-subs',
            '--sub-langs', 'en',
            '--convert-subs', 'srt',
            '-o', '%(title)s.%(ext)s',
            url
        ])

    return video_title

def load_existing_srt():
    """List and select from existing SRT files."""
    srt_files = sorted(glob.glob('*.srt'))

    if not srt_files:
        print("No .srt files found in current directory.")
        sys.exit(1)

    print("\nAvailable subtitle files:")
    for idx, filename in enumerate(srt_files, 1):
        print(f"{idx}. {filename}")

    while True:
        try:
            choice = input("\nSelect file number: ").strip()
            file_idx = int(choice) - 1

            if 0 <= file_idx < len(srt_files):
                selected_file = srt_files[file_idx]
                # Extract title from filename (remove .en.srt or .srt)
                title = selected_file.replace('.en.srt', '').replace('.srt', '')
                return selected_file, title
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
        except (KeyboardInterrupt, EOFError):
            print("\n\nCancelled.")
            sys.exit(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download and summarize YouTube video subtitles')
    parser.add_argument('--load', action='store_true', help='Load an existing .srt file instead of downloading')
    parser.add_argument('url', nargs='?', help='YouTube URL to download')

    args = parser.parse_args()

    if args.load:
        srt_file, video_title = load_existing_srt()
    else:
        if args.url:
            url = args.url
        else:
            url = input("Enter YouTube URL: ")

        video_title = download_subtitles(url)

        # Find the downloaded SRT file
        srt_files = glob.glob('*.en.srt')
        if not srt_files:
            print("Error: Could not find downloaded subtitle file")
            sys.exit(1)

        # Use the most recently created file
        srt_file = max(srt_files, key=os.path.getctime)

    # Extract and clean text
    subtitle_text = clean_srt_text(srt_file)

    # Create semantic chunks with overlap
    print(f"Creating semantic chunks from transcript ({len(subtitle_text.split())} words)...")
    chunks = chunk_text_semantic(subtitle_text, chunk_size=500, overlap=100)
    print(f"Created {len(chunks)} overlapping chunks")

    # Generate summary
    summary = summarize_with_ollama(subtitle_text, video_title, chunks)

    if summary:
        print("\n" + "="*60)
        print(f"SUMMARY: {video_title}")
        print("="*60)
        print(summary)
        print("="*60 + "\n")

        # Start interactive Q&A with RAG
        interactive_qa(subtitle_text, video_title, summary, chunks)
