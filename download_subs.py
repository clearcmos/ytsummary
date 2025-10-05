#!/usr/bin/env python3
import subprocess
import sys
import re
import json
import requests
import glob
import os
import argparse

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

def ask_ollama(prompt, context=None, temperature=0.3):
    """Send a prompt to Ollama API with optional context."""
    try:
        full_prompt = prompt
        if context:
            full_prompt = f"{context}\n\n{prompt}"

        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'qwen2.5:7b-instruct',
                'prompt': full_prompt,
                'stream': False,
                'options': {
                    'temperature': temperature,  # Low temp for factual accuracy
                    'top_p': 0.4,  # Narrow probability distribution
                    'top_k': 10,  # Limit token choices for consistency
                    'repeat_penalty': 1.1  # Prevent repetition
                }
            },
            timeout=120
        )
        response.raise_for_status()

        result = response.json()
        return result.get('response', 'No response generated')

    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Ollama: {e}")
        return None

def summarize_with_ollama(text, title):
    """Send text to Ollama API for summarization."""
    print("\nGenerating summary with Ollama...")

    prompt = f"""You are tasked with creating an accurate, factual summary of a YouTube video based ONLY on the provided subtitle transcript.

CRITICAL INSTRUCTIONS:
- Base your summary ONLY on information explicitly stated in the subtitles
- If you're unsure about something, do NOT guess or infer
- Quote specific numbers, prices, and technical details exactly as mentioned
- If the speaker makes a joke or correction, note the correction, not the joke
- Do not add information from your general knowledge

Video Title: "{title}"

Provide a concise summary covering:
- Main topic/purpose
- Key points discussed (with specific details from transcript)
- Important takeaways or conclusions

SUBTITLE TRANSCRIPT:
{text}

Remember: Only use information directly from the transcript above. If something is unclear in the subtitles, mention that it's unclear."""

    return ask_ollama(prompt, temperature=0.2)  # Very low temp for summarization

def interactive_qa(subtitle_text, video_title, summary):
    """Interactive Q&A session about the video."""
    print("\n" + "="*60)
    print("Q&A MODE - Ask questions about the video (type 'quit' to exit)")
    print("="*60 + "\n")

    conversation_history = []

    while True:
        try:
            question = input("\nYour question: ").strip()

            if question.lower() in ['quit', 'exit', 'q']:
                print("Exiting Q&A mode.")
                break

            if not question:
                continue

            # Build context with video info and conversation history
            context = f"""You are answering questions about a YouTube video titled "{video_title}".

CRITICAL INSTRUCTIONS:
- Answer ONLY based on information in the subtitle transcript below
- If the answer is not in the subtitles, say "This information is not mentioned in the video"
- Quote specific details from the transcript when possible
- Do NOT use your general knowledge or make assumptions
- If something is ambiguous, explain what the subtitles actually say

SUBTITLE TRANSCRIPT:
{subtitle_text}

SUMMARY (for reference):
{summary}

PREVIOUS CONVERSATION:
{chr(10).join(conversation_history) if conversation_history else 'None'}

Answer the following question based ONLY on the video transcript above:"""

            answer = ask_ollama(question, context)

            if answer:
                print(f"\nAnswer: {answer}")
                # Store in history for context
                conversation_history.append(f"Q: {question}")
                conversation_history.append(f"A: {answer}")
                # Keep only last 4 exchanges to avoid token overflow
                if len(conversation_history) > 8:
                    conversation_history = conversation_history[-8:]
            else:
                print("Failed to get an answer. Please try again.")

        except KeyboardInterrupt:
            print("\n\nExiting Q&A mode.")
            break
        except EOFError:
            print("\n\nExiting Q&A mode.")
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

    # Generate summary
    summary = summarize_with_ollama(subtitle_text, video_title)

    if summary:
        print("\n" + "="*60)
        print(f"SUMMARY: {video_title}")
        print("="*60)
        print(summary)
        print("="*60 + "\n")

        # Start interactive Q&A
        interactive_qa(subtitle_text, video_title, summary)
