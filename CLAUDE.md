# YouTube Summary Tool - Project Guidelines

## Project Overview
This tool downloads YouTube video subtitles and generates AI-powered summaries with interactive Q&A capabilities using a local Ollama instance.

## Stack
- **Language**: Python 3
- **Package Manager**: Nix (shell.nix)
- **Dependencies**:
  - yt-dlp (YouTube subtitle download)
  - python3Packages.requests (Ollama API communication)
- **AI Model**: qwen2.5:7b-instruct (via Ollama at localhost:11434)

## Key Features
- Auto-detects human vs auto-generated subtitles
- Downloads and converts to SRT format
- Generates factual summaries with hallucination prevention
- Interactive Q&A mode with conversation context
- `--load` flag to process existing .srt files

## AI Configuration
**Critical**: The tool is optimized for accuracy over creativity:
- Temperature: 0.2 (summaries), 0.3 (Q&A)
- top_p: 0.4 (narrow probability distribution)
- top_k: 10 (limit token choices)
- Explicit grounding instructions to prevent hallucinations

## Development Principles
1. **Accuracy First** - All prompts emphasize using ONLY transcript information
2. **No Premature Optimization** - Test real-world usage before adding complexity
3. **Minimal Dependencies** - Keep the tool simple and self-contained

## Known Limitations (intentional - to be addressed after real-world testing)
- No chunking for very long videos (may hit token limits)
- No visual context (subtitle-only analysis)
- Auto-generated subtitle quality varies by video
- May struggle with heavy sarcasm/irony (AI takes text literally)

## When Making Changes
- **DO NOT** add features preemptively
- **DO** test on diverse video types before modifying
- **DO** maintain low temperature settings for factual accuracy
- **DO** keep prompts explicit about grounding in transcript
- **DO NOT** compromise accuracy for creativity

## File Structure
- `download_subs.py` - Main script
- `shell.nix` - Nix environment with dependencies
- `*.en.srt` - Downloaded subtitle files (gitignored)
- `*.srt` - Any subtitle files

## Usage Patterns
```bash
# Download and summarize
python download_subs.py <youtube-url>
python download_subs.py  # prompts for URL

# Load existing subtitle
python download_subs.py --load
```

## Testing Strategy
Before adding features, test on:
- Short videos (< 10 min)
- Medium videos (10-30 min)
- Long videos (30min - 2hr)
- Technical content
- Casual conversation
- Educational lectures
- Product reviews

Document what breaks, then fix based on real issues.
