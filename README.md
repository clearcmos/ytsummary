# YouTube Summary AI

AI-powered YouTube video summarizer with RAG-enhanced Q&A using local Ollama models.

## Features

- Semantic chunking with overlap for accurate context preservation
- Advanced conversational RAG with history-aware query reformulation
- Hybrid search retrieval (BM25 + semantic embeddings) with query expansion
- Interactive Q&A that understands follow-up questions
- Modern web interface with streaming responses
- CLI tool for quick summaries
- Privacy-focused: All processing happens locally

## Installation

### NixOS Module (Recommended)

Add to your flake inputs:

```nix
{
  inputs.ytsummary.url = "github:clearmos/ytsummary";
}
```

Enable the service:

```nix
{
  imports = [ inputs.ytsummary.nixosModules.default ];

  services.ytsummary = {
    enable = true;
    port = 8000;

    # Model configuration (optional)
    model = "qwen2.5:7b-instruct";  # Default model
    autoConfigureOllama = true;     # Auto-setup Ollama (default: true)

    # Other options
    retention.maxAge = "30d";  # Auto-cleanup old files
  };
}
```

**Zero-config setup**: With `autoConfigureOllama = true` (default), the module automatically:
- Enables Ollama service
- Downloads and loads the specified model
- No manual Ollama configuration needed!

### Nix Package

```bash
nix run github:clearmos/ytsummary
```

## Usage

### Web Interface
Navigate to `http://localhost:8000` after starting the service.

### CLI
```bash
ytsummary <youtube-url>
ytsummary --load  # Load existing subtitle
```

## Requirements

- NixOS with flakes enabled
- **That's it!** The module handles Ollama setup automatically

### Manual Ollama Setup (if autoConfigureOllama = false)

```bash
# If you disable auto-configuration, install the model manually:
ollama pull qwen2.5:7b-instruct
```

## License

MIT
