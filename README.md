# YouTube Summary AI

AI-powered YouTube video summarizer with RAG-enhanced Q&A using local Ollama models.

## Features

- 🎯 Semantic chunking with overlap for accurate context preservation
- 🔍 Vector embedding retrieval using sentence-transformers
- 💬 Interactive Q&A with conversation history
- 🌐 Modern web interface with streaming responses
- 📊 CLI tool for quick summaries
- 🔒 Privacy-focused: All processing happens locally

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
    ollamaUrl = "http://localhost:11434";
    retention.maxAge = "30d";  # Auto-cleanup old files
  };
}
```

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

- Ollama running with `qwen2.5:7b-instruct` model
- NixOS with flakes enabled

## License

MIT
