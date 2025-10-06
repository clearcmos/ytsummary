# Usage Guide

## Quick Start

### 1. Add to Your NixOS Configuration

```nix
# flake.nix
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    ytsummary.url = "github:clearcmos/ytsummary";
  };

  outputs = { nixpkgs, ytsummary, ... }: {
    nixosConfigurations.yourhost = nixpkgs.lib.nixosSystem {
      modules = [
        ytsummary.nixosModules.default
        {
          services.ytsummary = {
            enable = true;
            port = 8000;

            # Model configuration (optional - uses defaults)
            model = "qwen2.5:7b-instruct";  # Default model
            autoConfigureOllama = true;     # Auto-setup Ollama (default: true)
          };
        }
      ];
    };
  };
}
```

### 2. Rebuild and Start

```bash
sudo nixos-rebuild switch
```

The service starts automatically. Access at: **http://localhost:8000**

## Web Interface

1. **Enter YouTube URL** in the input field
2. Click **Summarize** - it downloads subtitles and streams the summary
3. **Ask questions** in the Q&A section below the summary

### Features
- ✨ **Dark/Light Mode** - Toggle in top-right corner
- 📊 **Real-time Streaming** - Watch the AI generate responses live
- 🔍 **Advanced Conversational RAG** - History-aware query reformulation + hybrid search (BM25 + semantic)
- 💬 **Smart Context Handling** - Automatically reformulates follow-up questions using chat history
- 🎯 **Grounded Responses** - AI cites exact phrases from transcript, reducing hallucinations
- 🚀 **2025 Best Practices** - Implements state-of-the-art retrieval techniques for maximum accuracy

## CLI Usage

```bash
# Summarize a video
nix run github:clearcmos/ytsummary -- <youtube-url>

# Or if installed via module
ytsummary <youtube-url>

# Load existing subtitle
ytsummary --load
```

## Configuration Options

```nix
services.ytsummary = {
  enable = true;

  # Model Configuration
  model = "qwen2.5:7b-instruct";            # Ollama model to use (default)
  autoConfigureOllama = true;               # Auto-setup Ollama (default: true)

  # Network
  port = 8000;                              # Web interface port
  ollamaUrl = "http://localhost:11434";     # Ollama API endpoint

  # Storage
  dataDir = "/var/lib/ytsummary";           # Subtitle storage

  # User/Group
  user = "ytsummary";                       # Service user
  group = "ytsummary";                      # Service group

  # Retention Policy
  retention = {
    enabled = true;                         # Auto-cleanup old files
    maxAge = "30d";                         # Keep files for 30 days
  };
};
```

### Zero-Config Setup

With `autoConfigureOllama = true` (default), the module automatically:
- ✅ Enables `services.ollama`
- ✅ Adds the configured model to `services.ollama.loadModels`
- ✅ Sets up proper systemd dependencies

**Just enable and it works!** No manual Ollama configuration needed.

## Traefik Integration

```nix
services.traefik.dynamicConfigOptions.http = {
  routers.ytsummary = {
    rule = "Host(`yt.yourdomain.com`)";
    service = "ytsummary";
  };

  services.ytsummary.loadBalancer.servers = [
    { url = "http://localhost:8000"; }
  ];
};
```

## Troubleshooting

### Check Service Status
```bash
systemctl status ytsummary
journalctl -u ytsummary -f
```

### Verify Ollama
```bash
curl http://localhost:11434/api/tags
```

### Manual Start (for testing)
```bash
cd /var/lib/ytsummary
ytsummary-web --port 8000
```

### View Downloaded Subtitles
```bash
ls -lah /var/lib/ytsummary/*.srt
```

## Requirements

- ✅ NixOS with flakes enabled
- **That's it!** The module handles everything else automatically

### Manual Model Installation (if autoConfigureOllama = false)

If you disable auto-configuration, install the model manually:

```bash
# Imperatively
ollama pull qwen2.5:7b-instruct

# Or declaratively in your config
services.ollama = {
  enable = true;
  loadModels = [ "qwen2.5:7b-instruct" ];
};
```

### Using Alternative Models

```nix
services.ytsummary = {
  enable = true;
  model = "llama3.2:3b";  # Faster, smaller model
  # OR
  model = "qwen2.5:14b-instruct";  # More capable, slower
};
```

## How It Works

### Advanced Hybrid Retrieval (2025 RAG Best Practices)

The Q&A system uses state-of-the-art retrieval techniques for maximum accuracy:

**1. History-Aware Query Reformulation** (Conversational RAG)
- Automatically reformulates follow-up questions using chat history
- Example conversation:
  - Q1: "How do you start using MCP?"
  - Q2: "How is he using docker?"
  - → Reformulated: "How does the speaker use Docker Desktop for MCP installation?"
- Uses LLM to create standalone questions before retrieval
- Dramatically improves accuracy for conversational Q&A

**2. Query Expansion**
- Automatically generates query variations for better recall
- Example: "How easy is it?" → also searches for "what is the way", "what are the steps"
- Applies to both original and reformulated queries

**3. Hybrid Search (BM25 + Semantic)**
- **BM25**: Keyword-based search catches exact phrases like "really easy", "setup instructions"
- **Semantic**: Embedding-based search understands meaning and synonyms
- **Reciprocal Rank Fusion (RRF)**: Intelligently combines both ranking methods

**4. Enhanced Context**
- 150-token chunk overlap ensures important context isn't split
- Top-5 retrieval (increased from 3) provides more comprehensive coverage

**5. Grounded Prompting**
- AI instructed to cite exact phrases and qualifiers
- Explicitly told to mention WHERE in the video information appears
- Reduces hallucinations by grounding responses in retrieved text

This multi-stage approach significantly improves factual accuracy compared to naive semantic-only RAG, especially for:
- Follow-up questions in conversations
- Questions about specific details, adjectives, or technical terms
- Queries using pronouns or context-dependent references ("it", "that", "this")
