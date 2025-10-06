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
            ollamaUrl = "http://localhost:11434";
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
- 🔍 **RAG-Enhanced Q&A** - Semantic search finds relevant transcript sections
- 💬 **Conversation History** - Maintains context across questions

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

  port = 8000;                              # Web interface port
  ollamaUrl = "http://localhost:11434";     # Ollama API endpoint
  dataDir = "/var/lib/ytsummary";           # Subtitle storage

  user = "ytsummary";                       # Service user
  group = "ytsummary";                      # Service group

  retention = {
    enabled = true;                         # Auto-cleanup old files
    maxAge = "30d";                         # Keep files for 30 days
  };
};
```

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
- ✅ Ollama service running
- ✅ Ollama model: `qwen2.5:7b-instruct`

```bash
# Install Ollama model
ollama pull qwen2.5:7b-instruct
```
