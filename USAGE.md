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
