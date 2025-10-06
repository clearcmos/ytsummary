#!/usr/bin/env python3
"""
Model Manager - Detects and helps configure Ollama models
"""
import subprocess
import sys
import json
from typing import Optional, List, Tuple

DEFAULT_MODEL = "qwen2.5:7b-instruct"
RECOMMENDED_MODELS = [
    ("qwen2.5:7b-instruct", "Default - Best balance of speed and accuracy"),
    ("llama3.2:3b", "Faster, smaller model"),
    ("qwen2.5:14b-instruct", "More capable, slower"),
]

def check_ollama_installed() -> bool:
    """Check if Ollama is installed"""
    try:
        subprocess.run(['which', 'ollama'], capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def get_installed_models() -> List[str]:
    """Get list of installed Ollama models"""
    try:
        result = subprocess.run(
            ['ollama', 'list'],
            capture_output=True,
            text=True,
            check=True
        )
        # Parse output (skip header line)
        models = []
        for line in result.stdout.strip().split('\n')[1:]:
            if line.strip():
                # Extract model name (first column)
                name = line.split()[0]
                models.append(name)
        return models
    except subprocess.CalledProcessError:
        return []

def check_model_available(model_name: str) -> bool:
    """Check if specific model is installed"""
    installed = get_installed_models()
    # Match base name (e.g., "qwen2.5:7b-instruct" matches "qwen2.5:7b-instruct")
    return any(model_name in m for m in installed)

def suggest_model_installation() -> Tuple[str, str]:
    """Generate instructions for installing the default model"""

    # Check if running on NixOS
    is_nixos = subprocess.run(
        ['test', '-f', '/etc/NIXOS'],
        capture_output=True
    ).returncode == 0

    if is_nixos:
        nixos_config = f'''
# Add to your NixOS configuration:

services.ollama = {{
  enable = true;
  loadModels = [ "{DEFAULT_MODEL}" ];
}};

# Then rebuild:
sudo nixos-rebuild switch
'''
        imperative_install = f'''
# Or install imperatively (non-declarative):
ollama pull {DEFAULT_MODEL}
'''
        return (
            f"NixOS Declarative Configuration (Recommended)",
            nixos_config + "\n" + imperative_install
        )
    else:
        return (
            f"Install Model",
            f"ollama pull {DEFAULT_MODEL}"
        )

def get_model_for_app(preferred_model: Optional[str] = None) -> Tuple[str, bool]:
    """
    Get model to use for the application

    Returns:
        (model_name, needs_installation)
    """
    if not check_ollama_installed():
        raise RuntimeError("Ollama is not installed. Install from: https://ollama.com")

    # Check preferred model first
    if preferred_model and check_model_available(preferred_model):
        return (preferred_model, False)

    # Check default model
    if check_model_available(DEFAULT_MODEL):
        return (DEFAULT_MODEL, False)

    # Check any recommended model
    for model, _ in RECOMMENDED_MODELS:
        if check_model_available(model):
            return (model, False)

    # No suitable model found
    return (DEFAULT_MODEL, True)

def print_model_status():
    """Print current model status and suggestions"""
    if not check_ollama_installed():
        print("❌ Ollama is not installed")
        print("   Install from: https://ollama.com")
        return

    print("✅ Ollama is installed")

    installed = get_installed_models()
    print(f"\n📦 Installed models ({len(installed)}):")
    for model in installed:
        print(f"   • {model}")

    model, needs_install = get_model_for_app()

    if needs_install:
        print(f"\n⚠️  Default model '{DEFAULT_MODEL}' not found")
        print("\n🔧 Installation options:")
        title, instructions = suggest_model_installation()
        print(f"\n{title}:")
        print(instructions)

        print("\n💡 Alternative models:")
        for alt_model, description in RECOMMENDED_MODELS[1:]:
            if check_model_available(alt_model):
                print(f"   ✅ {alt_model} - {description} (already installed)")
            else:
                print(f"   ⬜ {alt_model} - {description}")
    else:
        print(f"\n✅ Using model: {model}")

if __name__ == "__main__":
    print_model_status()
