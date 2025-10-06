{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    python3
    python3Packages.requests
    python3Packages.numpy
    python3Packages.scikit-learn
    python3Packages.sentence-transformers
    python3Packages.rich
    python3Packages.fastapi
    python3Packages.uvicorn
    python3Packages.pydantic
    yt-dlp
  ];

  shellHook = ''
    echo "🚀 YouTube Summary AI - Test Environment"
    echo ""
    echo "Available commands:"
    echo "  Start web server:  uvicorn app:app --host 0.0.0.0 --port 8000 --reload"
    echo "  CLI tool:          python download_subs.py <youtube-url>"
    echo "  CLI with --load:   python download_subs.py --load"
    echo ""
    echo "Web UI will be at: http://localhost:8000"
    echo ""
    echo "Make sure Ollama is running with qwen2.5:7b-instruct model!"
    echo ""
  '';
}
