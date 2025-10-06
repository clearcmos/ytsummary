#!/usr/bin/env python3
"""
YouTube Summary Web App - FastAPI Backend with SSE Streaming
Modern 2025 architecture with async streaming support
"""
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import asyncio
import json
from typing import AsyncGenerator
import sys
import os

# Import existing functionality
from download_subs import (
    check_human_subs,
    clean_srt_text,
    chunk_text_semantic,
    retrieve_relevant_chunks,
    ask_ollama,
    download_subtitles
)

app = FastAPI(title="YouTube Summary AI", version="1.0.0")

# Model configuration
OLLAMA_MODEL = os.environ.get('YTSUMMARY_MODEL', 'qwen2.5:7b-instruct')
OLLAMA_URL = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')

@app.on_event("startup")
async def startup_event():
    """Verify Ollama and model availability on startup"""
    try:
        # Check if Ollama is reachable
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]

            # Check if configured model is available
            if not any(OLLAMA_MODEL in name for name in model_names):
                print(f"⚠️  Warning: Model '{OLLAMA_MODEL}' not found in Ollama")
                print(f"   Available models: {', '.join(model_names) if model_names else 'none'}")
                print(f"\n   To install the model:")
                print(f"   - Imperatively: ollama pull {OLLAMA_MODEL}")
                print(f"   - NixOS declarative: Add to services.ollama.loadModels = [\"{OLLAMA_MODEL}\"];")
            else:
                print(f"✅ Ollama model '{OLLAMA_MODEL}' is available")
        else:
            print(f"⚠️  Warning: Could not connect to Ollama at {OLLAMA_URL}")
    except Exception as e:
        print(f"⚠️  Warning: Could not verify Ollama setup: {e}")
        print(f"   Make sure Ollama is running at {OLLAMA_URL}")

# Request/Response models
class URLRequest(BaseModel):
    url: str

class QuestionRequest(BaseModel):
    question: str
    video_title: str
    chunks_data: list
    subtitle_text: str
    summary: str
    conversation_history: list = []

# Global storage for video data (in production, use Redis/DB)
video_cache = {}

@app.get("/")
async def read_root():
    """Serve the main web interface"""
    from fastapi.responses import FileResponse
    _app_dir = os.path.dirname(os.path.abspath(__file__))
    return FileResponse(os.path.join(_app_dir, "static/index.html"))

@app.post("/api/download")
async def download_video(request: URLRequest):
    """Download subtitles and return video info"""
    try:
        # Download subtitles
        video_title = download_subtitles(request.url)

        # Find the SRT file
        import glob
        srt_files = glob.glob('*.en.srt')
        if not srt_files:
            raise HTTPException(status_code=404, detail="Subtitle file not found")

        srt_file = max(srt_files, key=os.path.getctime)

        # Extract and clean text
        subtitle_text = clean_srt_text(srt_file)

        # Create chunks
        chunks = chunk_text_semantic(subtitle_text, chunk_size=500, overlap=100)

        # Store in cache (use string hash to avoid integer overflow)
        video_id = str(abs(hash(request.url)))
        video_cache[video_id] = {
            'title': video_title,
            'subtitle_text': subtitle_text,
            'chunks': chunks,
            'srt_file': srt_file
        }

        return {
            'video_id': video_id,
            'title': video_title,
            'word_count': len(subtitle_text.split()),
            'chunk_count': len(chunks)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def stream_summary(text: str, title: str, chunks: list) -> AsyncGenerator[str, None]:
    """Stream summary generation with SSE format"""
    # Determine context
    if len(text.split()) < 2000:
        context_text = text
    else:
        context_chunks = [chunks[0], chunks[len(chunks)//2], chunks[-1]]
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

    # Stream the response
    for chunk, accumulated in ask_ollama(prompt, temperature=0.2, stream=True):
        if chunk is None:
            break
        # Send SSE formatted data
        yield f"data: {json.dumps({'chunk': chunk, 'accumulated': accumulated})}\n\n"
        await asyncio.sleep(0.01)

    yield "data: [DONE]\n\n"

@app.get("/api/summary/{video_id}")
async def get_summary(video_id: str):
    """Stream summary generation using SSE"""
    if video_id not in video_cache:
        raise HTTPException(status_code=404, detail="Video not found")

    data = video_cache[video_id]

    return StreamingResponse(
        stream_summary(data['subtitle_text'], data['title'], data['chunks']),
        media_type="text/event-stream"
    )

async def stream_answer(question: str, context: str) -> AsyncGenerator[str, None]:
    """Stream Q&A answer with SSE format"""
    for chunk, accumulated in ask_ollama(question, context, stream=True):
        if chunk is None:
            break
        yield f"data: {json.dumps({'chunk': chunk, 'accumulated': accumulated})}\n\n"
        await asyncio.sleep(0.01)

    yield "data: [DONE]\n\n"

@app.post("/api/question")
async def ask_question(request: QuestionRequest):
    """Stream answer to user question using SSE"""

    # Reconstruct chunks from request data
    chunks = request.chunks_data

    # Retrieve relevant chunks
    if len(chunks) > 5:
        relevant_chunks = retrieve_relevant_chunks(request.question, chunks, top_k=3)
        retrieved_text = "\n\n---RELEVANT SECTION---\n\n".join([c['text'] for c in relevant_chunks])
    else:
        retrieved_text = request.subtitle_text

    # Build context
    context = f"""Answer questions about the YouTube video "{request.video_title}" using ONLY the retrieved transcript sections below.

CRITICAL RULES:
1. Answer ONLY from the retrieved transcript sections below
2. If the answer isn't in these sections, say "This information is not in the video sections I can access"
3. Pay attention to exact wording, especially negative statements ("not for", "not recommended", "avoid")
4. Quote or paraphrase specific details from the transcript
5. Do NOT use general knowledge about this topic

RETRIEVED TRANSCRIPT SECTIONS:
{retrieved_text}

SUMMARY (for reference):
{request.summary}

PREVIOUS CONVERSATION:
{chr(10).join(request.conversation_history) if request.conversation_history else 'None'}

Answer the question using only the transcript sections above:"""

    return StreamingResponse(
        stream_answer(request.question, context),
        media_type="text/event-stream"
    )

# Mount static files (use absolute path based on app location)
_app_dir = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(_app_dir, "static")), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
