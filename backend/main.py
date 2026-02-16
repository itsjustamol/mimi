from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
from pathlib import Path
import sys
import os
import uuid
import asyncio
from io import BytesIO
from dotenv import load_dotenv
import anthropic
import requests
from PIL import Image, ImageDraw, ImageFont

from meme_search import MemeSearchEngine

# Load environment variables
load_dotenv()

app = FastAPI(title="Mimi - Semantic Meme Search")

# CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize search engine lazily (avoid blocking server start on Railway)
search_engine = None

# Directory for server-side generated meme images
GENERATED_DIR = Path("generated_memes")
GENERATED_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Pillow-based meme renderer (used for Imgflip & non-Memegen templates)
# ---------------------------------------------------------------------------

def _find_font(size: int) -> ImageFont.FreeTypeFont:
    """Find a bold font suitable for meme text."""
    candidates = [
        "/System/Library/Fonts/Supplemental/Impact.ttf",          # macOS
        "/Library/Fonts/Impact.ttf",                               # macOS alt
        "/usr/share/fonts/truetype/msttcorefonts/Impact.ttf",     # Linux
        "C:\\Windows\\Fonts\\impact.ttf",                          # Windows
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",   # Linux fallback
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    # absolute last resort – PIL default (bitmap, won't look great)
    return ImageFont.load_default()


def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int, draw: ImageDraw.ImageDraw) -> list[str]:
    """Word-wrap text so each line fits within max_width pixels."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        test = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [text]


def _draw_outlined_text(draw: ImageDraw.ImageDraw, xy: tuple, text: str,
                        font: ImageFont.FreeTypeFont, outline: int):
    """Draw white text with a black outline (classic meme style)."""
    x, y = xy
    # black outline
    for dx in range(-outline, outline + 1):
        for dy in range(-outline, outline + 1):
            if dx != 0 or dy != 0:
                draw.text((x + dx, y + dy), text, font=font, fill="black")
    # white fill
    draw.text((x, y), text, font=font, fill="white")


def render_meme_image(image_url: str, caption_lines: list[str]) -> str:
    """Download template image, overlay meme text, save and return filename."""
    resp = requests.get(image_url, timeout=15)
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content)).convert("RGBA")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    font_size = max(int(w / 12), 24)
    font = _find_font(font_size)
    outline = max(int(font_size / 12), 2)
    margin = int(w * 0.05)
    max_text_w = w - margin * 2

    def draw_block(lines_raw: list[str], y_anchor: str):
        """Render wrapped text block at top or bottom."""
        all_lines: list[str] = []
        for raw in lines_raw:
            all_lines.extend(_wrap_text(raw.upper(), font, max_text_w, draw))

        # compute total text block height
        line_heights = []
        for ln in all_lines:
            bbox = draw.textbbox((0, 0), ln, font=font)
            line_heights.append(bbox[3] - bbox[1])
        spacing = int(font_size * 0.15)
        block_h = sum(line_heights) + spacing * (len(all_lines) - 1)

        if y_anchor == "top":
            y = int(h * 0.03)
        else:  # bottom
            y = h - block_h - int(h * 0.03)

        for i, ln in enumerate(all_lines):
            bbox = draw.textbbox((0, 0), ln, font=font)
            lw = bbox[2] - bbox[0]
            x = (w - lw) / 2
            _draw_outlined_text(draw, (x, y), ln, font, outline)
            y += line_heights[i] + spacing

    if len(caption_lines) == 1:
        draw_block(caption_lines, "top")
    elif len(caption_lines) >= 2:
        draw_block([caption_lines[0]], "top")
        draw_block([caption_lines[-1]], "bottom")

    # save
    filename = f"{uuid.uuid4().hex[:12]}.jpg"
    out_path = GENERATED_DIR / filename
    img.convert("RGB").save(out_path, "JPEG", quality=92)
    return filename


class SearchQuery(BaseModel):
    query: str
    limit: int = 20

class MemeGenerationRequest(BaseModel):
    template_id: str
    template_name: str
    template_url: str = ""
    description: str

class IndexStatus(BaseModel):
    total_memes: int
    status: str
    sources: dict

async def _init_search_engine():
    """Initialize search engine in background so server can start immediately."""
    global search_engine
    print("Initializing semantic search engine...")
    # Run the CPU-heavy model loading in a thread to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    engine = await loop.run_in_executor(None, MemeSearchEngine)
    search_engine = engine
    print("Loading memes and creating embeddings...")
    await search_engine.initialize()
    print(f"✓ Indexed {len(search_engine.memes)} memes")

@app.on_event("startup")
async def startup_event():
    """Start model loading in background – server is responsive immediately."""
    asyncio.create_task(_init_search_engine())

@app.get("/health")
async def health_check():
    """Health check endpoint for Railway"""
    return {"status": "ok"}

@app.get("/")
async def root():
    """Serve the frontend"""
    return FileResponse("../frontend/index.html")

@app.get("/styles.css")
async def get_styles():
    """Serve CSS file"""
    return FileResponse("../frontend/styles.css")

@app.get("/script.js")
async def get_script():
    """Serve JavaScript file"""
    return FileResponse("../frontend/script.js")

@app.get("/api/generated/{filename}")
async def serve_generated_meme(filename: str):
    """Serve a server-side generated meme image."""
    path = GENERATED_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path, media_type="image/jpeg")

@app.get("/api/status")
async def get_status():
    """Get indexing status"""
    if search_engine is None:
        return {"total_memes": 0, "status": "loading", "sources": {}}
    return {
        "total_memes": len(search_engine.memes),
        "status": "ready" if search_engine.is_ready else "indexing",
        "sources": {
            "imgflip": len([m for m in search_engine.memes if m["source"] == "imgflip"]),
            "memegen": len([m for m in search_engine.memes if m["source"] == "memegen"]),
            "local": len([m for m in search_engine.memes if m["source"] == "local"])
        }
    }

@app.post("/api/search")
async def search_memes(query: SearchQuery):
    """Search for memes using natural language"""
    if search_engine is None or not search_engine.is_ready:
        raise HTTPException(status_code=503, detail="Search engine is still loading. Please wait a moment and try again.")

    results = await search_engine.search(query.query, limit=query.limit)

    return {
        "query": query.query,
        "results": results,
        "count": len(results)
    }

@app.post("/api/generate-meme")
async def generate_meme(request: MemeGenerationRequest):
    """Generate a meme with AI-generated captions"""

    # Check if API key is set
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="ANTHROPIC_API_KEY not set. Please add it to .env file"
        )

    try:
        # Step 1: Use Claude to generate meme caption from description
        client = anthropic.Anthropic(api_key=api_key)

        prompt = f"""You are an expert meme caption writer. Given a meme template and user's description, generate perfectly formatted meme text.

Meme Template: {request.template_name}
User's Description: {request.description}

IMPORTANT: Understand this specific meme's format:
- "Drake": Line 1 = rejected/bad option, Line 2 = preferred/good option
- "Expanding Brain" / "Galaxy Brain": Multiple escalating lines (simple → enlightened)
- "Distracted Boyfriend": Line 1 = distraction, Line 2 = boyfriend label, Line 3 = girlfriend label
- "Two Buttons": Short labels for two conflicting choices
- "Is This A Pigeon?": Line 1 = pointing at thing, Line 2 = wrong label question
- "Change My Mind": Single line = controversial opinion
- Most others: Line 1 = setup/context, Line 2 = punchline/reaction

Rules:
- Match the EXACT format this meme template needs
- Keep it SHORT and PUNCHY (max 50 chars per line)
- Be funny and relatable to the user's situation
- Use natural, conversational language
- Separate lines with " / " (space slash space)
- DO NOT include the template name in your output
- DO NOT use newlines within a line
- Return ONLY the meme text lines separated by " / ", nothing else

Example output format: "Line one text / Line two text" """

        message = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )

        caption_text = message.content[0].text.strip()

        # Clean up caption text - remove newlines and extra whitespace
        caption_text = caption_text.replace('\n', ' ').replace('\r', '')

        # Split caption into lines
        caption_lines = [line.strip() for line in caption_text.split(" / ")]

        # Clean each line - remove problematic characters
        caption_lines = [
            line.replace('\n', ' ').replace('\r', '').strip()
            for line in caption_lines
            if line.strip()
        ]

        # Step 2: Generate the meme image
        if request.template_id.startswith("memegen_"):
            # --- Memegen templates: use their URL API (reliable) ---
            def memegen_encode(text: str) -> str:
                text = text.replace("~", "~t")
                text = text.replace("_", "__")
                text = text.replace("-", "--")
                text = text.replace("?", "~q")
                text = text.replace("%", "~p")
                text = text.replace("#", "~h")
                text = text.replace("/", "~s")
                text = text.replace("\\", "~b")
                text = text.replace('"', "''")
                text = text.replace(" ", "_")
                return text

            template_id = request.template_id.replace("memegen_", "")
            encoded_lines = [memegen_encode(line) for line in caption_lines]
            meme_url = f"https://api.memegen.link/images/{template_id}/{'/'.join(encoded_lines)}.jpg"
        else:
            # --- Imgflip / other templates: render server-side with Pillow ---
            filename = render_meme_image(request.template_url, caption_lines)
            meme_url = f"/api/generated/{filename}"

        return {
            "success": True,
            "caption": caption_text,
            "caption_lines": caption_lines,
            "meme_url": meme_url,
            "template_name": request.template_name
        }

    except anthropic.APIError as e:
        raise HTTPException(status_code=500, detail=f"Claude API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating meme: {str(e)}")

@app.post("/api/reindex")
async def reindex():
    """Force re-indexing of all memes"""
    if search_engine is None:
        raise HTTPException(status_code=503, detail="Search engine is still loading.")
    await search_engine.initialize(force_reindex=True)
    return {"status": "success", "total_memes": len(search_engine.memes)}

# Mount static files (for serving frontend assets)
app.mount("/static", StaticFiles(directory="../frontend"), name="static")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
