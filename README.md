# mimi — semantic meme search

A tastefully designed web application for searching memes using natural language queries. Built with visual semantic search powered by OpenAI's CLIP model.

## ✨ Features

- 🔍 **Semantic Search**: Search memes using natural language descriptions instead of keywords
- 🎨 **Beautiful UI**: Clean, refined interface inspired by art gallery aesthetics
- 🖼️ **Dual Sources**: Search through 100+ popular meme templates from Imgflip + your personal collection
- ⚡ **Fast Performance**: Embeddings are cached after first run for instant searches
- 🎯 **Relevance Scoring**: See how well each meme matches your query

## 🚀 Quick Start

### Prerequisites

- Python 3.8+ (3.10 recommended)
- pip
- ~2GB disk space for CLIP model

### Installation

1. **Clone and navigate to the project**:
   ```bash
   cd /Users/amol/github/mimi
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   This will install:
   - FastAPI (web framework)
   - CLIP model via transformers
   - Image processing libraries
   - All required dependencies

### Running the App

1. **Start the backend server**:
   ```bash
   cd backend
   python main.py
   ```

   On first run, this will:
   - Download the CLIP model (~350MB)
   - Fetch 100 popular meme templates from Imgflip
   - Create embeddings for all memes (takes ~2-3 minutes)
   - Cache embeddings for future runs

   You'll see:
   ```
   Loading CLIP model...
   ✓ CLIP model loaded on cpu
   Fetching memes from Imgflip...
   ✓ Fetched 100 memes from Imgflip
   Creating embeddings...
   ✓ Created embeddings for 100 memes
   ✓ Cache saved
   ```

2. **Open the app**:
   - Navigate to `http://localhost:8000` in your browser
   - Wait for the status indicator to show "ready"
   - Start searching!

## 🎯 How to Use

### Searching

Use natural language to describe the meme you're looking for:

- ✅ **Good**: "reaction when someone is confidently wrong"
- ✅ **Good**: "existential crisis but make it funny"
- ✅ **Good**: "awkward silence energy"
- ❌ **Less effective**: "drake", "distracted boyfriend"

The semantic search understands concepts, emotions, and situations better than exact template names.

### Adding Your Own Memes

1. Place your meme images in the `memes/` folder:
   ```bash
   cp ~/Downloads/my-favorite-meme.jpg memes/
   ```

2. Supported formats: `.jpg`, `.jpeg`, `.png`, `.gif`, `.webp`

3. Restart the server or trigger a re-index:
   ```bash
   curl -X POST http://localhost:8000/api/reindex
   ```

   Your personal memes will now be searchable alongside Imgflip templates!

## 🏗️ Project Structure

```
mimi/
├── backend/
│   ├── main.py              # FastAPI server
│   └── meme_search.py       # CLIP-based search engine
├── frontend/
│   ├── index.html           # Main HTML
│   ├── styles.css           # Tasteful styling
│   └── script.js            # Frontend logic
├── memes/                   # Your personal memes (add here!)
├── requirements.txt         # Python dependencies
├── embeddings_cache.json    # Cached embeddings (auto-generated)
└── README.md
```

## 🔧 API Endpoints

- `GET /api/status` - Check indexing status and meme count
- `POST /api/search` - Search memes with natural language
  ```json
  {
    "query": "confused but trying to understand",
    "limit": 20
  }
  ```
- `POST /api/reindex` - Force re-indexing of all memes

## 🎨 Design Philosophy

The UI is inspired by art gallery aesthetics with:
- Refined serif typography (Cormorant Garamond)
- Clean, spacious layouts
- Subtle grain texture and gradient orbs
- Smooth animations and transitions
- High contrast, readable text

## 🧠 How It Works

1. **Indexing**:
   - CLIP model converts each meme image into a 512-dimensional embedding vector
   - Embeddings capture visual and semantic meaning
   - Cached to disk for fast subsequent loads

2. **Searching**:
   - Your text query is converted to the same embedding space
   - Cosine similarity finds the most semantically similar memes
   - Results ranked by relevance score

3. **Why CLIP?**:
   - Joint vision-language model trained on 400M image-text pairs
   - Understands both images and natural language descriptions
   - No fine-tuning needed for meme search

## 🔮 Future Ideas

- [ ] Support for local meme folders with subdirectories
- [ ] Save favorite searches
- [ ] Meme collections/tags
- [ ] Browser extension for quick access
- [ ] Mobile-responsive design improvements
- [ ] Dark mode toggle
- [ ] Export/share meme search results

## 🐛 Troubleshooting

**"Module not found" errors**:
- Make sure you activated the virtual environment
- Run `pip install -r requirements.txt` again

**Slow first-time startup**:
- Normal! CLIP model download + embedding creation takes time
- Subsequent runs are much faster (uses cache)

**"Connection refused" errors**:
- Make sure backend server is running on port 8000
- Check `python backend/main.py` is active

**Empty search results**:
- Try more descriptive queries
- CLIP works best with conceptual descriptions
- Check that memes are properly indexed (see status indicator)

## 📝 License

MIT License - feel free to use, modify, and share!

## 🙏 Credits

- CLIP model by OpenAI
- Meme templates from Imgflip
- Design inspired by semantic.art
- Built with FastAPI, transformers, and vanilla JS

---

**Made with ❤️ for finding the perfect meme at the perfect time**
