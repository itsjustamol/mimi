import os
import json
import requests
from pathlib import Path
from typing import List, Dict, Any
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from io import BytesIO

class MemeSearchEngine:
    def __init__(self):
        self.memes: List[Dict[str, Any]] = []
        self.embeddings: np.ndarray = None
        self.is_ready = False
        # Initialize CLIP model - using base model for memory efficiency
        print("Loading CLIP model (base variant optimized for Railway free tier)...")
        model_name = "openai/clip-vit-base-patch32"
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"✓ CLIP base model loaded on {self.device}")

        # Paths
        self.cache_file = Path("embeddings_cache.json")
        self.local_memes_dir = Path("../memes")
        self.local_memes_dir.mkdir(exist_ok=True)

    async def initialize(self, force_reindex: bool = False):
        """Load memes from Imgflip and local folder, then create embeddings"""

        # Try to load from cache first
        if not force_reindex and self.cache_file.exists():
            print("Loading from cache...")
            try:
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                    self.memes = cache_data['memes']
                    self.embeddings = np.array(cache_data['embeddings'])
                    self.is_ready = True
                    print(f"✓ Loaded {len(self.memes)} memes from cache")
                    return
            except Exception as e:
                print(f"Cache load failed: {e}. Re-indexing...")

        # Fetch from Imgflip
        print("Fetching memes from Imgflip...")
        imgflip_memes = self._fetch_imgflip_memes()
        print(f"✓ Fetched {len(imgflip_memes)} memes from Imgflip")

        # Fetch from Memegen.link
        print("Fetching memes from Memegen.link...")
        memegen_memes = self._fetch_memegen_memes()
        print(f"✓ Fetched {len(memegen_memes)} memes from Memegen.link")

        # Load local memes
        print("Loading local memes...")
        local_memes = self._load_local_memes()
        print(f"✓ Found {len(local_memes)} local memes")

        # Combine all memes
        all_memes = imgflip_memes + memegen_memes + local_memes

        # Deduplicate by normalized name (prefer imgflip, then memegen, then local)
        print("Deduplicating memes...")
        seen_names = set()
        deduplicated_memes = []

        for meme in all_memes:
            # Normalize name for comparison (lowercase, remove special chars)
            normalized_name = ''.join(c.lower() for c in meme['name'] if c.isalnum())

            if normalized_name not in seen_names:
                seen_names.add(normalized_name)
                deduplicated_memes.append(meme)

        self.memes = deduplicated_memes
        print(f"✓ Deduplicated to {len(self.memes)} unique memes (removed {len(all_memes) - len(self.memes)} duplicates)")

        # Create embeddings for all memes
        print("Creating embeddings (this may take a minute)...")
        self.embeddings = self._create_embeddings()
        print(f"✓ Created embeddings for {len(self.memes)} memes")

        # Save to cache
        self._save_cache()

        self.is_ready = True

    def _fetch_imgflip_memes(self) -> List[Dict[str, Any]]:
        """Fetch popular meme templates from Imgflip"""
        try:
            response = requests.get("https://api.imgflip.com/get_memes", timeout=10)
            data = response.json()

            if data.get("success"):
                memes = []
                # Get all available templates (not just top 100)
                for meme in data["data"]["memes"]:
                    memes.append({
                        "id": f"imgflip_{meme['id']}",
                        "name": meme["name"],
                        "url": meme["url"],
                        "source": "imgflip",
                        "width": meme["width"],
                        "height": meme["height"]
                    })
                return memes
        except Exception as e:
            print(f"Error fetching Imgflip memes: {e}")

        return []

    def _fetch_memegen_memes(self) -> List[Dict[str, Any]]:
        """Fetch meme templates from Memegen.link (no auth required)"""
        try:
            response = requests.get("https://api.memegen.link/templates/", timeout=15)
            data = response.json()

            memes = []
            for template in data:
                # Use the blank template image (no text overlay)
                memes.append({
                    "id": f"memegen_{template['id']}",
                    "name": template['name'],
                    "url": template['blank'],
                    "source": "memegen",
                    "keywords": template.get('keywords', [])
                })

            return memes
        except Exception as e:
            print(f"Error fetching Memegen templates: {e}")
            return []

    def _fetch_reddit_memes(self) -> List[Dict[str, Any]]:
        """Fetch memes from Reddit via Meme-API (no auth required)"""
        memes = []

        # Popular meme subreddits to fetch from
        subreddits = [
            ('memes', 50),
            ('dankmemes', 50),
            ('MemeEconomy', 30),
            ('AdviceAnimals', 20),
        ]

        for subreddit, count in subreddits:
            try:
                print(f"  Fetching from r/{subreddit}...")
                url = f"https://meme-api.com/gimme/{subreddit}/{count}"
                response = requests.get(url, timeout=15)
                data = response.json()

                if 'memes' in data:
                    for meme in data['memes']:
                        # Only include image posts (not videos)
                        if meme.get('url', '').lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                            memes.append({
                                "id": f"reddit_{meme['postLink'].split('/')[-2]}",
                                "name": meme['title'][:100],  # Truncate long titles
                                "url": meme['url'],
                                "source": "reddit",
                                "subreddit": subreddit,
                                "author": meme.get('author', 'unknown'),
                                "upvotes": meme.get('ups', 0)
                            })
            except Exception as e:
                print(f"  Error fetching from r/{subreddit}: {e}")
                continue

        print(f"✓ Fetched {len(memes)} memes from Reddit")
        return memes

    def _load_local_memes(self) -> List[Dict[str, Any]]:
        """Load memes from local folder"""
        memes = []
        supported_formats = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}

        for img_path in self.local_memes_dir.iterdir():
            if img_path.suffix.lower() in supported_formats:
                memes.append({
                    "id": f"local_{img_path.stem}",
                    "name": img_path.stem.replace('_', ' ').replace('-', ' '),
                    "url": f"file://{img_path.absolute()}",
                    "path": str(img_path.absolute()),
                    "source": "local"
                })

        return memes

    def _generate_meme_context(self, meme: Dict[str, Any]) -> str:
        """Generate contextual description for a meme to enhance search"""
        name = meme['name'].lower()

        # Common meme contexts - helps search understand meaning beyond just visuals
        context_map = {
            'drake': 'choice preference rejection approval disapproval comparing options',
            'distracted boyfriend': 'temptation choice betrayal looking away interest distraction',
            'expanding brain': 'intelligence levels complexity smart dumb enlightenment ascension',
            'two buttons': 'difficult choice dilemma decision anxiety stress',
            'is this a pigeon': 'confusion misunderstanding incorrect identification oblivious',
            'change my mind': 'debate argument unpopular opinion challenge convince',
            'woman yelling at cat': 'argument disagreement confusion defensive angry',
            'surprised pikachu': 'shock surprise obvious consequence unexpected reaction',
            'one does not simply': 'challenge difficulty impossible task warning',
            'mocking spongebob': 'mockery sarcasm ridicule imitation teasing',
            'this is fine': 'denial crisis disaster calm chaos acceptance',
            'galaxy brain': 'genius stupid smart dumb intelligence enlightened',
            'angry': 'frustration mad upset annoyed irritated',
            'awkward': 'uncomfortable cringe embarrassing silence tension',
            'success': 'victory win achievement accomplishment celebration',
            'sad': 'depressed unhappy crying disappointment melancholy',
            'thinking': 'contemplating wondering confused pondering decision',
            'laughing': 'humor funny hilarious comedy joy amusement',
            'facepalm': 'disappointment frustration stupid obvious mistake',
            'waiting': 'bored impatient anticipation expecting delay',
        }

        # Find matching context
        context = meme['name']
        for key, description in context_map.items():
            if key in name:
                context = f"{meme['name']}. {description}"
                break

        # Add Memegen keywords if available
        if meme.get('source') == 'memegen' and meme.get('keywords'):
            keywords = ' '.join(meme['keywords'])
            context = f"{context}. {keywords}"

        return context

    def _create_embeddings(self) -> np.ndarray:
        """Create multi-modal CLIP embeddings combining visual and text context"""
        embeddings = []

        for i, meme in enumerate(self.memes):
            try:
                # Load image
                if meme["source"] == "local":
                    image = Image.open(meme["path"]).convert("RGB")
                else:
                    response = requests.get(meme["url"], timeout=10)
                    image = Image.open(BytesIO(response.content)).convert("RGB")

                # Create visual embedding
                img_inputs = self.processor(images=image, return_tensors="pt")
                img_inputs = {k: v.to(self.device) for k, v in img_inputs.items()}

                with torch.no_grad():
                    # Visual features
                    vision_outputs = self.model.vision_model(**img_inputs)
                    image_features = vision_outputs.pooler_output
                    image_features = self.model.visual_projection(image_features)
                    image_features = F.normalize(image_features, p=2, dim=-1)

                    # Text context features to enhance understanding
                    context_text = self._generate_meme_context(meme)
                    text_inputs = self.processor(text=[context_text], return_tensors="pt", padding=True)
                    text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

                    text_outputs = self.model.text_model(**text_inputs)
                    text_features = text_outputs.pooler_output
                    text_features = self.model.text_projection(text_features)
                    text_features = F.normalize(text_features, p=2, dim=-1)

                    # Combine visual (70%) + text context (30%) for better semantic understanding
                    combined_features = 0.7 * image_features + 0.3 * text_features
                    combined_features = F.normalize(combined_features, p=2, dim=-1)

                embeddings.append(combined_features.cpu().numpy()[0])

                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{len(self.memes)} memes...")

            except Exception as e:
                print(f"Error processing {meme['name']}: {e}")
                # Add zero embedding as placeholder (512 dims for base model)
                embeddings.append(np.zeros(512))

        return np.array(embeddings)

    async def search(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search for memes using natural language query"""

        # Create text embedding
        inputs = self.processor(text=[query], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            # Use text model directly
            text_outputs = self.model.text_model(**inputs)
            text_features = text_outputs.pooler_output
            # Project to joint embedding space
            text_features = self.model.text_projection(text_features)
            # Normalize
            text_features = F.normalize(text_features, p=2, dim=-1)

        text_embedding = text_features.cpu().numpy()[0]

        # Calculate cosine similarities
        similarities = np.dot(self.embeddings, text_embedding)

        # Filter out low-confidence matches (below 0.18 similarity)
        min_threshold = 0.18
        valid_indices = np.where(similarities >= min_threshold)[0]

        if len(valid_indices) == 0:
            # If no good matches, return top results anyway but with warning
            top_indices = np.argsort(similarities)[::-1][:limit]
        else:
            # Get top results from valid matches
            valid_similarities = similarities[valid_indices]
            sorted_valid = np.argsort(valid_similarities)[::-1]
            top_indices = valid_indices[sorted_valid][:limit]

        results = []
        for idx in top_indices:
            meme = self.memes[idx].copy()
            meme["score"] = float(similarities[idx])
            results.append(meme)

        return results

    def _save_cache(self):
        """Save memes and embeddings to cache file"""
        try:
            cache_data = {
                "memes": self.memes,
                "embeddings": self.embeddings.tolist()
            }
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f)
            print("✓ Cache saved")
        except Exception as e:
            print(f"Error saving cache: {e}")
