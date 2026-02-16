# Deploying mimi to Railway

## Prerequisites

- GitHub account
- Railway account (sign up at https://railway.app)

## Deployment Steps

### 1. Push to GitHub

```bash
# Create a new GitHub repository (via GitHub website)
# Then push your code:
git remote add origin https://github.com/YOUR_USERNAME/mimi.git
git branch -M main
git push -u origin main
```

### 2. Deploy to Railway

1. **Go to Railway**: https://railway.app
2. **Login** with GitHub
3. Click **"New Project"**
4. Select **"Deploy from GitHub repo"**
5. Choose your **mimi** repository
6. Railway will automatically detect it's a Python app and start deploying

### 3. Add Environment Variables

After deployment starts:

1. Go to your project in Railway
2. Click on your service
3. Go to **"Variables"** tab
4. Click **"New Variable"**
5. Add:
   - **Name**: `ANTHROPIC_API_KEY`
   - **Value**: `your_anthropic_api_key_here`
6. Add:
   - **Name**: `SEARCH_MODE`
   - **Value**: `auto`
7. Add:
   - **Name**: `MAX_REMOTE_MEMES`
   - **Value**: `160`
8. Click **"Add"**

### 4. Wait for Deployment

Railway will:
- Install Python dependencies (~2-3 minutes)
- Download CLIP model assets
- Start the server
- Give you a public URL like: `https://mimi-production.up.railway.app`

**Total time: 5-10 minutes**

### 5. Access Your App

Once deployed, Railway will show:
- ✅ **Deployed** status
- 🌐 **Public URL** - click to open your app

Your meme search app is now live! 🎉

---

## Important Notes

### First Request Will Be Slow
- The first startup does indexing and embedding creation
- This is normal and depends on `MAX_REMOTE_MEMES`
- Subsequent restarts are faster (cache is reused)

### Free Tier Limits
Railway free tier includes:
- $5/month credit
- ~500 hours runtime
- Should be plenty for personal use

### Monitoring
- Check **"Deployments"** tab for build logs
- Check **"Metrics"** tab for usage
- Check **"Logs"** tab for runtime errors

### Troubleshooting

**Build fails:**
- Check that `requirements.txt` is in the root
- Verify `Procfile` exists

**App crashes on startup:**
- Check logs in Railway dashboard
- Verify `ANTHROPIC_API_KEY` is set correctly
- Keep `SEARCH_MODE=auto` on free tier
- Reduce `MAX_REMOTE_MEMES` (for example: `120`)

**Can't access the app:**
- Make sure deployment shows "Deployed" status
- Click the generated Railway URL (not localhost!)

**Out of memory:**
- Railway free tier has 512MB RAM
- CLIP model needs ~400MB
- May need to upgrade to Pro ($5/month for 8GB RAM)

---

## Updating Your Deployment

To deploy changes:

```bash
git add .
git commit -m "Your changes"
git push origin main
```

Railway automatically redeploys on every push to `main`! 🚀
