# Render Deployment Guide

Your Empwave app is ready to deploy on **Render** with automatic keep-alive to prevent spindown!

## What is `.joblib`?

`.joblib` files are **serialized Python objects** - specifically your trained emotion classifier:
- `empwave_classifier.joblib` = 28 logistic regression models (one per emotion)
- Loaded once at app startup into memory
- NOT a database - just a pre-trained model file
- Already optimized to 56.5KB (27% smaller than original)

## Deployment Steps

### 1. Push to GitHub
```bash
cd /Users/prajwalahluwalia/Desktop/empwave
git add -A
git commit -m "Add Render deployment config and keep-alive service"
git push origin main
```

### 2. Deploy via Render Blueprint

1. Go to https://dashboard.render.com
2. Click **"New +"** → **"Blueprint"**
3. Select your **empwave** GitHub repo
4. Click **"Create from Blueprint"**
5. Wait ~2-3 minutes for deployment

Render reads `render.yaml` and automatically creates:
- ✅ Web Service (main Flask app)
- ✅ Background Service (keep-alive pinger)

### 3. Update Keep-Alive URL

After deployment, you'll see your app URL like: `https://empwave-abc123.onrender.com`

**Update `render.yaml`:**
```yaml
- type: background
  name: empwave-keepalive
  ...
  envVars:
    - key: APP_URL
      value: https://empwave-abc123.onrender.com  # ← CHANGE THIS
```

Then push to redeploy:
```bash
git add render.yaml
git commit -m "Update Render app URL for keep-alive"
git push origin main
```

### 4. Done! 🎉

Your app is live and will:
- ✅ Auto-rebuild on every git push
- ✅ Stay awake 24/7 (keep-alive pings every 10 min)
- ✅ Serve with optimized models
- ✅ Handle no database (frontend + backend only)

---

## How Keep-Alive Works

Render's free tier spins down inactive apps after 15 minutes. Our solution:

1. **Health Check Endpoint** (`/health`)
   - Lightweight endpoint your app already has
   - Used to verify app is running

2. **Background Keep-Alive Service**
   - Runs independently as a second service
   - Pings `/health` every 10 minutes
   - Counts as 1 background worker (free tier allows 1)
   - No additional cost ✅

3. **Result**
   - App never spins down
   - Always responds instantly (no 50-second delay)
   - Seamless user experience

---

## Architecture

```
┌─────────────────────────────────────┐
│         Render.com Account          │
├─────────────────────────────────────┤
│                                     │
│  ┌─────────────────────────────┐   │
│  │   WEB SERVICE (Port 7860)   │   │
│  │  Flask App + Brain Logic    │   │
│  │  Models: ~350MB RAM         │   │
│  │  Responds to user requests  │   │
│  └─────────────────────────────┘   │
│              ▲                      │
│              │ (pings every 10min) │
│              │                     │
│  ┌─────────────────────────────┐   │
│  │  BACKGROUND SERVICE         │   │
│  │  Keep-Alive Pinger          │   │
│  │  runs: keep_alive.py        │   │
│  │  (independent, always on)   │   │
│  └─────────────────────────────┘   │
│                                     │
└─────────────────────────────────────┘
```

---

## Your App Stack

- **Backend:** Python Flask (no changes needed)
- **Frontend:** HTML + Three.js + Web Speech API (no database)
- **Models:** Sentence encoder (ONNX) + Emotion classifier (joblib)
- **Deployment:** Render with keep-alive service
- **Cost:** Free (2 services = 1 web + 1 background worker on free tier)

---

## Troubleshooting

**App takes >50 seconds to respond?**
- Check if keep-alive is running (look for "Keep-alive ping" in Background Service logs)
- Verify `APP_URL` in `render.yaml` is correct

**Keep-alive service not working?**
- Check Background Service logs: should see "Keep-alive ping successful" every 10 min
- Verify web service is healthy (check `/health` endpoint manually)

**Deploy failed?**
- Check build logs in Render dashboard
- Ensure `render.yaml` is in repo root
- Verify `requirements.txt` has all dependencies

---

**Ready to deploy?** Follow the 3 steps above and you're live! 🚀
