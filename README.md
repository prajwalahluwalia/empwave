---
title: Empwave
emoji: 🧠
colorFrom: purple
colorTo: blue
sdk: gradio
app_port: 7860
pinned: false
---

# Brain Reaction Prototype - Setup & Running Guide

A minimalist, immersive web-based interactive prototype that simulates a human brain reacting to typed text in real-time. Built with Python Flask, Three.js, and the Web Speech API.

## 📋 Prerequisites

- **Python 3.10+** - Download from [python.org](https://www.python.org/downloads/)
- **pip** - Comes with Python 3.4+
- **A modern web browser** - Chrome, Firefox, Safari, or Edge (supports Web Speech API)
- **Microphone/Speaker** - For Web Speech synthesis and optional voice input

## 🚀 Quick Start (5 minutes)

### 1. Install Dependencies

```bash
# Navigate to the project directory
cd /Users/prajwalahluwalia/Desktop/empwave

# Create a virtual environment (recommended)
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install Flask and the local semantic NLP model
pip install -r requirements.txt
```

### 2. Run the Application

```bash
# Make sure you're in the project directory and virtual environment is activated
python app.py
```

You should see output like:
```
 * Running on http://127.0.0.1:5007
 * Debug mode: on
```

### 3. Open in Browser

Open your web browser and navigate to:
```
http://localhost:5007
```

## 🎮 How to Use

1. **Type a Sentence** - Enter any text in the input field at the bottom of the screen
2. **Click "Play"** - Or press Enter to speak the text aloud
3. **Watch the Brain React** - As the text is spoken, the brain's 3D wireframe will:
   - Rotate smoothly in 3D space
   - Pulse gently with a heartbeat rhythm
   - Display glowing colored particle streams that flow toward specific brain regions
4. **See Active Regions** - The top-right corner shows the regions selected by local semantic NLP analysis

Empwave uses a layered local NLP pipeline. A trained GoEmotions model detects
multiple emotions, while semantic clause analysis detects cognitive, sensory,
movement, memory, and language intents. A transparent weighted mapping combines
those signals into illustrative brain-region scores and always includes
auditory decoding for spoken input.

The sentence encoder loads once when the Flask process starts and is then
reused by every request. Deployment builds export an INT8 ONNX graph so the
production process does not load PyTorch. The trained scikit-learn emotion heads
are also exported as plain NumPy coefficients so SciPy and scikit-learn are
not loaded by the production process. No API key or hosted inference service
is required.

## Train an Empwave classifier

```bash
source .empenv/bin/activate
python scripts/build_dataset.py --source-root ~/Desktop/data
python scripts/train_model.py
```

Training uses the frozen MiniLM sentence encoder and learns a supervised
logistic classifier for every GoEmotions label. Semantic intents remain a
separate layer, and the runtime maps both layers to brain regions. The model
artifact and test metrics are written to `models/trained/`.

## Emotion feedback review

This feature is currently disabled by default.

After a simulation, users can select **Review detected emotions** and submit a
consented self-report. Feedback is stored separately in
`data/feedback/empwave_feedback.sqlite3`; it never retrains the model
automatically. Duplicate, unusually broad, repeated, and strong-disagreement
submissions are flagged for review rather than treated as malicious by default.

```bash
EMPWAVE_ENABLE_EMOTION_FEEDBACK=1 python app.py

python scripts/review_feedback.py list
python scripts/review_feedback.py approve 12 --note "Verified correction"
python scripts/review_feedback.py reject 13 --note "Spam submission"
```

## Deploy on Hugging Face Spaces

### Quick Setup (3 steps):

1. **Create a Hugging Face account** (free at [huggingface.co](https://huggingface.co))

2. **Create a new Space:**
   - Go to [huggingface.co/new-space](https://huggingface.co/new-space)
   - Name: `empwave`
   - License: Choose one (MIT recommended)
   - Space SDK: **Gradio** (not Docker)
   - Click "Create Space"

3. **Connect your GitHub repo:**
   - In your new Space, go to **Settings → Repository** 
   - Enable "Sync with a Git repository"
   - Connect this GitHub repo: `prajwalahluwalia/empwave`
   - Branch: `main`
   - Click "Save"

**That's it!** HF Spaces will automatically:
- ✅ Detect `requirements.txt` and install dependencies
- ✅ Run `python scripts/cache_model.py` during build (configure in app.py)
- ✅ Start your Flask app on port 7860
- ✅ Deploy live at `hf.co/spaces/your-username/empwave`

### Optional: Add a startup script

Create `app_hf.py` in your root (HF Spaces looks for this):

```python
import os
from empwave import create_app

app = create_app()

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=7860,
        debug=False
    )
```

Or keep your current `app.py` — it already works with HF Spaces.

### Environment Variables

If needed, set them in Space **Settings → Variables**:
- `FLASK_ENV`: `production`
- `ALLOWED_ORIGIN`: Your Space URL
- `EMPWAVE_ENABLE_EMOTION_FEEDBACK`: `0` (or `1` to enable)

### Notes

- The trained `.joblib` artifact is covered by `.gitattributes`
- See `deployment_stratergy.md` for Git LFS setup
- Free tier has 2GB RAM (plenty for your model)
- Spins down after inactivity (normal for free tier)

### Keyboard Shortcuts

- **Enter** - Speak the text
- **Backspace** - Clear text

## 🛑 Stop the Server

Press `Ctrl+C` in your terminal.

## 📁 Project Structure

```
empwave/
├── app.py                  # Application entry point
├── requirements.txt
├── empwave/
│   ├── __init__.py         # Flask application factory
│   ├── routes.py           # Web and NLP API routes
│   └── services/
│       └── intent_classifier.py
├── static/models/brain.obj # Anatomical brain surface
├── scripts/
│   ├── build_dataset.py     # Build weakly supervised training data
│   └── train_model.py       # Train and evaluate region classifiers
├── data/processed/          # Generated train/validation/test JSONL
├── models/trained/          # Generated trained model and metrics
├── templates/
│   └── index.html         # Complete frontend (Tailwind, Three.js, Web Speech API)
├── tests/
│   └── test_intent_classifier.py
└── README.md              # This file
```

## 🔧 Technical Details

### Backend (app.py)

- **Flask Application** - Lightweight Python web server
- **Text Analysis Endpoint** - `/api/process-speech` returns semantic region scores, strengths, evidence, and listener-focused reasons
- **Local Semantic NLP** - Uses Sentence Transformers rather than browser keyword matching

### Frontend (index.html)

- **Three.js** - Creates and renders the 3D wireframe brain
- **Tailwind CSS** - Minimal, modern dark-mode styling
- **Web Speech API** - Native browser text-to-speech
- **Canvas Rendering** - Full-screen immersive experience
- **Particle System** - Custom particle streams for visual feedback

### Key Features

1. **3D Brain Wireframe** - Procedurally generated brain lobes and corpus callosum
2. **Real-time Particle Effects** - Particle streams triggered on each spoken word
3. **Brain Region Mapping** - Semantic analysis maps words to brain regions
4. **Smooth Animations** - Continuous rotation, pulsing, and particle motion
5. **Responsive Design** - Adapts to window resizing
6. **No External Dependencies** (Frontend) - Uses only CDN-hosted Three.js and Tailwind

## 🎨 Customization

### Adjust Brain Rotation Speed

Edit `app.py` lines in the animation loop:
```javascript
brain.rotation.x += 0.0003;  // Change these values
brain.rotation.y += 0.0005;
```

### Modify Brain Size

In `index.html`, the scale factor in `BrainGeometry` constructor:
```javascript
this.addBrainLobe(vertices, indices, 0, 0, 0, 1, -1);  // Last param is scale
```

### Add More Brain Regions

1. Add to `brainRegions` object in `index.html`
2. Add keyword triggers in `app.py` `/api/analyze-text` function

Example:
```python
# In app.py
if any(word in text_lower for word in ['your', 'keywords']):
    regions.append('your_region_key')
```

```javascript
// In index.html
your_region_key: {
    position: new THREE.Vector3(x, y, z),
    color: 0xRRGGBB,
    label: "Your Region Name"
}
```

### Change Particle Colors

Modify the `color` property in the `brainRegions` object (hex RGB values).

## 🐛 Troubleshooting

### "Flask not found" error
```bash
pip install flask
```

### Web Speech API not working
- Ensure you're using a modern browser
- Check that microphone/speaker permissions are granted
- Note: Web Speech API works best in Chrome/Edge

### Brain not rendering
- Check browser console for errors (F12)
- Ensure JavaScript is enabled
- Try a different browser

### Port 5000 already in use
```bash
# Change port in app.py:
if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Use different port
```

## 📚 API Reference

### POST `/simulate`

**Request:**
```json
{
  "text": "I balance on a tightrope"
}
```

**Response:**
```json
{
  "regions": [
    {
      "id": "cerebellum",
      "strength": 0.715,
      "trigger": "I balance on a tightrope"
    }
  ],
  "fallback": false,
  "spoken_text": "A human-readable explanation for browser speech synthesis.",
  "analysis": {
    "model": "empwave-emotions-v2+semantic-intents"
  }
}
```

If no non-baseline region matches confidently, `regions` is empty and
`fallback` is `true`. `POST /api/process-speech` remains available for
backward compatibility.

## 🎯 Demo Prompts

Try these for interesting brain activation patterns:

- "Listen to my voice and understand what I'm saying"
- "I remember happy memories from my childhood"
- "Let me think about this logically and plan my approach"
- "I see colors and beautiful images"
- "I feel excited and want to dance"
- "Tell me a story about a journey"

## 🌐 Browser Compatibility

| Browser | Support |
|---------|---------|
| Chrome  | ✅ Full |
| Firefox | ✅ Full |
| Safari  | ✅ Full |
| Edge    | ✅ Full |

## 📄 License

This project is open source and available for educational and creative purposes.

## 🚀 Future Enhancements

- Real speech recognition (instead of just synthesis)
- EEG-style frequency visualization
- Multiple brain models (detailed anatomical, stylized, abstract)
- Sound visualization synced to audio
- User preference saving
- Network multiplayer viewing
- Optional server-generated TTS audio

## 💡 Notes

- The text-to-speech voices vary by browser and operating system
- Particle effects are GPU-accelerated for smooth performance
- The brain's rotation is purely aesthetic and continuous
- Semantic analysis runs in the Flask backend; Three.js rendering and speech
  synthesis run in the browser

---

**Happy exploring!** 🧠✨
