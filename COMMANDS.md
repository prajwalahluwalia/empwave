# 🔧 SynapseView - Essential Commands Reference

## 🚀 Starting the Application

### Start the Flask Server
```bash
cd /Users/prajwalahluwalia/Desktop/empwave
python3 app.py
```

**Expected Output:**
```
* Serving Flask app 'app'
* Debug mode: on
* Running on http://127.0.0.1:5007/model
* Press CTRL+C to quit
```

### Access in Browser
```
http://localhost:5000
```

---

## 🛑 Stopping the Server

### Stop with Keyboard (while running)
```
Press: CTRL+C
```

### Stop Running Process
```bash
# Find Python process on port 5000
lsof -i :5000

# Kill the process (replace <PID> with actual process ID)
kill <PID>

# Force kill if needed
kill -9 <PID>
```

---

## 📦 Dependency Management

### Install Flask
```bash
pip install flask
```

### Check Flask Installation
```bash
pip show flask
pip list | grep -i flask
```

### Create requirements.txt (optional)
```bash
pip freeze > requirements.txt
```

### Install from requirements.txt
```bash
pip install -r requirements.txt
```

---

## 🧪 API Testing Commands

### Test Default Input (with curl)
```bash
curl -X POST http://localhost:5000/api/process-speech \
  -H "Content-Type: application/json" \
  -d '{"text": "i love your shoes"}'
```

### Test with High Emotion
```bash
curl -X POST http://localhost:5000/api/process-speech \
  -H "Content-Type: application/json" \
  -d '{"text": "That is absolutely amazing and wonderful!"}'
```

### Test with Cognition Keywords
```bash
curl -X POST http://localhost:5000/api/process-speech \
  -H "Content-Type: application/json" \
  -d '{"text": "Let me think about this logically"}'
```

### Pretty Print JSON Response
```bash
curl -s -X POST http://localhost:5000/api/process-speech \
  -H "Content-Type: application/json" \
  -d '{"text": "hello"}' | python3 -m json.tool
```

### Test with Python requests
```python
import requests
import json

response = requests.post(
    'http://localhost:5000/api/process-speech',
    json={'text': 'hello world'},
    headers={'Content-Type': 'application/json'}
)
print(json.dumps(response.json(), indent=2))
```

---

## 📊 Port Management

### Check if Port 5000 is in Use
```bash
lsof -i :5000
netstat -an | grep :5000
ss -tuln | grep :5000
```

### Use Different Port
Edit `app.py`, last line:
```python
if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Change 5000 to your desired port
```

### Kill All Python Processes on Port 5000
```bash
lsof -ti:5000 | xargs kill -9
```

---

## 🔍 Debugging & Inspection

### View Browser Console
1. Open browser: http://localhost:5000
2. Press: `F12` or `Cmd+Option+I` (Mac)
3. Go to "Console" tab to see logs/errors

### View Server Logs
```bash
# Server output appears in terminal where you ran:
python3 app.py

# Look for:
# - Request logs: GET / HTTP/1.1
# - API calls: POST /api/process-speech HTTP/1.1
# - Errors: [ERROR] ...
```

### Check WebGL Support
```javascript
// Type in browser console (F12):
console.log(!!window.WebGLRenderingContext)
```

### Monitor Network Requests
1. Browser Console (F12)
2. Go to "Network" tab
3. Reload page or click Play button
4. Observe API call to `/api/process-speech`

---

## 📝 File Editing Commands

### View Files
```bash
# View app.py
cat /Users/prajwalahluwalia/Desktop/empwave/app.py

# View index.html (first 50 lines)
head -50 /Users/prajwalahluwalia/Desktop/empwave/templates/index.html

# View entire index.html
cat /Users/prajwalahluwalia/Desktop/empwave/templates/index.html | less
```

### Edit Files
```bash
# Edit with nano (easy for beginners)
nano /Users/prajwalahluwalia/Desktop/empwave/app.py

# Edit with vim (advanced)
vim /Users/prajwalahluwalia/Desktop/empwave/app.py

# Edit with VS Code
code /Users/prajwalahluwalia/Desktop/empwave/

# Edit with your default editor
open /Users/prajwalahluwalia/Desktop/empwave/app.py
```

---

## 📂 Project Navigation

### Navigate to Project Directory
```bash
cd /Users/prajwalahluwalia/Desktop/empwave
```

### List Files
```bash
# Simple list
ls -la

# Show file sizes
ls -lh

# Show only Python files
ls -la *.py

# Show directory tree
tree -L 2
find . -type f -name "*.py" -o -name "*.html" -o -name "*.md"
```

### Create New File
```bash
touch filename.txt
nano filename.txt  # Create and edit
```

### Delete Files
```bash
rm filename.txt
rm -f templates/old_index.html
```

---

## 🐍 Python Utilities

### Check Python Version
```bash
python3 --version
python --version
```

### Virtual Environment (Recommended)
```bash
# Create virtual environment
python3 -m venv venv

# Activate it (macOS/Linux)
source venv/bin/activate

# Activate it (Windows)
venv\Scripts\activate

# Deactivate when done
deactivate
```

### Run Python Script
```bash
python3 app.py
python app.py
```

### Interactive Python Shell
```bash
python3
# Type Python commands
# Exit with: exit() or Ctrl+D
```

---

## 🌐 Browser Testing

### Test in Different Browsers
```bash
# Chrome (macOS)
open -a "Google Chrome" http://localhost:5000

# Firefox (macOS)
open -a Firefox http://localhost:5000

# Safari (macOS)
open -a Safari http://localhost:5000
```

### Test Responsive Design
1. Open in browser
2. Press F12 (Developer Tools)
3. Click device toggle icon (tablet icon)
4. Select different device/screen sizes

### Test Full-Screen Mode
```
Press: F11 (most browsers)
Esc to exit
```

---

## 🔐 Production Deployment

### Using Gunicorn (Production Server)
```bash
# Install
pip install gunicorn

# Run with 4 workers
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Run with custom port
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

### Enable HTTPS (Self-signed Certificate)
```bash
# Generate certificate (macOS/Linux)
openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365

# Run Flask with SSL
python3 -c "from app import app; app.run(ssl_context=('cert.pem', 'key.pem'))"
```

### Deploy to Heroku
```bash
# Install Heroku CLI
brew install heroku

# Login
heroku login

# Create app
heroku create synapseview-app

# Deploy
git push heroku main

# View logs
heroku logs --tail
```

---

## 📊 Performance Testing

### Monitor CPU/Memory Usage
```bash
# macOS Activity Monitor
open -a "Activity Monitor"

# Terminal (top command)
top -p <PID>

# Terminal (htop - if installed)
brew install htop
htop
```

### Measure API Response Time
```bash
# Simple timing
time curl -X POST http://localhost:5000/api/process-speech \
  -H "Content-Type: application/json" \
  -d '{"text": "hello"}'

# With detailed timing
curl -w "Total: %{time_total}s\n" \
  -X POST http://localhost:5000/api/process-speech \
  -H "Content-Type: application/json" \
  -d '{"text": "hello"}'
```

### Load Testing
```bash
# Install Apache Bench
brew install httpd

# Test with 1000 requests, 10 concurrent
ab -n 1000 -c 10 http://localhost:5000/

# Load test API endpoint
ab -n 100 -c 5 -p data.json -T application/json http://localhost:5000/api/process-speech
```

---

## 🐛 Common Issues & Fixes

### Issue: "Command not found: python3"
```bash
# Install Python
brew install python3

# Or use python instead
python --version
python app.py
```

### Issue: "ModuleNotFoundError: No module named 'flask'"
```bash
pip install flask
# or
pip3 install flask
```

### Issue: Permission Denied
```bash
# Make file executable
chmod +x app.py

# Run with python explicitly
python3 app.py
```

### Issue: "Address already in use"
```bash
# Kill process using port 5000
lsof -i :5000 | tail -1 | awk '{print $2}' | xargs kill -9

# Or use different port (edit app.py)
```

---

## 📚 Documentation Reference

### View Documentation Files
```bash
# Quick start
cat /Users/prajwalahluwalia/Desktop/empwave/QUICKSTART.txt

# Setup guide
cat /Users/prajwalahluwalia/Desktop/empwave/SETUP.md

# Summary
cat /Users/prajwalahluwalia/Desktop/empwave/SYNAPSEVIEW_SUMMARY.md

# This file
cat /Users/prajwalahluwalia/Desktop/empwave/COMMANDS.md
```

### Search Documentation
```bash
grep -r "activation_intensity" /Users/prajwalahluwalia/Desktop/empwave/

grep -r "OrbitControls" /Users/prajwalahluwalia/Desktop/empwave/

grep -i "troubleshoot" /Users/prajwalahluwalia/Desktop/empwave/*.md
```

---

## 🔄 Git Commands (Optional)

### Initialize Git (if needed)
```bash
cd /Users/prajwalahluwalia/Desktop/empwave
git init
```

### Check Status
```bash
git status
```

### Add and Commit Changes
```bash
git add .
git commit -m "Update SynapseView"
```

### View Commit History
```bash
git log --oneline
```

### Create New Branch
```bash
git checkout -b feature-name
```

---

## 💾 Backup Commands

### Backup Project
```bash
# Compress entire project
tar -czf synapseview-backup.tar.gz /Users/prajwalahluwalia/Desktop/empwave/

# Or use zip
zip -r synapseview-backup.zip /Users/prajwalahluwalia/Desktop/empwave/
```

### List Backup
```bash
tar -tzf synapseview-backup.tar.gz | head -20
```

### Restore from Backup
```bash
tar -xzf synapseview-backup.tar.gz
```

---

## 🎯 Quick Command Shortcuts

```bash
# One-liner to start app
cd /Users/prajwalahluwalia/Desktop/empwave && python3 app.py

# One-liner test
curl -s -X POST http://localhost:5000/api/process-speech -H "Content-Type: application/json" -d '{"text": "i love your shoes"}' | python3 -m json.tool

# One-liner to find and kill process
lsof -ti:5000 | xargs kill -9

# One-liner to show all files
find /Users/prajwalahluwalia/Desktop/empwave -type f -not -path '*/\.*'

# One-liner to count lines of code
find /Users/prajwalahluwalia/Desktop/empwave -name "*.py" -o -name "*.html" | xargs wc -l
```

---

## 📞 Help Commands

### Python Help
```bash
python3 -m flask --help
python3 app.py --help
```

### Manual Pages
```bash
man curl
man python3
man lsof
```

### Online Resources
- Flask Docs: https://flask.palletsprojects.com/
- Three.js Docs: https://threejs.org/docs/
- Web Speech API: https://developer.mozilla.org/en-US/docs/Web/API/Web_Speech_API

---

**Save this file as a quick reference!** 🚀

