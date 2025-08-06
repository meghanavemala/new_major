# 🚀 Quick Setup Guide for AI Video Summarizer

This guide will help you set up the AI Video Summarizer quickly and efficiently.

## 📋 Prerequisites Checklist

Before starting, ensure you have:

- [ ] Python 3.8 or higher
- [ ] At least 4GB RAM (8GB recommended)
- [ ] 10GB free disk space
- [ ] Internet connection for translation services
- [ ] Git installed

## 🔧 System Dependencies

### Windows
```bash
# Install FFmpeg
# Download from https://ffmpeg.org/download.html
# Add to system PATH

# Install Tesseract OCR
# Download from https://github.com/UB-Mannheim/tesseract/wiki
# Install with all language packs
```

### macOS
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install ffmpeg tesseract tesseract-lang
```

### Ubuntu/Debian
```bash
# Update package list
sudo apt update

# Install FFmpeg
sudo apt install ffmpeg

# Install Tesseract with Indian language support
sudo apt install tesseract-ocr tesseract-ocr-hin tesseract-ocr-ben tesseract-ocr-tam tesseract-ocr-tel tesseract-ocr-kan tesseract-ocr-mal tesseract-ocr-urd tesseract-ocr-guj tesseract-ocr-pan tesseract-ocr-ori tesseract-ocr-asm tesseract-ocr-mar tesseract-ocr-nep tesseract-ocr-san

# Install additional dependencies
sudo apt install python3-dev python3-pip python3-venv build-essential
```

## 📥 Installation Steps

### 1. Clone and Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/video-summarizer.git
cd video-summarizer

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 2. Install Python Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt_tab')"

# Test installation
python -c "import torch; import whisper; import cv2; print('✅ All dependencies installed successfully!')"
```

### 3. Verify System Tools
```bash
# Test FFmpeg
ffmpeg -version

# Test Tesseract
tesseract --version
tesseract --list-langs

# Test Python imports
python -c "
import whisper
import transformers
import pytesseract
import easyocr
print('✅ All AI models accessible!')
"
```

### 4. Create Configuration
```bash
# Create environment file
cat > .env << EOF
# Flask Configuration
FLASK_SECRET_KEY=your-secret-key-change-this
FLASK_ENV=production

# Processing Limits
MAX_CONTENT_LENGTH=104857600
MAX_VIDEO_DURATION=3600

# OCR Configuration
ENABLE_OCR=true

# Model Settings
WHISPER_MODEL_SIZE=medium
TRANSLATION_METHOD=google
EOF
```

### 5. Test Run
```bash
# Start the application
python app.py

# Check if it's running
curl http://localhost:5000
```

## 🌐 Browser Setup

1. Open your browser
2. Navigate to `http://localhost:5000`
3. You should see the AI Video Summarizer interface

## 🔍 Troubleshooting

### Common Issues and Solutions

#### ❌ "FFmpeg not found"
```bash
# Windows: Ensure FFmpeg is in PATH
where ffmpeg

# macOS/Linux: Check installation
which ffmpeg

# If not found, reinstall following system dependencies above
```

#### ❌ "No module named 'torch'"
```bash
# Reinstall PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### ❌ "Tesseract not found"
```bash
# Check Tesseract installation
tesseract --version

# If not found, install following system dependencies
# Then test Python integration:
python -c "import pytesseract; print(pytesseract.image_to_string('test.png'))"
```

#### ❌ "CUDA out of memory"
```bash
# Use CPU-only mode by setting in your environment:
export CUDA_VISIBLE_DEVICES=""
# Or modify model sizes in config to use smaller models
```

#### ❌ Translation errors
```bash
# Check internet connection
ping translate.googleapis.com

# Fallback to offline translation
# Edit app.py and set: TRANSLATION_METHOD=m2m100
```

#### ❌ Port already in use
```bash
# Check what's using port 5000
# Windows:
netstat -ano | findstr :5000
# macOS/Linux:
lsof -i :5000

# Kill the process or use different port:
python app.py --port 5001
```

## 🧪 Test With Sample Video

1. Download a short test video:
```bash
# Create test directory
mkdir test_videos
cd test_videos

# Download a sample video (or use your own)
# Upload it through the web interface
```

2. Test basic functionality:
   - Upload the video
   - Select source language (or auto-detect)
   - Select target language
   - Choose voice and quality settings
   - Click "Process Video"

3. Expected results:
   - Processing progress should show
   - Topics should be generated
   - Summary videos should be playable
   - Downloads should work

## ⚡ Performance Optimization

### For Better Speed:
```bash
# Use GPU if available
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Reduce model sizes for faster processing
# Edit the configuration to use smaller models:
# WHISPER_MODEL_SIZE=base
# Use 480p resolution instead of 1080p
```

### For Better Quality:
```bash
# Use larger models (slower but more accurate)
# WHISPER_MODEL_SIZE=large
# Enable all OCR features
# Use 1080p resolution
```

## 🔒 Production Deployment

### Security Setup:
```bash
# Generate secure secret key
python -c "import secrets; print(secrets.token_hex(32))"

# Update .env with the generated key
# Set FLASK_ENV=production
```

### Using Gunicorn:
```bash
# Install gunicorn
pip install gunicorn

# Run with gunicorn
gunicorn --workers 4 --bind 0.0.0.0:8000 app:app
```

### Using Docker:
```bash
# Build Docker image
docker build -t video-summarizer .

# Run container
docker run -p 5000:5000 -v $(pwd)/uploads:/app/uploads -v $(pwd)/processed:/app/processed video-summarizer
```

## 📊 Monitoring and Logs

### Check Application Logs:
```bash
# View recent logs
tail -f app.log

# Check for errors
grep ERROR app.log
```

### Monitor Performance:
```bash
# Check disk usage
df -h

# Check memory usage
free -h

# Monitor processing
ps aux | grep python
```

## 🆘 Getting Help

If you encounter issues:

1. **Check the logs**: Look at `app.log` for error messages
2. **Verify dependencies**: Ensure all system tools are installed
3. **Test components**: Run individual tests for each component
4. **Check GitHub Issues**: Look for similar problems
5. **Create an Issue**: Provide logs and system information

### System Information Template:
```bash
# Get system info for bug reports
echo "OS: $(uname -a)"
echo "Python: $(python --version)"
echo "FFmpeg: $(ffmpeg -version | head -1)"
echo "Tesseract: $(tesseract --version | head -1)"
echo "GPU: $(nvidia-smi | head -1 || echo 'No NVIDIA GPU')"
pip list | grep -E "(torch|whisper|opencv|transformers)"
```

## 🎉 You're Ready!

Congratulations! Your AI Video Summarizer is now set up and ready to use. Start by uploading a short video to test all features.

### Next Steps:
- Try different languages
- Experiment with voice options
- Test topic selection
- Explore download features
- Customize settings for your needs

Happy summarizing! 🎬✨