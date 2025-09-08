# GPU Optimization Setup Guide

This guide will help you set up GPU acceleration for your video summarizer project.

## Prerequisites

### 1. NVIDIA GPU Requirements
- NVIDIA GPU with CUDA Compute Capability 3.5 or higher
- NVIDIA drivers (version 450.80.02 or higher)
- CUDA Toolkit 12.1 or compatible version

### 2. Check GPU Availability
Run this command to check if your GPU is detected:
```bash
nvidia-smi
```

## Installation Steps

### 1. Install CUDA Toolkit (if not already installed)
Download and install CUDA Toolkit 12.1 from NVIDIA's website:
https://developer.nvidia.com/cuda-downloads

### 2. Install GPU-optimized PyTorch
The requirements.txt has been updated with GPU-specific PyTorch versions. Install them:

```bash
# Uninstall existing PyTorch if any
pip uninstall torch torchvision torchaudio

# Install GPU-optimized PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 
```

### 3. Install Additional GPU Dependencies
```bash
# Install other requirements
pip install -r requirements.txt

# Install additional GPU-accelerated libraries
pip install accelerate bitsandbytes
```

### 4. Verify GPU Installation
Run this Python script to verify GPU is working:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

## Configuration

### 1. Environment Variables (Optional)
You can set these environment variables to control GPU behavior:

```bash
# Force GPU usage (will fail if GPU not available)
export CUDA_VISIBLE_DEVICES=0

# Set GPU memory fraction (0.0 to 1.0)
export CUDA_MEMORY_FRACTION=0.9

# Enable mixed precision training
export TORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### 2. FFmpeg GPU Support
For video processing acceleration, ensure FFmpeg is compiled with NVIDIA support:

```bash
# Check if FFmpeg supports NVIDIA encoders
ffmpeg -encoders | grep nvenc

# If not available, install FFmpeg with NVIDIA support
# On Windows: Download from https://github.com/BtbN/FFmpeg-Builds
# On Linux: Install from your package manager or compile from source
```

## Usage

### 1. Automatic GPU Detection
The application will automatically detect and use GPU if available. You'll see logs like:
```
Using device: cuda
GPU: NVIDIA GeForce RTX 4090
GPU Memory: 24.0GB / 24.0GB
```

### 2. GPU Status Endpoint
Check GPU status via the API:
```bash
curl http://localhost:5000/api/gpu-status
```

### 3. Force CPU Usage (if needed)
If you want to force CPU usage, set:
```bash
export CUDA_VISIBLE_DEVICES=""
```

## Performance Optimizations

### 1. Model-Specific Optimizations
- **Whisper**: Uses mixed precision (FP16) on GPU for faster inference
- **BART/T5**: Uses optimized beam search and early stopping
- **M2M100**: Uses batch processing for translation
- **Video Encoding**: Uses NVIDIA NVENC for hardware-accelerated encoding

### 2. Memory Management
The application automatically:
- Clears GPU memory after each operation
- Uses optimal batch sizes based on available memory
- Implements memory fraction limits to prevent OOM errors

### 3. Monitoring GPU Usage
Monitor GPU usage during processing:
```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Or use the built-in status endpoint
curl http://localhost:5000/api/gpu-status
```

## Troubleshooting

### 1. CUDA Out of Memory Errors
If you encounter OOM errors:
- Reduce batch sizes in the code
- Set lower memory fraction: `export CUDA_MEMORY_FRACTION=0.7`
- Use smaller models (e.g., Whisper 'base' instead of 'large')

### 2. FFmpeg GPU Encoding Issues
If GPU encoding fails:
- Check FFmpeg installation: `ffmpeg -encoders | grep nvenc`
- Fallback to CPU encoding is automatic
- Update NVIDIA drivers if needed

### 3. Model Loading Issues
If models fail to load on GPU:
- Check CUDA compatibility: `torch.cuda.is_available()`
- Verify PyTorch installation: `torch.version.cuda`
- Try restarting the application

### 4. Performance Issues
If GPU performance is slower than expected:
- Check GPU utilization: `nvidia-smi`
- Ensure proper cooling and power supply
- Update NVIDIA drivers
- Check for thermal throttling

## Expected Performance Improvements

With GPU acceleration, you should see:
- **Transcription**: 3-5x faster with Whisper
- **Translation**: 2-3x faster with M2M100
- **Summarization**: 2-4x faster with BART
- **Video Encoding**: 2-3x faster with NVENC

## Support

If you encounter issues:
1. Check the logs for GPU-related messages
2. Verify GPU status with `nvidia-smi`
3. Test GPU functionality with the verification script
4. Check the `/api/gpu-status` endpoint for detailed information

The application will automatically fall back to CPU if GPU is not available, so it will work regardless of your hardware setup.
