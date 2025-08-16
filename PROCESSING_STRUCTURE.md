# Video Processing Structure

## Overview
After processing a video, the system creates a well-organized folder structure in the `processed/` directory.

## Folder Structure
```
processed/
└── {video_id}/
    ├── keyframes/           # All extracted keyframes
    ├── audio/              # TTS audio files for each topic
    ├── videos/             # Generated summary videos
    ├── metadata.json       # Complete video metadata
    └── topic_{n}_metadata.json  # Individual topic metadata
```

## File Organization

### Keyframes Directory
- Contains all extracted keyframes from the video
- Each keyframe includes timestamp, OCR text, and confidence scores
- Organized by extraction order and topic relevance

### Audio Directory
- TTS-generated audio summaries for each topic
- Files named: `topic_0_summary.wav`, `topic_1_summary.wav`, etc.
- High-quality audio optimized for video integration

### Videos Directory
- Generated summary videos for each topic
- Files named: `topic_0_summary.mp4`, `topic_1_summary.mp4`, etc.
- Each video contains relevant keyframes synchronized with audio

### Metadata Files
- `metadata.json`: Complete video processing information
- `topic_{n}_metadata.json`: Individual topic details including:
  - Start/end times
  - Summary text
  - Keywords
  - Keyframe count
  - Associated files

## Benefits of New Structure

1. **Better Organization**: All related files are grouped by video ID
2. **Easier Management**: Clear separation of different file types
3. **Scalability**: Easy to add new processing outputs
4. **Debugging**: Better error tracking and file location
5. **User Experience**: Faster file access and download

## Processing Flow

1. **Video Upload/Download** → Creates unique video ID
2. **Transcription** → Generates text segments with timestamps
3. **Keyframe Extraction** → Extracts visual content with OCR
4. **Topic Clustering** → Groups content into meaningful topics
5. **Summary Generation** → Creates text summaries for each topic
6. **TTS Generation** → Converts summaries to audio
7. **Video Creation** → Combines keyframes and audio into videos
8. **File Organization** → Structures all outputs in organized folders

## Key Improvements

- **More Keyframes**: 30+ keyframes per topic for smooth video feel
- **Better Clustering**: Improved K-means clustering with similarity thresholds
- **Organized Storage**: Logical folder structure for easy access
- **Enhanced Metadata**: Comprehensive information about each processing step
- **Efficient Processing**: Optimized algorithms for faster completion
