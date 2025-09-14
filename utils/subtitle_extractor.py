"""
Enhanced Subtitle Extractor

This module extracts subtitles from videos with precise timestamps and saves them in JSON format.
It supports multiple subtitle formats and provides detailed timestamp information for each sentence.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import subprocess
import re

logger = logging.getLogger(__name__)

class SubtitleExtractor:
    """Enhanced subtitle extraction with timestamp precision"""
    
    def __init__(self):
        self.supported_formats = ['.srt', '.vtt', '.ass', '.ssa', '.sub']
    
    def extract_subtitles_from_video(
        self, 
        video_path: str, 
        output_dir: str, 
        video_id: str,
        language: str = "auto"
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Extract subtitles from video and save in JSON format with timestamps
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save subtitle files
            video_id: Unique identifier for the video
            language: Language code for subtitle extraction
            
        Returns:
            Tuple of (subtitle_file_path, segments_list)
        """
        try:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # First, try to extract embedded subtitles
            embedded_subtitles = self._extract_embedded_subtitles(video_path, output_dir, video_id)
            
            if embedded_subtitles:
                logger.info(f"Found embedded subtitles: {embedded_subtitles}")
                segments = self._parse_subtitle_file(embedded_subtitles)
                if segments:
                    json_path = self._save_segments_to_json(segments, output_dir, video_id)
                    return json_path, segments
            
            # If no embedded subtitles, check if we already have transcription segments
            # This integrates with the existing transcription pipeline
            segments_file = os.path.join(output_dir, f"{video_id}_segments.json")
            if os.path.exists(segments_file):
                logger.info(f"Using existing transcription segments: {segments_file}")
                with open(segments_file, 'r', encoding='utf-8') as f:
                    segments = json.load(f)
                return segments_file, segments
            
            # If no existing segments, we need transcription (handled by existing pipeline)
            logger.info("No subtitles found. Transcription will be handled by existing pipeline.")
            return None, []
            
        except Exception as e:
            logger.error(f"Subtitle extraction failed: {e}")
            raise
    
    def _extract_embedded_subtitles(self, video_path: str, output_dir: str, video_id: str) -> Optional[str]:
        """Extract embedded subtitles using ffmpeg"""
        try:
            # First, check what subtitle streams are available
            probe_cmd = [
                'ffmpeg', '-i', video_path, '-hide_banner', 
                '-loglevel', 'error', '-select_streams', 's', 
                '-show_entries', 'stream=index:stream_tags=language', 
                '-of', 'csv=p=0'
            ]
            
            try:
                result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
                if result.returncode != 0:
                    logger.info("No subtitle streams found or ffprobe failed")
                    return None
            except subprocess.TimeoutExpired:
                logger.warning("Subtitle probe timed out")
                return None
            
            # Extract subtitle in SRT format (most compatible)
            subtitle_path = os.path.join(output_dir, f"{video_id}_subtitles.srt")
            extract_cmd = [
                'ffmpeg', '-i', video_path, '-map', '0:s:0', 
                '-c:s', 'srt', subtitle_path, '-y'
            ]
            
            result = subprocess.run(extract_cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0 and os.path.exists(subtitle_path):
                logger.info(f"Successfully extracted subtitles to: {subtitle_path}")
                return subtitle_path
            else:
                logger.info("No embedded subtitles found or extraction failed")
                return None
                
        except Exception as e:
            logger.warning(f"Embedded subtitle extraction failed: {e}")
            return None
    
    def _parse_subtitle_file(self, subtitle_path: str) -> List[Dict[str, Any]]:
        """Parse subtitle file and convert to segments format"""
        try:
            file_extension = Path(subtitle_path).suffix.lower()
            
            if file_extension == '.srt':
                return self._parse_srt_file(subtitle_path)
            elif file_extension == '.vtt':
                return self._parse_vtt_file(subtitle_path)
            else:
                logger.warning(f"Unsupported subtitle format: {file_extension}")
                return []
                
        except Exception as e:
            logger.error(f"Failed to parse subtitle file {subtitle_path}: {e}")
            return []
    
    def _parse_srt_file(self, srt_path: str) -> List[Dict[str, Any]]:
        """Parse SRT subtitle file"""
        segments = []
        
        try:
            with open(srt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split into subtitle blocks
            blocks = re.split(r'\n\s*\n', content.strip())
            
            for block in blocks:
                lines = block.strip().split('\n')
                if len(lines) < 3:
                    continue
                
                # Parse timing line (format: 00:00:00,000 --> 00:00:04,240)
                timing_line = lines[1]
                timing_match = re.match(
                    r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})',
                    timing_line
                )
                
                if not timing_match:
                    continue
                
                # Convert to seconds
                start_time = (
                    int(timing_match.group(1)) * 3600 +  # hours
                    int(timing_match.group(2)) * 60 +     # minutes
                    int(timing_match.group(3)) +          # seconds
                    int(timing_match.group(4)) / 1000     # milliseconds
                )
                
                end_time = (
                    int(timing_match.group(5)) * 3600 +   # hours
                    int(timing_match.group(6)) * 60 +     # minutes
                    int(timing_match.group(7)) +          # seconds
                    int(timing_match.group(8)) / 1000     # milliseconds
                )
                
                # Extract text (remove HTML tags if present)
                text = ' '.join(lines[2:])
                text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
                text = text.strip()
                
                if text:
                    segments.append({
                        "start": start_time,
                        "end": end_time,
                        "text": text,
                        "words": None  # Compatible with existing format
                    })
            
            logger.info(f"Parsed {len(segments)} segments from SRT file")
            return segments
            
        except Exception as e:
            logger.error(f"Failed to parse SRT file: {e}")
            return []
    
    def _parse_vtt_file(self, vtt_path: str) -> List[Dict[str, Any]]:
        """Parse WebVTT subtitle file"""
        segments = []
        
        try:
            with open(vtt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split into blocks and skip header
            lines = content.split('\n')
            current_segment = None
            
            for line in lines:
                line = line.strip()
                
                # Skip empty lines and headers
                if not line or line.startswith('WEBVTT') or line.startswith('NOTE'):
                    continue
                
                # Check if it's a timing line (format: 00:00.000 --> 00:04.240)
                timing_match = re.match(
                    r'(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{2}):(\d{2})\.(\d{3})',
                    line
                )
                
                if timing_match:
                    # Save previous segment if exists
                    if current_segment and current_segment.get('text'):
                        segments.append(current_segment)
                    
                    # Start new segment
                    start_time = (
                        int(timing_match.group(1)) * 60 +      # minutes
                        int(timing_match.group(2)) +           # seconds
                        int(timing_match.group(3)) / 1000      # milliseconds
                    )
                    
                    end_time = (
                        int(timing_match.group(4)) * 60 +      # minutes
                        int(timing_match.group(5)) +           # seconds
                        int(timing_match.group(6)) / 1000      # milliseconds
                    )
                    
                    current_segment = {
                        "start": start_time,
                        "end": end_time,
                        "text": "",
                        "words": None
                    }
                elif current_segment is not None:
                    # Add text to current segment
                    text = re.sub(r'<[^>]+>', '', line)  # Remove HTML tags
                    if text:
                        if current_segment['text']:
                            current_segment['text'] += ' ' + text
                        else:
                            current_segment['text'] = text
            
            # Don't forget the last segment
            if current_segment and current_segment.get('text'):
                segments.append(current_segment)
            
            logger.info(f"Parsed {len(segments)} segments from VTT file")
            return segments
            
        except Exception as e:
            logger.error(f"Failed to parse VTT file: {e}")
            return []
    
    def _save_segments_to_json(self, segments: List[Dict[str, Any]], output_dir: str, video_id: str) -> str:
        """Save segments to JSON file"""
        json_path = os.path.join(output_dir, f"{video_id}_segments.json")
        
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(segments, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(segments)} segments to {json_path}")
            return json_path
            
        except Exception as e:
            logger.error(f"Failed to save segments to JSON: {e}")
            raise
    
    def enhance_segments_with_sentences(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enhance segments by splitting them into individual sentences with timestamps
        """
        enhanced_segments = []
        
        try:
            import nltk
            from nltk.tokenize import sent_tokenize
            
            # Ensure NLTK data is available
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
            
            for segment in segments:
                text = segment.get('text', '').strip()
                if not text:
                    continue
                
                # Split into sentences
                sentences = sent_tokenize(text)
                
                if len(sentences) <= 1:
                    # Single sentence, keep as is
                    enhanced_segments.append(segment)
                else:
                    # Multiple sentences, split proportionally
                    total_duration = segment['end'] - segment['start']
                    total_chars = sum(len(s) for s in sentences)
                    
                    current_time = segment['start']
                    
                    for sentence in sentences:
                        # Proportional time allocation based on character count
                        sentence_duration = (len(sentence) / total_chars) * total_duration
                        sentence_end = current_time + sentence_duration
                        
                        enhanced_segments.append({
                            'start': round(current_time, 2),
                            'end': round(sentence_end, 2),
                            'text': sentence.strip(),
                            'words': None,
                            'original_segment': segment  # Reference to original
                        })
                        
                        current_time = sentence_end
            
            logger.info(f"Enhanced {len(segments)} segments into {len(enhanced_segments)} sentence-level segments")
            return enhanced_segments
            
        except Exception as e:
            logger.warning(f"Sentence enhancement failed, returning original segments: {e}")
            return segments

def extract_and_save_subtitles(
    video_path: str, 
    output_dir: str, 
    video_id: str, 
    language: str = "auto",
    enhance_sentences: bool = True
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Main function to extract subtitles and save in JSON format
    
    Args:
        video_path: Path to video file
        output_dir: Output directory
        video_id: Video identifier
        language: Language code
        enhance_sentences: Whether to split into sentence-level segments
        
    Returns:
        Tuple of (json_file_path, segments_list)
    """
    extractor = SubtitleExtractor()
    
    # Extract subtitles
    json_path, segments = extractor.extract_subtitles_from_video(
        video_path, output_dir, video_id, language
    )
    
    # Enhance with sentence-level splitting if requested
    if enhance_sentences and segments:
        segments = extractor.enhance_segments_with_sentences(segments)
        
        # Re-save enhanced segments
        if json_path:
            json_path = extractor._save_segments_to_json(segments, output_dir, video_id)
    
    return json_path, segments