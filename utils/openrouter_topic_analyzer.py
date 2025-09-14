"""
OpenRouter Topic Analyzer

This module provides intelligent topic analysis using OpenRouter models:
1. Clusters transcript segments into coherent topics
2. Generates meaningful topic names and descriptions
3. Creates comprehensive summaries for each topic
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import time

from .openrouter_client import OpenRouterClient

logger = logging.getLogger(__name__)

class OpenRouterTopicAnalyzer:
    """Advanced topic analysis using OpenRouter models"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the analyzer
        
        Args:
            config: Configuration dictionary with OpenRouter settings
        """
        self.config = config or {}
        
        # Initialize OpenRouter client if API key is available
        self.openrouter_client = None
        self.use_openrouter = self.config.get("USE_OPENROUTER", True)
        
        if self.use_openrouter:
            try:
                self.openrouter_client = OpenRouterClient()
                logger.info("OpenRouter client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenRouter client: {e}")
                self.use_openrouter = False
    
    def analyze_segments(
        self, 
        segments: List[Dict[str, Any]], 
        output_dir: str,
        min_topics: int = 3,
        max_topics: int = 8
    ) -> Tuple[List[Dict[str, Any]], str, str]:
        """
        Analyze transcript segments and cluster them into topics
        
        Args:
            segments: List of transcript segments with 'text', 'start', 'end'
            output_dir: Directory to save analysis results
            min_topics: Minimum number of topics to generate
            max_topics: Maximum number of topics to generate
            
        Returns:
            Tuple of (topics_list, timestamps_file, segments_file)
        """
        try:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            logger.info(f"Analyzing {len(segments)} segments for topic clustering")
            
            # Use OpenRouter for intelligent clustering if available
            if self.use_openrouter and self.openrouter_client:
                topics = self._analyze_with_openrouter(segments, min_topics, max_topics)
            else:
                logger.info("OpenRouter not available, using fallback analysis")
                topics = self._fallback_analysis(segments, min_topics, max_topics)
            
            # Save results
            timestamps_file = self._save_timestamps(topics, output_dir)
            segments_file = self._save_segments_with_topics(segments, topics, output_dir)
            
            logger.info(f"Topic analysis completed. Generated {len(topics)} topics")
            return topics, timestamps_file, segments_file
            
        except Exception as e:
            logger.error(f"Topic analysis failed: {e}")
            raise
    
    def _analyze_with_openrouter(
        self, 
        segments: List[Dict[str, Any]], 
        min_topics: int, 
        max_topics: int
    ) -> List[Dict[str, Any]]:
        """Analyze topics using OpenRouter models"""
        try:
            logger.info("Using OpenRouter for topic clustering")
            
            # Use OpenRouter client to cluster topics
            topics = self.openrouter_client.cluster_text_into_topics(segments)
            
            # Validate and adjust topic count
            if len(topics) < min_topics:
                logger.info(f"Got {len(topics)} topics, less than minimum {min_topics}. Using fallback.")
                return self._fallback_analysis(segments, min_topics, max_topics)
            elif len(topics) > max_topics:
                logger.info(f"Got {len(topics)} topics, more than maximum {max_topics}. Keeping top {max_topics}")
                topics = topics[:max_topics]
            
            # Enhance topics with additional metadata
            enhanced_topics = []
            for i, topic in enumerate(topics):
                # Ensure all required fields
                enhanced_topic = {
                    "id": i + 1,
                    "name": topic.get("name", f"Topic {i + 1}"),
                    "description": topic.get("description", "Generated topic"),
                    "start_time": topic.get("start_time", 0),
                    "end_time": topic.get("end_time", segments[-1]["end"] if segments else 0),
                    "keywords": topic.get("keywords", []),
                    "segment_indices": topic.get("segment_indices", [])
                }
                
                # Calculate actual duration
                enhanced_topic["duration"] = enhanced_topic["end_time"] - enhanced_topic["start_time"]
                
                # Ensure segment indices are valid
                valid_indices = [idx for idx in enhanced_topic["segment_indices"] if 0 <= idx < len(segments)]
                enhanced_topic["segment_indices"] = valid_indices
                
                # If no valid indices, try to map by timestamp
                if not valid_indices:
                    enhanced_topic["segment_indices"] = self._map_segments_by_time(
                        segments, enhanced_topic["start_time"], enhanced_topic["end_time"]
                    )
                
                enhanced_topics.append(enhanced_topic)
            
            return enhanced_topics
            
        except Exception as e:
            logger.error(f"OpenRouter topic analysis failed: {e}")
            return self._fallback_analysis(segments, min_topics, max_topics)
    
    def _fallback_analysis(
        self, 
        segments: List[Dict[str, Any]], 
        min_topics: int, 
        max_topics: int
    ) -> List[Dict[str, Any]]:
        """Fallback topic analysis using simple heuristics"""
        logger.info("Using fallback topic analysis")
        
        if not segments:
            return []
        
        # Simple time-based segmentation
        total_duration = segments[-1]["end"] - segments[0]["start"]
        num_topics = min(max_topics, max(min_topics, len(segments) // 10))
        
        if num_topics <= 1:
            num_topics = min_topics
        
        segment_duration = total_duration / num_topics
        topics = []
        
        for i in range(num_topics):
            start_time = segments[0]["start"] + i * segment_duration
            end_time = start_time + segment_duration
            
            # Map segments to this topic
            segment_indices = self._map_segments_by_time(segments, start_time, end_time)
            
            # Extract keywords from segment text
            keywords = self._extract_simple_keywords(
                [segments[idx]["text"] for idx in segment_indices if idx < len(segments)]
            )
            
            topics.append({
                "id": i + 1,
                "name": f"Topic {i + 1}",
                "description": f"Content from {start_time:.1f}s to {end_time:.1f}s",
                "start_time": start_time,
                "end_time": end_time,
                "duration": segment_duration,
                "keywords": keywords,
                "segment_indices": segment_indices
            })
        
        return topics
    
    def _map_segments_by_time(
        self, 
        segments: List[Dict[str, Any]], 
        start_time: float, 
        end_time: float
    ) -> List[int]:
        """Map segments to time range"""
        indices = []
        
        for i, segment in enumerate(segments):
            seg_start = segment.get("start", 0)
            seg_end = segment.get("end", 0)
            
            # Check if segment overlaps with time range
            if (start_time <= seg_start <= end_time or 
                start_time <= seg_end <= end_time or
                (seg_start <= start_time and seg_end >= end_time)):
                indices.append(i)
        
        return indices
    
    def _extract_simple_keywords(self, texts: List[str]) -> List[str]:
        """Extract simple keywords from text"""
        try:
            if not texts:
                return []
            
            # Combine all text
            combined_text = " ".join(texts).lower()
            
            # Simple keyword extraction using word frequency
            import re
            from collections import Counter
            
            # Remove punctuation and split into words
            words = re.findall(r'\b[a-zA-Z]{3,}\b', combined_text)
            
            # Common stopwords to remove
            stopwords = {
                'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
                'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
                'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
                'might', 'can', 'this', 'that', 'these', 'those', 'you', 'your', 'we',
                'our', 'they', 'their', 'them', 'one', 'two', 'three', 'said', 'say',
                'get', 'got', 'go', 'went', 'come', 'came', 'see', 'saw', 'know', 'knew'
            }
            
            # Filter words and count frequency
            filtered_words = [word for word in words if word not in stopwords and len(word) > 3]
            word_counts = Counter(filtered_words)
            
            # Return top keywords
            return [word for word, count in word_counts.most_common(5)]
            
        except Exception as e:
            logger.warning(f"Keyword extraction failed: {e}")
            return []
    
    def _save_timestamps(self, topics: List[Dict[str, Any]], output_dir: str) -> str:
        """Save topic timestamps to file"""
        timestamps_file = os.path.join(output_dir, "topic_timestamps.json")
        
        try:
            timestamps_data = {
                "topics": [{
                    "id": topic["id"],
                    "name": topic["name"],
                    "start_time": topic["start_time"],
                    "end_time": topic["end_time"],
                    "duration": topic["duration"]
                } for topic in topics]
            }
            
            with open(timestamps_file, 'w', encoding='utf-8') as f:
                json.dump(timestamps_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved topic timestamps to {timestamps_file}")
            return timestamps_file
            
        except Exception as e:
            logger.error(f"Failed to save timestamps: {e}")
            return ""
    
    def _save_segments_with_topics(
        self, 
        segments: List[Dict[str, Any]], 
        topics: List[Dict[str, Any]], 
        output_dir: str
    ) -> str:
        """Save segments with topic assignments"""
        segments_file = os.path.join(output_dir, "segments_with_topics.json")
        
        try:
            # Create a copy of segments and add topic assignments
            enhanced_segments = []
            
            for i, segment in enumerate(segments):
                enhanced_segment = segment.copy()
                
                # Find which topic this segment belongs to
                for topic in topics:
                    if i in topic.get("segment_indices", []):
                        enhanced_segment["topic_id"] = topic["id"]
                        enhanced_segment["topic_name"] = topic["name"]
                        break
                
                enhanced_segments.append(enhanced_segment)
            
            with open(segments_file, 'w', encoding='utf-8') as f:
                json.dump(enhanced_segments, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved enhanced segments to {segments_file}")
            return segments_file
            
        except Exception as e:
            logger.error(f"Failed to save enhanced segments: {e}")
            return ""

class OpenRouterSummarizer:
    """Generate summaries for topics using OpenRouter"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.openrouter_client = None
        self.use_openrouter = self.config.get("USE_OPENROUTER", True)
        
        if self.use_openrouter:
            try:
                self.openrouter_client = OpenRouterClient()
                logger.info("OpenRouter summarizer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenRouter for summarization: {e}")
                self.use_openrouter = False
    
    def summarize_topics(
        self, 
        topics: List[Dict[str, Any]], 
        segments: List[Dict[str, Any]],
        output_dir: str
    ) -> str:
        """
        Generate summaries for all topics
        
        Args:
            topics: List of topic dictionaries
            segments: Original transcript segments
            output_dir: Directory to save summaries
            
        Returns:
            Path to summaries file
        """
        try:
            summaries_file = os.path.join(output_dir, "topic_summaries.json")
            
            if self.use_openrouter and self.openrouter_client:
                summaries = self._generate_openrouter_summaries(topics, segments)
            else:
                summaries = self._generate_fallback_summaries(topics, segments)
            
            # Save summaries
            summaries_data = []
            for topic in topics:
                topic_id = str(topic["id"])
                summary = summaries.get(topic_id, f"Summary for {topic['name']}")
                
                summaries_data.append({
                    "topic_id": topic["id"],
                    "name": topic["name"],
                    "description": topic.get("description", ""),
                    "start_time": topic["start_time"],
                    "end_time": topic["end_time"],
                    "duration": topic["duration"],
                    "keywords": topic.get("keywords", []),
                    "summary": summary
                })
            
            with open(summaries_file, 'w', encoding='utf-8') as f:
                json.dump(summaries_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved topic summaries to {summaries_file}")
            return summaries_file
            
        except Exception as e:
            logger.error(f"Topic summarization failed: {e}")
            raise
    
    def _generate_openrouter_summaries(
        self, 
        topics: List[Dict[str, Any]], 
        segments: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Generate summaries using OpenRouter"""
        logger.info("Generating summaries using OpenRouter")
        return self.openrouter_client.generate_all_summaries(topics, segments)
    
    def _generate_fallback_summaries(
        self, 
        topics: List[Dict[str, Any]], 
        segments: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Generate enhanced fallback summaries with basic error correction"""
        logger.info("Generating enhanced fallback summaries")
        summaries = {}
        
        for topic in topics:
            topic_id = str(topic["id"])
            
            # Extract text from segments
            topic_segments = [
                segments[i] for i in topic.get("segment_indices", []) 
                if i < len(segments)
            ]
            
            if topic_segments:
                # Get all text from topic segments
                all_text = " ".join([seg["text"] for seg in topic_segments])
                
                # Basic error correction and cleanup
                cleaned_text = self._basic_text_cleanup(all_text)
                
                # Create a more structured summary
                if len(cleaned_text) > 400:
                    # Take first part and add continuation
                    summary = cleaned_text[:350].rsplit('.', 1)[0] + "."
                    if not summary.endswith('.'):
                        summary += "..."
                else:
                    summary = cleaned_text
                
                # Add topic context
                summary = f"This section covers {topic['name'].lower()}. {summary}"
                
                if topic.get("keywords"):
                    summary += f"\n\nKey concepts discussed: {', '.join(topic['keywords'])}"
                
                summaries[topic_id] = summary
            else:
                summaries[topic_id] = f"This section focuses on {topic['name'].lower()}. Content analysis is being processed to provide a detailed summary."
        
        return summaries
    
    def _basic_text_cleanup(self, text: str) -> str:
        """Apply basic text cleanup and error correction"""
        try:
            import re
            
            # Basic cleanup
            cleaned = text.strip()
            
            # Fix common transcription errors - ESPECIALLY ALGORITHM NAMES
            common_fixes = {
                # CRITICAL: Algorithm name corrections
                r'\bbubble soar\b': 'bubble sort',
                r'\bquick soar\b': 'quicksort',
                r'\bquicksor\b': 'quicksort',
                r'\bquick sor\b': 'quicksort',
                r'\btim soar\b': 'timsort',
                r'\bheap soar\b': 'heapsort',
                r'\bheapsor\b': 'heapsort',
                r'\bmerge soar\b': 'merge sort',
                r'\binsertion soar\b': 'insertion sort',
                r'\bselection soar\b': 'selection sort',
                
                # Other technical terms
                r'\balgorythm\b': 'algorithm',
                r'\balgorthm\b': 'algorithm', 
                r'\bcomparsion\b': 'comparison',
                r'\bperformence\b': 'performance',
                r'\befficienty\b': 'efficiency',
                r'\bimplemention\b': 'implementation',
                r'\bexampel\b': 'example',
                
                # Common words
                r'\bteh\b': 'the',
                r'\band and\b': 'and',
                r'\btha\b': 'the',
                r'\bis is\b': 'is',
                r'\bwill will\b': 'will',
                r'\bcan can\b': 'can',
                
                # Fix repeated words
                r'\b(\w+)\s+\1\b': r'\1',
                # Fix spacing around punctuation
                r'\s+([,.!?])': r'\1',
                r'([,.!?])\s*([A-Z])': r'\1 \2'
            }
            
            for pattern, replacement in common_fixes.items():
                cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
            
            # Fix multiple spaces
            cleaned = re.sub(r'\s+', ' ', cleaned)
            
            # Ensure proper sentence capitalization
            sentences = cleaned.split('. ')
            capitalized_sentences = []
            for sentence in sentences:
                if sentence:
                    sentence = sentence.strip()
                    if sentence:
                        sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
                        capitalized_sentences.append(sentence)
            
            cleaned = '. '.join(capitalized_sentences)
            
            # Ensure proper ending
            if cleaned and not cleaned.endswith(('.', '!', '?')):
                cleaned += '.'
                
            return cleaned
            
        except Exception as e:
            logger.warning(f"Text cleanup failed: {e}")
            return text.strip()

def analyze_video_topics(
    segments: List[Dict[str, Any]], 
    output_dir: str,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Main function to analyze video topics and generate summaries
    
    Args:
        segments: List of transcript segments
        output_dir: Output directory for results
        config: Configuration dictionary
        
    Returns:
        Tuple of (topics_with_summaries, summaries_file_path)
    """
    try:
        # Initialize analyzer and summarizer
        analyzer = OpenRouterTopicAnalyzer(config)
        summarizer = OpenRouterSummarizer(config)
        
        # Analyze topics
        topics, timestamps_file, segments_file = analyzer.analyze_segments(
            segments, output_dir
        )
        
        # Generate summaries
        summaries_file = summarizer.summarize_topics(topics, segments, output_dir)
        
        # Load and return summaries
        with open(summaries_file, 'r', encoding='utf-8') as f:
            topic_summaries = json.load(f)
        
        return topic_summaries, summaries_file
        
    except Exception as e:
        logger.error(f"Video topic analysis failed: {e}")
        raise