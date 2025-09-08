"""
Advanced Production-Ready Summarization System

This module provides state-of-the-art summarization capabilities using the latest
Hugging Face models with production optimizations, caching, and error handling.

Features:
- Multiple advanced summarization models (BART, T5, Pegasus, etc.)
- Automatic model selection based on content type and length
- Intelligent chunking and hierarchical summarization
- Production-ready caching and performance monitoring
- Support for multiple languages with specialized models
- GPU acceleration with fallback to CPU
- Quality scoring and automatic model switching

Author: Video Summarizer Team
Created: 2024
"""

import os
import logging
import re
import json
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path
import requests
import torch
from transformers import (
    pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, 
    BartForConditionalGeneration, T5ForConditionalGeneration,
    PegasusForConditionalGeneration, AutoModel
)
from sentence_transformers import SentenceTransformer
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk
from .gpu_config import get_device, is_gpu_available, optimize_for_model, clear_gpu_memory, log_gpu_status

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    logger.info("Downloading NLTK data...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

class ProductionSummarizer:
    """Advanced production-ready summarization system."""
    
    def __init__(self, cache_dir: str = "summarization_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Model configurations for different use cases
        self.model_configs = {
            'bart_large_cnn': {
                'model_name': 'facebook/bart-large-cnn',
                'max_length': 1024,
                'min_length': 50,
                'best_for': 'news_articles',
                'quality_score': 9.5,
                'speed_score': 7.0
            },
            'bart_large_xsum': {
                'model_name': 'facebook/bart-large-xsum',
                'max_length': 1024,
                'min_length': 30,
                'best_for': 'abstractive_summary',
                'quality_score': 9.0,
                'speed_score': 7.0
            },
            't5_large': {
                'model_name': 't5-large',
                'max_length': 512,
                'min_length': 30,
                'best_for': 'general_text',
                'quality_score': 8.5,
                'speed_score': 6.0
            },
            'pegasus_large': {
                'model_name': 'google/pegasus-large',
                'max_length': 1024,
                'min_length': 30,
                'best_for': 'news_summary',
                'quality_score': 9.2,
                'speed_score': 5.5
            },
            'distilbart_cnn': {
                'model_name': 'sshleifer/distilbart-cnn-12-6',
                'max_length': 1024,
                'min_length': 50,
                'best_for': 'fast_summary',
                'quality_score': 7.5,
                'speed_score': 9.0
            },
            'mt5_multilingual': {
                'model_name': 'google/mt5-large',
                'max_length': 512,
                'min_length': 30,
                'best_for': 'multilingual',
                'quality_score': 8.0,
                'speed_score': 6.5
            }
        }
        
        # Language-specific model preferences
        self.language_models = {
            'en': ['bart_large_cnn', 'bart_large_xsum', 'pegasus_large'],
            'hi': ['mt5_multilingual', 't5_large'],
            'es': ['mt5_multilingual', 't5_large'],
            'fr': ['mt5_multilingual', 't5_large'],
            'de': ['mt5_multilingual', 't5_large'],
            'zh': ['mt5_multilingual', 't5_large'],
            'ja': ['mt5_multilingual', 't5_large'],
            'ko': ['mt5_multilingual', 't5_large'],
            'ar': ['mt5_multilingual', 't5_large'],
            'ru': ['mt5_multilingual', 't5_large'],
            'pt': ['mt5_multilingual', 't5_large']
        }
        
        # Initialize models cache
        self.models = {}
        self.sentence_transformer = None
        self.device = get_device()
        
        # Performance monitoring
        self.performance_stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'model_loads': 0,
            'average_time': 0.0,
            'error_count': 0
        }
        
        # Load sentence transformer for similarity
        self._load_sentence_transformer()
        
        logger.info(f"ProductionSummarizer initialized on {self.device}")
        log_gpu_status()
    
    def _load_sentence_transformer(self):
        """Load sentence transformer for text similarity."""
        try:
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            if self.device == 'cuda':
                self.sentence_transformer = self.sentence_transformer.to(self.device)
            logger.info("Sentence transformer loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load sentence transformer: {e}")
    
    def _get_cache_key(self, text: str, model_name: str, params: Dict) -> str:
        """Generate cache key for text and parameters."""
        content_hash = hashlib.md5(text.encode()).hexdigest()
        param_hash = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()
        return f"{model_name}_{content_hash}_{param_hash}"
    
    def _load_model(self, model_key: str) -> Optional[Any]:
        """Load and cache model."""
        if model_key in self.models:
            return self.models[model_key]
        
        try:
            config = self.model_configs[model_key]
            model_name = config['model_name']
            
            logger.info(f"Loading model: {model_name}")
            self.performance_stats['model_loads'] += 1
            
            # Get GPU optimizations
            optimizations = optimize_for_model(model_name)
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            if 'bart' in model_name.lower():
                model = BartForConditionalGeneration.from_pretrained(model_name)
            elif 't5' in model_name.lower():
                model = T5ForConditionalGeneration.from_pretrained(model_name)
            elif 'pegasus' in model_name.lower():
                model = PegasusForConditionalGeneration.from_pretrained(model_name)
            else:
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            # Move to device
            model = model.to(self.device)
            
            # Create pipeline
            summarizer = pipeline(
                "summarization",
                model=model,
                tokenizer=tokenizer,
                device=0 if self.device == 'cuda' else -1,
                framework='pt',
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
            )
            
            self.models[model_key] = {
                'pipeline': summarizer,
                'config': config,
                'optimizations': optimizations,
                'loaded_at': time.time()
            }
            
            logger.info(f"Model {model_name} loaded successfully on {self.device}")
            return self.models[model_key]
            
        except Exception as e:
            logger.error(f"Failed to load model {model_key}: {e}")
            return None
    
    def _select_best_model(self, text: str, language: str = 'en', 
                          content_type: str = 'general') -> str:
        """Select the best model for the given text and requirements."""
        
        # Get language-specific models
        available_models = self.language_models.get(language, self.language_models['en'])
        
        # Filter by content type preferences
        if content_type == 'news':
            preferred_models = ['bart_large_cnn', 'pegasus_large', 'bart_large_xsum']
        elif content_type == 'academic':
            preferred_models = ['bart_large_xsum', 't5_large', 'pegasus_large']
        elif content_type == 'conversational':
            preferred_models = ['bart_large_cnn', 'distilbart_cnn', 't5_large']
        else:
            preferred_models = ['bart_large_cnn', 'bart_large_xsum', 't5_large']
        
        # Find intersection of available and preferred models
        suitable_models = [m for m in available_models if m in preferred_models]
        
        if not suitable_models:
            suitable_models = available_models
        
        # Select based on text length and quality requirements
        text_length = len(text.split())
        
        if text_length > 2000:
            # Long text - prioritize quality
            suitable_models.sort(key=lambda x: self.model_configs[x]['quality_score'], reverse=True)
        else:
            # Short text - balance quality and speed
            suitable_models.sort(key=lambda x: 
                self.model_configs[x]['quality_score'] * 0.7 + 
                self.model_configs[x]['speed_score'] * 0.3, reverse=True)
        
        return suitable_models[0]
    
    def _chunk_text_intelligently(self, text: str, max_chunk_length: int = 1000) -> List[str]:
        """Intelligently chunk text preserving semantic boundaries."""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(word_tokenize(sentence))
            
            if current_length + sentence_length > max_chunk_length and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _calculate_summary_quality(self, original_text: str, summary: str) -> float:
        """Calculate quality score for summary."""
        if not self.sentence_transformer:
            return 0.5  # Default score if no similarity model
        
        try:
            # Calculate semantic similarity
            embeddings = self.sentence_transformer.encode([original_text, summary])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            
            # Calculate compression ratio (good summaries are 10-30% of original)
            compression_ratio = len(summary.split()) / len(original_text.split())
            compression_score = 1.0 - abs(compression_ratio - 0.2) / 0.2
            
            # Calculate coherence (sentence count vs word count)
            sentence_count = len(sent_tokenize(summary))
            word_count = len(summary.split())
            coherence_score = min(1.0, sentence_count / (word_count / 15))  # ~15 words per sentence
            
            # Combined quality score
            quality_score = (similarity * 0.5 + compression_score * 0.3 + coherence_score * 0.2)
            return min(1.0, max(0.0, quality_score))
            
        except Exception as e:
            logger.warning(f"Quality calculation failed: {e}")
            return 0.5
    
    def _hierarchical_summarize(self, text: str, model_key: str, 
                               target_length: int = 150) -> str:
        """Perform hierarchical summarization for long texts."""
        chunks = self._chunk_text_intelligently(text, max_chunk_length=800)
        
        if len(chunks) == 1:
            return self._summarize_chunk(chunks[0], model_key, target_length)
        
        # Summarize each chunk
        chunk_summaries = []
        for chunk in chunks:
            chunk_summary = self._summarize_chunk(chunk, model_key, target_length // len(chunks))
            chunk_summaries.append(chunk_summary)
        
        # Combine and re-summarize
        combined_summary = ' '.join(chunk_summaries)
        
        if len(combined_summary.split()) > target_length * 1.5:
            final_summary = self._summarize_chunk(combined_summary, model_key, target_length)
        else:
            final_summary = combined_summary
        
        return final_summary
    
    def _summarize_chunk(self, text: str, model_key: str, target_length: int) -> str:
        """Summarize a single text chunk."""
        model_data = self._load_model(model_key)
        if not model_data:
            return text[:target_length] + "..." if len(text) > target_length else text
        
        config = model_data['config']
        pipeline = model_data['pipeline']
        optimizations = model_data['optimizations']
        
        try:
            result = pipeline(
                text,
                min_length=max(30, target_length // 3),
                max_length=min(target_length, config['max_length']),
                num_beams=optimizations.get('num_beams', 4),
                early_stopping=optimizations.get('early_stopping', True),
                do_sample=False,
                truncation=True
            )
            
            return result[0]['summary_text']
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return text[:target_length] + "..." if len(text) > target_length else text
    
    def summarize(self, text: str, language: str = 'en', 
                 content_type: str = 'general', target_length: int = 150,
                 quality_threshold: float = 0.7, use_cache: bool = True) -> Dict[str, Any]:
        """
        Advanced summarization with automatic model selection and quality scoring.
        
        Args:
            text: Text to summarize
            language: Language code
            content_type: Type of content ('news', 'academic', 'conversational', 'general')
            target_length: Target summary length in words
            quality_threshold: Minimum quality score to accept
            use_cache: Whether to use caching
            
        Returns:
            Dictionary with summary, quality score, model used, and metadata
        """
        start_time = time.time()
        self.performance_stats['total_requests'] += 1
        
        # Check cache first
        if use_cache:
            cache_key = self._get_cache_key(text, f"{language}_{content_type}", {
                'target_length': target_length,
                'quality_threshold': quality_threshold
            })
            cache_file = self.cache_dir / f"{cache_key}.json"
            
            if cache_file.exists():
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)
                    
                    # Check if cache is still valid (24 hours)
                    if datetime.now().timestamp() - cached_data['timestamp'] < 86400:
                        self.performance_stats['cache_hits'] += 1
                        logger.info("Using cached summary")
                        return cached_data['result']
                except Exception as e:
                    logger.warning(f"Cache read failed: {e}")
        
        try:
            # Select best model
            model_key = self._select_best_model(text, language, content_type)
            
            # Perform summarization
            if len(text.split()) > 1500:
                summary = self._hierarchical_summarize(text, model_key, target_length)
            else:
                summary = self._summarize_chunk(text, model_key, target_length)
            
            # Calculate quality score
            quality_score = self._calculate_summary_quality(text, summary)
            
            # If quality is too low, try with a different model
            if quality_score < quality_threshold and len(text.split()) > 500:
                logger.info(f"Quality score {quality_score:.2f} below threshold, trying alternative model")
                
                # Try alternative models
                available_models = self.language_models.get(language, self.language_models['en'])
                alternative_models = [m for m in available_models if m != model_key]
                
                for alt_model in alternative_models[:2]:  # Try up to 2 alternatives
                    alt_summary = self._summarize_chunk(text, alt_model, target_length)
                    alt_quality = self._calculate_summary_quality(text, alt_summary)
                    
                    if alt_quality > quality_score:
                        summary = alt_summary
                        quality_score = alt_quality
                        model_key = alt_model
                        break
            
            # Prepare result
            result = {
                'summary': summary,
                'quality_score': quality_score,
                'model_used': model_key,
                'original_length': len(text.split()),
                'summary_length': len(summary.split()),
                'compression_ratio': len(summary.split()) / len(text.split()),
                'processing_time': time.time() - start_time,
                'language': language,
                'content_type': content_type,
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache result
            if use_cache:
                try:
                    cache_data = {
                        'result': result,
                        'timestamp': datetime.now().timestamp()
                    }
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        json.dump(cache_data, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    logger.warning(f"Cache write failed: {e}")
            
            # Update performance stats
            self.performance_stats['average_time'] = (
                (self.performance_stats['average_time'] * (self.performance_stats['total_requests'] - 1) + 
                 result['processing_time']) / self.performance_stats['total_requests']
            )
            
            logger.info(f"Summarization completed: {result['processing_time']:.2f}s, quality: {quality_score:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            self.performance_stats['error_count'] += 1
            
            # Fallback to simple truncation
            fallback_summary = text[:target_length * 6] + "..." if len(text) > target_length * 6 else text
            
            return {
                'summary': fallback_summary,
                'quality_score': 0.3,
                'model_used': 'fallback',
                'original_length': len(text.split()),
                'summary_length': len(fallback_summary.split()),
                'compression_ratio': len(fallback_summary.split()) / len(text.split()),
                'processing_time': time.time() - start_time,
                'language': language,
                'content_type': content_type,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
        
        finally:
            # Clear GPU memory
            clear_gpu_memory()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        cache_hit_rate = (
            self.performance_stats['cache_hits'] / 
            max(1, self.performance_stats['total_requests'])
        ) * 100
        
        return {
            **self.performance_stats,
            'cache_hit_rate': cache_hit_rate,
            'models_loaded': len(self.models),
            'device': self.device,
            'gpu_available': is_gpu_available()
        }
    
    def clear_cache(self, older_than_days: int = 7):
        """Clear old cache files."""
        cutoff_time = datetime.now() - timedelta(days=older_than_days)
        cutoff_timestamp = cutoff_time.timestamp()
        
        cleared_count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                if cache_file.stat().st_mtime < cutoff_timestamp:
                    cache_file.unlink()
                    cleared_count += 1
            except Exception as e:
                logger.warning(f"Failed to delete cache file {cache_file}: {e}")
        
        logger.info(f"Cleared {cleared_count} old cache files")
        return cleared_count

# Global instance
production_summarizer = ProductionSummarizer()

def summarize_text(text: str, language: str = 'en', content_type: str = 'general',
                  target_length: int = 150, quality_threshold: float = 0.7) -> Dict[str, Any]:
    """Convenience function for text summarization."""
    return production_summarizer.summarize(
        text, language, content_type, target_length, quality_threshold
    )

def get_summarization_stats() -> Dict[str, Any]:
    """Get summarization performance statistics."""
    return production_summarizer.get_performance_stats()
