import logging
import re
from typing import List, Dict, Any, Optional
import requests
from transformers import pipeline
import torch
from nltk.tokenize import sent_tokenize
import os
import json
from datetime import datetime
from .gpu_config import get_device, is_gpu_available, optimize_for_model, clear_gpu_memory, log_gpu_status

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hugging Face Inference API (Free tier)
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
# HuggingFace API token
HUGGINGFACE_API_TOKEN = "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"  # Replace with your token

# Fallback model configuration (when API fails or for other languages)
LOCAL_MODEL_NAME = "sshleifer/distilbart-cnn-12-6"  # Smaller, faster model
MAX_CHUNK_LENGTH = 1024  # Maximum length for each text chunk
MIN_SUMMARY_LENGTH = 50
MAX_SUMMARY_LENGTH = 150

class EnhancedSummarizer:
    def __init__(self):
        self.local_summarizer = None
        
    def _load_local_model(self):
        """Load the local model if not already loaded"""
        if self.local_summarizer is None:
            # Get optimized device and settings
            device = get_device()
            optimizations = optimize_for_model(LOCAL_MODEL_NAME)
            
            logger.info(f"Using device: {device}")
            log_gpu_status()
            
            try:
                self.local_summarizer = pipeline(
                    "summarization",
                    model=LOCAL_MODEL_NAME,
                    device=device,
                    token=HUGGINGFACE_API_TOKEN,
                    torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
                    framework='pt'
                )
                logger.info(f"Loaded local summarizer model on {'GPU' if device == 'cuda' else 'CPU'}")
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                # Fallback to a smaller, public model if token is invalid
                try:
                    self.local_summarizer = pipeline(
                        "summarization",
                        model="facebook/bart-base",
                        device=device,
                        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
                        framework='pt'
                    )
                    logger.info(f"Loaded fallback summarizer model on {'GPU' if device == 'cuda' else 'CPU'}")
                except Exception as e:
                    logger.error(f"Failed to load fallback model: {str(e)}")
                    raise

    def _chunk_text(self, text: str, max_length: int = MAX_CHUNK_LENGTH) -> List[str]:
        """Split text into chunks based on sentences"""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length > max_length:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

    def _try_api_summary(self, text: str) -> Optional[str]:
        """Try to get summary from Hugging Face API"""
        if not HUGGINGFACE_API_TOKEN:
            return None
            
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
        try:
            response = requests.post(
                HUGGINGFACE_API_URL,
                headers=headers,
                json={"inputs": text, "parameters": {
                    "min_length": MIN_SUMMARY_LENGTH,
                    "max_length": MAX_SUMMARY_LENGTH,
                    "do_sample": False
                }}
            )
            if response.status_code == 200:
                summary = response.json()[0]["summary_text"]
                return summary
        except Exception as e:
            logger.warning(f"API summarization failed: {str(e)}")
        return None

    def _get_local_summary(self, text: str) -> str:
        """Get summary using local model"""
        self._load_local_model()
        
        try:
            # Get optimization settings
            optimizations = optimize_for_model(LOCAL_MODEL_NAME)
            
            result = self.local_summarizer(
                text,
                min_length=MIN_SUMMARY_LENGTH,
                max_length=MAX_SUMMARY_LENGTH,
                do_sample=False,
                num_beams=optimizations.get('num_beams', 4),
                early_stopping=optimizations.get('early_stopping', True),
                truncation=True
            )
            return result[0]["summary_text"]
        except Exception as e:
            logger.error(f"Local summarization failed: {str(e)}")
            return text[:MAX_SUMMARY_LENGTH] + "..."
        finally:
            # Clear GPU memory after summarization
            clear_gpu_memory()

    def summarize(self, text: str, cache_dir: Optional[str] = None) -> str:
        """
        Main summarization method that tries API first, then falls back to local model
        
        Args:
            text: Text to summarize
            cache_dir: Optional directory to cache results
        
        Returns:
            Summarized text
        """
        if not text.strip():
            return ""

        # Check cache if directory provided
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, f"summary_{hash(text)}.json")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cached = json.load(f)
                        if datetime.now().timestamp() - cached['timestamp'] < 86400:  # 24h cache
                            return cached['summary']
                except Exception:
                    pass

        # Split into chunks if text is too long
        if len(text.split()) > MAX_CHUNK_LENGTH:
            chunks = self._chunk_text(text)
            summaries = []
            
            for chunk in chunks:
                # Try API first
                chunk_summary = self._try_api_summary(chunk)
                if not chunk_summary:
                    # Fallback to local model
                    chunk_summary = self._get_local_summary(chunk)
                summaries.append(chunk_summary)
            
            # Combine chunk summaries
            combined = " ".join(summaries)
            
            # If combined summary is still too long, summarize again
            if len(combined.split()) > MAX_CHUNK_LENGTH:
                final_summary = self.summarize(combined)
            else:
                final_summary = combined
        else:
            # For shorter texts, try API first
            final_summary = self._try_api_summary(text)
            if not final_summary:
                final_summary = self._get_local_summary(text)

        # Cache the result if cache_dir provided
        if cache_dir:
            try:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'summary': final_summary,
                        'timestamp': datetime.now().timestamp()
                    }, f)
            except Exception as e:
                logger.warning(f"Failed to cache summary: {str(e)}")

        return final_summary

    def get_key_points(self, text: str, num_points: int = 5) -> List[str]:
        """Extract key points from the text"""
        summary = self.summarize(text)
        sentences = sent_tokenize(summary)
        return sentences[:min(num_points, len(sentences))]
    
    def extract_key_concepts(self, text: str, num_keywords: int = 10) -> List[str]:
        """
        Extract key concepts from text using multiple methods for better results
        
        Args:
            text: Input text
            num_keywords: Number of keywords to extract
            
        Returns:
            List of key concepts/keywords
        """
        try:
            import yake
            from nltk.tokenize import word_tokenize
            from nltk.corpus import stopwords
            from nltk.stem import WordNetLemmatizer
            import nltk
            
            # Download required NLTK data
            try:
                nltk.data.find('tokenizers/punkt')
                nltk.data.find('corpora/stopwords')
                nltk.data.find('corpora/wordnet')
            except LookupError:
                nltk.download('punkt')
                nltk.download('stopwords')
                nltk.download('wordnet')
            
            # Clean text
            text = re.sub(r'[^\w\s]', ' ', text)
            text = text.lower()
            
            # Initialize YAKE keyword extractor
            kw_extractor = yake.KeywordExtractor(
                lan="en",
                n=2,  # Extract 1-2 word keywords
                dedupLim=0.7,  # Similarity threshold for duplicate removal
                dedupFunc='seqm',
                windowsSize=2,
                top=num_keywords * 2  # Get more keywords than needed for filtering
            )
            
            # Get keywords using YAKE
            keywords = kw_extractor.extract_keywords(text)
            
            # Initialize lemmatizer and get stop words
            lemmatizer = WordNetLemmatizer()
            stop_words = set(stopwords.words('english'))
            
            # Process and filter keywords
            processed_keywords = []
            seen = set()
            
            for keyword, score in keywords:
                # Split multi-word keywords
                words = word_tokenize(keyword)
                
                # Process each word
                processed_words = []
                for word in words:
                    # Lemmatize and check if it's a valid word
                    if (
                        word not in stop_words and
                        len(word) > 2 and  # Skip very short words
                        not word.isnumeric()  # Skip numbers
                    ):
                        lemma = lemmatizer.lemmatize(word)
                        if lemma not in seen:
                            processed_words.append(lemma)
                            seen.add(lemma)
                
                if processed_words:
                    # Join multi-word concepts back together
                    concept = ' '.join(processed_words)
                    if concept not in processed_keywords:
                        processed_keywords.append(concept)
            
            # Return top keywords
            return processed_keywords[:num_keywords]
            
        except Exception as e:
            logger.error(f"Error in keyword extraction: {str(e)}")
            # Fallback to simple word frequency if advanced extraction fails
            words = word_tokenize(text.lower())
            words = [w for w in words if w.isalnum() and len(w) > 2]
            from collections import Counter
            word_freq = Counter(words).most_common(num_keywords)
            return [word for word, _ in word_freq]
    
    def __del__(self):
        """Cleanup GPU memory when object is destroyed"""
        try:
            clear_gpu_memory()
        except Exception:
            pass
