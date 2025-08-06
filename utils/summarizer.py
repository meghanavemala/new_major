import logging
from typing import List, Dict, Any, Optional
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    logger.info("Downloading NLTK data...")
    nltk.download('punkt')
    nltk.download('stopwords')

# Initialize models as None. They will be loaded on first use.
SUMMARIZERS = {
    'en': None,  # English
    'hi': None,  # Hindi
    'kn': None   # Kannada
}

# Model configurations for different languages
MODEL_CONFIGS = {
    'en': {
        'model_name': 'facebook/bart-large-cnn',
        'min_length': 30,
        'max_length': 130,
        'repetition_penalty': 2.5,
        'length_penalty': 1.0,
        'num_beams': 4,
    },
    'hi': {
        'model_name': 'csebuetnlp/mT5_multilingual_XLSum',
        'min_length': 30,
        'max_length': 100,
        'repetition_penalty': 2.5,
        'length_penalty': 1.0,
        'num_beams': 4,
    },
    'kn': {
        'model_name': 'csebuetnlp/mT5_multilingual_XLSum',
        'min_length': 30,
        'max_length': 100,
        'repetition_penalty': 2.5,
        'length_penalty': 1.0,
        'num_beams': 4,
    }
}

def preprocess_text(text: str, language: str = 'en') -> str:
    """Preprocess text before summarization."""
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove special characters and numbers
    if language == 'en':
        text = ''.join([char for char in text if char.isalnum() or char.isspace() or char in ',.!?'])
    
    return text

def extract_key_sentences(text: str, num_sentences: int = 3, language: str = 'en') -> List[str]:
    """Extract key sentences using a simple heuristic."""
    try:
        # Tokenize into sentences
        sentences = sent_tokenize(text, language='english' if language == 'en' else language)
        
        # Simple scoring based on word frequency
        words = [word.lower() for word in word_tokenize(text) 
                if word.lower() not in stopwords.words('english' if language == 'en' else language) 
                and word not in string.punctuation]
        
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Score sentences based on word frequency
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            for word in word_tokenize(sentence.lower()):
                if word in word_freq:
                    sentence_scores[i] = sentence_scores.get(i, 0) + word_freq[word]
        
        # Get top N sentences
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences]
        top_sentences = sorted([i[0] for i in top_sentences])
        
        return [sentences[i] for i in top_sentences]
    except Exception as e:
        logger.warning(f"Error in extract_key_sentences: {e}")
        return text[:500].split('. ')[:3]  # Fallback: first few sentences

def load_summarizer(language: str = 'en') -> Any:
    """Load the appropriate summarization model for the specified language."""
    global SUMMARIZERS
    
    if language not in SUMMARIZERS:
        logger.warning(f"Unsupported language: {language}. Defaulting to English.")
        language = 'en'
    
    if SUMMARIZERS[language] is None:
        try:
            model_name = MODEL_CONFIGS[language]['model_name']
            logger.info(f"Loading {language} summarization model: {model_name}")
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Using device: {device}")
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
            
            # Create pipeline
            SUMMARIZERS[language] = {
                'pipeline': pipeline(
                    'summarization',
                    model=model,
                    tokenizer=tokenizer,
                    device=0 if device == 'cuda' else -1,
                    framework='pt'
                ),
                'config': MODEL_CONFIGS[language]
            }
            logger.info(f"{language.capitalize()} summarization model loaded successfully.")
            
        except Exception as e:
            logger.error(f"Error loading {language} summarization model: {e}")
            return None
    
    return SUMMARIZERS[language]

def summarize_cluster(
    cluster: List[Dict[str, Any]], 
    language: str = 'en',
    use_extractive: bool = False
) -> str:
    """
    Generate a summary for a cluster of text segments.
    
    Args:
        cluster: List of text segments with 'text' keys
        language: Language code ('en', 'hi', 'kn')
        use_extractive: Whether to use extractive summarization (faster but less coherent)
        
    Returns:
        Generated summary text
    """
    if not cluster:
        return "No content to summarize."
    
    # Combine all text from the cluster
    cluster_text = " ".join([seg.get('text', '') for seg in cluster])
    
    if not cluster_text.strip():
        return "No content to summarize."
    
    language = language.lower()
    if language not in ['en', 'hi', 'kn']:
        logger.warning(f"Unsupported language: {language}. Defaulting to English.")
        language = 'en'
    
    try:
        # Preprocess text
        cluster_text = preprocess_text(cluster_text, language)
        
        # For very short texts, just return as is
        if len(word_tokenize(cluster_text)) < 30:
            return cluster_text
        
        # Use extractive summarization for non-English or as a fallback
        if use_extractive or language in ['hi', 'kn']:
            key_sentences = extract_key_sentences(cluster_text, num_sentences=3, language=language)
            return ' '.join(key_sentences)
        
        # Use abstractive summarization for English
        summarizer = load_summarizer(language)
        if not summarizer:
            raise Exception(f"Failed to load {language} summarization model")
        
        # Get model config
        config = summarizer['config']
        
        # Split long text into chunks if needed (to avoid token limits)
        max_chunk_length = 1024 if language == 'en' else 768
        if len(cluster_text) > max_chunk_length:
            # Simple chunking by sentences
            sentences = sent_tokenize(cluster_text)
            chunks = []
            current_chunk = []
            current_length = 0
            
            for sent in sentences:
                sent_length = len(word_tokenize(sent))
                if current_length + sent_length > max_chunk_length and current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                current_chunk.append(sent)
                current_length += sent_length
            
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            # Summarize each chunk
            chunk_summaries = []
            for chunk in chunks:
                summary = summarizer['pipeline'](
                    chunk,
                    min_length=config['min_length'],
                    max_length=config['max_length'],
                    repetition_penalty=config['repetition_penalty'],
                    length_penalty=config['length_penalty'],
                    num_beams=config['num_beams'],
                    truncation=True
                )
                chunk_summaries.append(summary[0]['summary_text'])
            
            # Combine chunk summaries
            combined_summary = ' '.join(chunk_summaries)
            
            # Final summary of combined chunk summaries if still too long
            if len(word_tokenize(combined_summary)) > 100:
                final_summary = summarizer['pipeline'](
                    combined_summary,
                    min_length=config['min_length'],
                    max_length=config['max_length'],
                    repetition_penalty=config['repetition_penalty'],
                    length_penalty=config['length_penalty'],
                    num_beams=config['num_beams'],
                    truncation=True
                )
                return final_summary[0]['summary_text']
            return combined_summary
        
        else:
            # Process in one go if text is short enough
            summary = summarizer['pipeline'](
                cluster_text,
                min_length=config['min_length'],
                max_length=config['max_length'],
                repetition_penalty=config['repetition_penalty'],
                length_penalty=config['length_penalty'],
                num_beams=config['num_beams'],
                truncation=True
            )
            return summary[0]['summary_text']
    
    except Exception as e:
        logger.error(f"Error in summarize_cluster: {e}", exc_info=True)
        # Fallback to extractive summarization
        try:
            key_sentences = extract_key_sentences(cluster_text, num_sentences=3, language=language)
            return ' '.join(key_sentences)
        except Exception as e2:
            logger.error(f"Fallback summarization also failed: {e2}")
            return cluster_text[:500] + "..."  # Return first 500 chars as fallback
