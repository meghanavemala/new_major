import os
import logging
import re
import gc
from typing import List, Dict, Any, Optional, Set
from transformers import AutoTokenizer, BartForConditionalGeneration
import torch
from torch.cuda.amp import autocast
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string
from .gpu_config import get_device, is_gpu_available, optimize_for_model, clear_gpu_memory, log_gpu_status
from config import load_config

# Enable memory efficient attention globally
os.environ['PYTORCH_TRANSFORMERS_CACHE'] = 'cache'  # Cache models locally
torch.hub.set_dir('cache')  # Set torch hub cache directory

# Supported languages list and a safe stopwords loader to avoid NLTK LookupError
SUPPORTED_LANGUAGES = {'en', 'hi', 'kn', 'te', 'ta', 'ml', 'mr', 'ur'}

def get_stopwords(language: str) -> Set[str]:
    """Return a robust set of stopwords/fillers for the given language.
    Uses NLTK for English (with graceful fallback) and predefined sets for others.
    """
    lang = language.lower()
    if lang == 'en':
        fillers = {
            'oh', 'uh', 'um', 'ah', 'er', 'hmm', 'huh', 'like', 'you know', 'i mean',
            'well', 'so', 'okay', 'right', 'just', 'actually', 'basically', 'literally',
            'really', 'very', 'quite', 'somewhat', 'maybe', 'perhaps', 'probably',
            'sort of', 'kind of', 'type of', 'something', 'anything', 'everything'
        }
        try:
            base = set(stopwords.words('english'))
        except LookupError:
            try:
                nltk.download('stopwords', quiet=True)
                base = set(stopwords.words('english'))
            except Exception:
                base = set()
        return base.union(fillers)

    if lang == 'hi':
        return {
            'तो', 'हैं', 'है', 'हूं', 'था', 'थे', 'थी', 'हो', 'ही', 'और', 'या', 'लेकिन', 'पर', 'में', 'से', 'को', 'ने',
            'अच्छा', 'ठीक', 'बस', 'क्या', 'वो', 'ये', 'वह', 'यह', 'उस', 'इस', 'अरे', 'अब', 'फिर', 'भी', 'थोड़ा', 'जैसे',
            'वगैरह', 'वगैर', 'आदि', 'इत्यादि', 'वाह', 'ओह', 'ओके', 'ओहो', 'हाँ', 'नहीं'
        }
    if lang == 'kn':
        return {
            'ಹೌದು', 'ಇಲ್ಲ', 'ಅದು', 'ಇದು', 'ಅವರು', 'ನಾನು', 'ನಾವು', 'ನೀವು', 'ಅವನು', 'ಅವಳು', 'ಅದೇ', 'ಇದೇ', 'ಆದರೆ', 'ಮತ್ತು', 'ಅಥವಾ',
            'ಅಂತ', 'ಎಂಬ', 'ಎಂದು', 'ಎಲ್ಲ', 'ಎಲ್ಲಾ', 'ಏನು', 'ಯಾರು', 'ಯಾವ', 'ಹಾಗೆ', 'ಹೇಗೆ', 'ಎಲ್ಲಿ', 'ಯಾವಾಗ', 'ಏಕೆ', 'ಓಹ್', 'ಅಯ್ಯೋ'
        }
    if lang == 'te':
        return {
            'అవును', 'కాదు', 'అది', 'ఇది', 'వారు', 'నేను', 'మనం', 'మీరు', 'అతను', 'ఆమె', 'కానీ', 'మరియు', 'లేదా',
            'అంటే', 'అని', 'అన్ని', 'ఏమిటి', 'ఎవరు', 'ఏ', 'అలా', 'ఎలా', 'ఎక్కడ', 'ఎప్పుడు', 'ఎందుకు', 'ఓహ్', 'అయ್యೋ'
        }
    if lang == 'ta':
        return {
            'ஆம்', 'இல்லை', 'அது', 'இது', 'அவர்கள்', 'நான்', 'நாங்கள்', 'நீங்கள்', 'அவன்', 'அவள்', 'ஆனால்', 'மற்றும்', 'அல்லது',
            'என்று', 'எல்லாம்', 'என்ன', 'யார்', 'எந்த', 'அப்படி', 'எப்படி', 'எங்கே', 'எப்போது', 'ஏன்', 'ஓ', 'அடடா'
        }
    if lang == 'ml':
        return {
            'അതെ', 'ഇല്ല', 'അത്', 'ഇത്', 'അവർ', 'ഞാൻ', 'നാം', 'നിങ്ങൾ', 'അവൻ', 'അവൾ', 'പക്ഷേ', 'ഒപ്പം', 'അല്ലെങ്കിൽ',
            'എന്ന്', 'എല്ലാം', 'എന്ത്', 'ആര്', 'ഏത്', 'അങ്ങനെ', 'എങ്ങനെ', 'എവിടെ', 'എപ്പോൾ', 'എന്തുകൊണ്ട്', 'ഓ', 'അയ്യോ'
        }
    if lang == 'mr':
        return {
            'होय', 'नाही', 'ते', 'हे', 'मी', 'आम्ही', 'तुम्ही', 'तो', 'ती', 'पण', 'आणि', 'किंवा',
            'असे', 'सर्व', 'काय', 'कोण', 'कोणता', 'असं', 'कसं', 'कुठे', 'कधी', 'का', 'ओह', 'अरेरे'
        }
    if lang == 'ur':
        return {
            'ہاں', 'نہیں', 'وہ', 'یہ', 'میں', 'ہم', 'تم', 'لیکن', 'اور', 'یا', 'کہ', 'سب', 'کیا', 'کون', 'کونسا',
            'ایسا', 'کیسے', 'کہاں', 'کب', 'کیوں', 'اوہ', 'اوہو'
        }
    # Default fallback
    return set()

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Download required NLTK data
# try:
#     nltk.data.find('tokenizers/punkt')
#     nltk.data.find('corpora/stopwords')
# except LookupError:
#     logger.info("Downloading NLTK data...")
#     nltk.download('punkt')
#     nltk.download('stopwords')

# Enhanced summarizer
from .enhanced_summarizer import EnhancedSummarizer

# Initialize enhanced summarizer
ENHANCED_SUMMARIZER = None

def get_summarizer():
    global ENHANCED_SUMMARIZER
    if ENHANCED_SUMMARIZER is None:
        ENHANCED_SUMMARIZER = EnhancedSummarizer()
    return ENHANCED_SUMMARIZER

# Load config to check for Low VRAM mode
app_config = load_config()
LOW_VRAM_MODE = app_config.get("LOW_VRAM_MODE", True)

if LOW_VRAM_MODE:
    logger.info("Low VRAM mode is enabled. Using distilbart for English summarization.")
    english_model_config = {
        'model_name': 'sshleifer/distilbart-cnn-12-6',
        'min_length': 30,
        'max_length': 80,  # Shorter length for a smaller model
        'repetition_penalty': 2.0,
        'length_penalty': 1.0,
        'num_beams': 2,      # Fewer beams to conserve memory
    }
else:
    logger.info("Low VRAM mode is disabled. Using pegasus-large for English summarization.")
    english_model_config = {
        'model_name': 'google/pegasus-large',
        'min_length': 50,
        'max_length': 130,
        'repetition_penalty': 2.5,
        'length_penalty': 1.0,
        'num_beams': 4,
    }

# Model configurations
MODEL_CONFIGS = {
    'en': english_model_config,
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

def preprocess_text(text: str, language: str = 'en', remove_fillers: bool = True) -> str:
    """
    Preprocess text before summarization with language-specific cleaning.
    
    Args:
        text: Input text to preprocess
        language: Language code (en, hi, kn, te, ta, ml, mr, ur)
        remove_fillers: Whether to remove filler words and stopwords
        
    Returns:
        Preprocessed text with noise, stopwords, and fillers removed
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs, mentions, and hashtags
    text = re.sub(r'https?://\S+|www\.\S+|@\w+|#\w+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove special characters and numbers (language-specific handling)
    if language == 'en':
        # Keep common punctuation for English
        text = re.sub(r'[^\w\s\'\-.,!?]', '', text)
    else:
        # For other languages, be more permissive with Unicode characters
        text = re.sub(r'[\u0900-\u097F\u0C80-\u0CFF\u0980-\u09FF\u0A80-\u0AFF\u0B00-\u0B7F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F\u0D80-\u0DFF\u0E80-\u0EFF\u0F00-\u0FFF\w\s\'\-.,!?]', '', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    # Remove stopwords and fillers if requested
    if remove_fillers:
        # Get language-specific stopwords
        lang = language if language in SUPPORTED_LANGUAGES else 'en'
        stop_words = get_stopwords(lang)
        
        # Tokenize and remove stopwords
        if lang == 'en':
            try:
                tokens = word_tokenize(text)
            except LookupError:
                try:
                    nltk.download('punkt', quiet=True)
                    tokens = word_tokenize(text)
                except Exception:
                    tokens = text.split()
        else:
            tokens = text.split()
        tokens = [token for token in tokens if token.lower() not in stop_words]
        text = ' '.join(tokens)
    
    return text.strip()

def extract_key_sentences(text: str, num_sentences: int = 3, language: str = 'en') -> List[str]:
    """
    Extract key sentences using a combination of word frequency and sentence position.
    
    Args:
        text: Input text to extract sentences from
        num_sentences: Maximum number of sentences to return
        language: Language code for processing
        
    Returns:
        List of key sentences in their original order
    """
    if not text or not text.strip():
        return []
    
    try:
        # Get language-specific settings
        lang = language if language in EXTENDED_STOPWORDS else 'en'
        stop_words = EXTENDED_STOPWORDS[lang]['stopwords']
        
        # Tokenize into sentences (language-specific if possible)
        try:
            sentences = sent_tokenize(text, language='english' if lang == 'en' else lang)
        except:
            # Fallback for unsupported languages
            sentence_enders = r'[.!?।॥]\s*'
            sentences = [s.strip() for s in re.split(sentence_enders, text) if s.strip()]
        
        if not sentences:
            return []
        
        # Simple scoring based on word frequency (excluding stopwords)
        words = []
        for sent in sentences:
            # Tokenize words (language-specific if possible)
            try:
                words.extend([w.lower() for w in word_tokenize(sent) if w.lower() not in stop_words and w not in string.punctuation])
            except:
                # Fallback for unsupported languages
                words.extend([w for w in sent.split() if w.lower() not in stop_words and w not in string.punctuation])
        
        # Calculate word frequencies
        word_freq = {}
        for word in words:
            if len(word) > 2:  # Ignore very short words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # If no meaningful words found, return first few sentences
        if not word_freq:
            return sentences[:min(num_sentences, len(sentences))]
        
        # Score sentences based on word frequency and position
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            # Position-based score (favor beginning and end of text)
            position_score = 1.0 - (abs((i / len(sentences)) - 0.5) * 1.8)  # 0.1-1.0 range
            
            # Content-based score
            content_score = 0
            try:
                sent_words = [w.lower() for w in word_tokenize(sentence)]
            except:
                sent_words = sentence.lower().split()
                
            for word in sent_words:
                content_score += word_freq.get(word, 0)
            
            # Normalize by sentence length (prefer medium-length sentences)
            length_penalty = min(1.0, len(sent_words) / 20)  # Favor sentences around 20 words
            
            # Combine scores
            sentence_scores[i] = (content_score * length_penalty) + (position_score * 0.5)
        
        # Get top N sentences, ensuring we don't exceed available sentences
        num_sentences = min(num_sentences, len(sentences))
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences]
        
        # Sort by original position to maintain coherence
        top_sentences = sorted([i[0] for i in top_sentences])
        
        return [sentences[i] for i in top_sentences if i < len(sentences)]
    
    except Exception as e:
        logger.warning(f"Error in extract_key_sentences: {e}")
        # Fallback: return first few sentences
        return text[:500].split('. ')[:num_sentences]

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
            
            # Get optimized device and settings
            device = get_device()
            optimizations = optimize_for_model(model_name)
            
            logger.info(f"Using device: {device}")
            log_gpu_status()
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = BartForConditionalGeneration.from_pretrained(model_name).to(device)
            model.to(torch.float16 if device == 'cuda' else torch.float32)
            
            # Store model and tokenizer
            SUMMARIZERS[language] = {
                'model': model,
                'tokenizer': tokenizer,
                'device': device,
                'config': MODEL_CONFIGS[language],
                'optimizations': optimizations
            }
            logger.info(f"{language.capitalize()} summarization model loaded successfully on {'GPU' if device == 'cuda' else 'CPU'}.")
            
        except Exception as e:
            logger.error(f"Error loading {language} summarization model: {e}")
            return None
    
    return SUMMARIZERS[language]

def summarize_cluster(
    cluster: List[Dict[str, Any]],
    language: str = 'en',
    use_extractive: bool = False,
    remove_fillers: bool = True,
    user_prompt: str = None
) -> str:
    """
    Generate a memory-efficient summary for a cluster of text segments.
    """
    if not cluster:
        return "No content to summarize."

    # 1. Combine text and preprocess
    cluster_text = " ".join([seg.get('text', '') for seg in cluster if seg.get('text')])
    if user_prompt:
        cluster_text = f"{user_prompt} [CONTEXT] {cluster_text}"

    processed_text = preprocess_text(cluster_text, language, remove_fillers)
    if len(processed_text.split()) < 20:
        return processed_text if processed_text.strip() else "No meaningful content to summarize."

    # 2. Handle extractive summarization
    if use_extractive or language != 'en':
        key_sentences = extract_key_sentences(processed_text, num_sentences=3, language=language)
        return ' '.join(key_sentences) if key_sentences else processed_text

    # 3. Abstractive summarization with memory management
    try:
        clear_gpu_memory()
        summarizer_payload = load_summarizer(language)
        if not summarizer_payload:
            raise RuntimeError(f"Failed to load summarizer for {language}")

        model = summarizer_payload['model']
        tokenizer = summarizer_payload['tokenizer']
        device = summarizer_payload['device']
        config = summarizer_payload['config']

        # 4. Chunk the text for memory efficiency
        max_chunk_length = 512  # Model's max token length
        sentences = sent_tokenize(processed_text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(tokenizer.encode(current_chunk + ' ' + sentence)) > max_chunk_length:
                chunks.append(current_chunk)
                current_chunk = sentence
            else:
                current_chunk += ' ' + sentence
        if current_chunk:
            chunks.append(current_chunk)

        # 5. Summarize each chunk
        chunk_summaries = []
        for chunk in chunks:
            try:
                inputs = tokenizer(chunk, return_tensors="pt", max_length=1024, truncation=True).to(device)
                
                with torch.no_grad(), autocast(enabled=(device=='cuda')):
                    summary_ids = model.generate(
                        inputs['input_ids'],
                        num_beams=config.get('num_beams', 2),
                        max_length=config.get('max_length', 150),
                        min_length=config.get('min_length', 30),
                        length_penalty=config.get('length_penalty', 1.5),
                        repetition_penalty=config.get('repetition_penalty', 2.0),
                        early_stopping=True
                    )
                summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                chunk_summaries.append(summary)
            finally:
                del inputs, summary_ids
                clear_gpu_memory()

        # 6. Combine and finalize
        combined_summary = ' '.join(chunk_summaries)
        if len(combined_summary.split()) > config.get('max_length', 150):
             # Final summarization of the combined summaries if too long
             return summarize_cluster([{'text': combined_summary}], language, False, False, None)

        return combined_summary

    except Exception as e:
        logger.error(f"Error in abstractive summarization: {e}", exc_info=True)
        key_sentences = extract_key_sentences(processed_text, num_sentences=3, language=language)
        return ' '.join(key_sentences) if key_sentences else processed_text[:500] + "..."
    finally:
        clear_gpu_memory()
    
