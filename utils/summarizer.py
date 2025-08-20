import logging
import re
from typing import List, Dict, Any, Optional, Set
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string

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
    use_extractive: bool = False,
    remove_fillers: bool = True,
    user_prompt: str = None
) -> str:
    """
    Generate a summary for a cluster of text segments with language-aware preprocessing.
    
    Args:
        cluster: List of text segments with 'text' keys
        language: Language code ('en', 'hi', 'kn', 'te', 'ta', 'ml', 'mr', 'ur')
        use_extractive: Whether to use extractive summarization (faster but less coherent)
        remove_fillers: Whether to remove filler words and stopwords before summarization
        user_prompt: Optional user prompt to guide summarization focus
        
    Returns:
        Generated summary text with improved language-specific preprocessing
    """
    if not cluster:
        return "No content to summarize."
    
    # Combine all text from the cluster
    cluster_text = " ".join([seg.get('text', '') for seg in cluster if seg.get('text')])
    
    if not cluster_text.strip():
        return "No content to summarize."
    
    # If user prompt is provided, enhance the text with prompt context
    if user_prompt and user_prompt.strip():
        # Add user prompt as context to guide summarization
        enhanced_text = f"{cluster_text} [Focus on: {user_prompt}]"
        cluster_text = enhanced_text
    
    # Normalize language code
    language = language.lower()
    if language not in SUPPORTED_LANGUAGES:
        logger.warning(f"Unsupported language: {language}. Defaulting to English.")
        language = 'en'
    
    try:
        # Preprocess text with language-specific cleaning and stopword removal
        processed_text = preprocess_text(
            cluster_text, 
            language=language, 
            remove_fillers=remove_fillers
        )
        
        # For very short texts after preprocessing, just return as is
        if language == 'en':
            try:
                tokens = word_tokenize(processed_text)
            except LookupError:
                try:
                    nltk.download('punkt', quiet=True)
                    tokens = word_tokenize(processed_text)
                except Exception:
                    tokens = processed_text.split()
        else:
            tokens = processed_text.split()
        if len(tokens) < 20:  # Reduced threshold due to stopword removal
            return processed_text if processed_text.strip() else "No meaningful content to summarize."
        
        # Use extractive summarization for non-English or as a fallback
        if use_extractive or language != 'en':
            key_sentences = extract_key_sentences(
                processed_text, 
                num_sentences=min(3, max(1, len(tokens) // 20)),  # Dynamic number of sentences
                language=language
            )
            return ' '.join(key_sentences) if key_sentences else processed_text
        
        # Use abstractive summarization for English
        summarizer = load_summarizer(language)
        if not summarizer:
            logger.warning(f"Falling back to extractive summarization for {language}")
            key_sentences = extract_key_sentences(processed_text, num_sentences=3, language=language)
            return ' '.join(key_sentences)
        
        # Get model config
        config = summarizer['config']
        
        # Split long text into chunks if needed (to avoid token limits)
        max_chunk_length = 1024 if language == 'en' else 768
        if len(processed_text) > max_chunk_length:
            # Simple chunking by sentences
            sentences = sent_tokenize(processed_text) if language == 'en' else processed_text.split('।' if language in ['hi', 'mr'] else '॥')
            chunks = []
            current_chunk = []
            current_length = 0
            
            for sent in sentences:
                sent_tokens = word_tokenize(sent) if language == 'en' else sent.split()
                sent_length = len(sent_tokens)
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
